#!/usr/bin/env python3
"""
Exp B: Case Type Recovery — train a classifier to predict case_type from
available features (scheme_description text, amount, industry), then use
PREDICTED case_type in the ranking GNN instead of ground truth or zero.

Three-way comparison:
  Oracle:   Ground-truth case_type  (Precision@5 = 0.9722)
  Predicted: Classifier-inferred    (test acc: 77.1%, MRR@5 recovery: 75.7%)
  Zeroed:   No case_type at all     (Precision@5 = 0.8207)
"""

import json, os, sys, math, random, warnings, gc
from collections import Counter, defaultdict
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings('ignore')

# ============================================================
# CONFIG — paths via environment variables
# ============================================================
PATHS = {
    'case_graphs_dir': os.environ.get('CASE_GRAPHS_DIR', './data/case_graphs'),
    'enterprise_subgraphs_dir': os.environ.get('ENTERPRISE_SUBGRAPHS_DIR', './data/enterprise_subgraphs'),
    'output_dir': os.environ.get('OUTPUT_DIR', './results'),
    'model_dir': os.environ.get('MODEL_DIR', './final_models'),
    'split_meta_path': os.environ.get('SPLIT_META_PATH', './final_results/split_meta_v9.json'),
    'vocab_path': os.environ.get('VOCAB_PATH', './data/node_vocabs.json'),
    'master_results_dir': os.environ.get('MASTER_RESULTS_DIR', './final_results'),
}

os.makedirs(PATHS['master_results_dir'], exist_ok=True)
os.makedirs(PATHS['output_dir'], exist_ok=True)

BATCH_SIZE = 64
LR = 1e-3
EPOCHS = 100
EARLY_STOP_PATIENCE = 20
SEED = 42

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ============================================================
# LOAD VOCAB & META
# ============================================================
with open(PATHS['vocab_path']) as f:
    VOCABS = json.load(f)

def build_vocab_map(items):
    m = {'<UNK>': 0}
    for i, item in enumerate(items):
        m[item] = i + 1
    return m

VOCAB_MAP = {
    'industry': build_vocab_map(VOCABS['industry']),
    'case_type': build_vocab_map(VOCABS['case_type']),
}

with open(PATHS['split_meta_path']) as f:
    meta = json.load(f)

train_ids = meta['train_ids']
val_ids = meta['val_ids']
test_ids = meta['test_ids']
case_data = meta['case_data']
unique_types = meta['unique_types']  # sub_type level (9 classes)
case_type_list = VOCABS['case_type']  # 6 classes
type_to_label = {t: i for i, t in enumerate(unique_types)}
case_type_to_idx = {t: i for i, t in enumerate(case_type_list)}

print(f"Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")
print(f"Sub-types (for ranking): {len(unique_types)} classes")
print(f"Case types (for classifier): {len(case_type_list)} classes: {case_type_list}")

# ============================================================
# STEP 1: Extract features for case_type classifier
# ============================================================
print("\n[Step 1] Extracting features for case_type classifier...")

cg_dir = PATHS['case_graphs_dir']
all_fnames = set(os.listdir(cg_dir))

# Build ID -> file mapping
id_to_fname = {}
for cid in train_ids + val_ids + test_ids:
    for cand in [cid + ".json", cid.lower() + ".json",
                 cid.replace("CASE_", "case_").lower() + ".json"]:
        if cand in all_fnames:
            id_to_fname[cid] = cand
            break

print(f"  Mapped {len(id_to_fname)}/{len(train_ids)+len(val_ids)+len(test_ids)} IDs to files")

def extract_features(cid):
    """Extract features from a case graph for case_type classification."""
    fname = id_to_fname.get(cid)
    if fname is None:
        return None
    with open(os.path.join(cg_dir, fname), encoding='utf-8') as f:
        raw = json.load(f)
    graph = raw.get('graph', raw)

    scheme_desc = ""
    industry_val = 0
    amount_val = 0.0
    amount_bucket = ""
    ct_ground_truth = ""

    for node in graph.get('nodes', []):
        props = node.get('properties', {})
        nt = node.get('type', '')
        if nt == 'ViolationEvent':
            scheme_desc = props.get('scheme_description', '')
            ct_ground_truth = props.get('case_type', '')
        elif nt == 'Company':
            industry_val = VOCAB_MAP['industry'].get(props.get('industry', ''), 0)

    cd = case_data.get(cid, {})
    amount_val = cd.get('amount', 0.0)
    amount_bucket = cd.get('amount_bucket', '')
    ct_from_meta = cd.get('case_type', ct_ground_truth)

    # Use meta case_type as ground truth (more reliable)
    ct = ct_from_meta or ct_ground_truth
    if ct not in case_type_to_idx:
        ct = '其他'

    return {
        'cid': cid,
        'scheme_description': scheme_desc,
        'industry': industry_val,
        'amount': amount_val,
        'amount_bucket': amount_bucket,
        'case_type': ct,
        'case_type_label': case_type_to_idx[ct],
    }

# Extract all features
all_features = {}
for cid in train_ids + val_ids + test_ids:
    feats = extract_features(cid)
    if feats:
        all_features[cid] = feats

train_feats = [all_features[c] for c in train_ids if c in all_features]
val_feats = [all_features[c] for c in val_ids if c in all_features]
test_feats = [all_features[c] for c in test_ids if c in all_features]
print(f"  Train: {len(train_feats)}, Val: {len(val_feats)}, Test: {len(test_feats)}")

# Show class distribution
train_ct_dist = Counter(f['case_type'] for f in train_feats)
print(f"  Train case_type distribution: {dict(train_ct_dist)}")

# ============================================================
# STEP 2: Build text features (TF-IDF over scheme_description)
# ============================================================
print("\n[Step 2] Building text features...")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Use jieba for Chinese word segmentation if available, else char-level
try:
    import jieba
    def tokenize(text):
        return list(jieba.cut(text))
    print("  Using jieba tokenization")
except ImportError:
    def tokenize(text):
        chars = list(text.replace(' ', ''))
        unigrams = chars
        bigrams = [chars[i]+chars[i+1] for i in range(len(chars)-1)]
        return unigrams + bigrams
    print("  Using char bigram tokenization (jieba not available)")

train_texts = [f['scheme_description'] for f in train_feats]
val_texts = [f['scheme_description'] for f in val_feats]
test_texts = [f['scheme_description'] for f in test_feats]

vectorizer = TfidfVectorizer(
    tokenizer=tokenize,
    max_features=5000,
    sublinear_tf=True,
    analyzer='word',
)
X_train_text = vectorizer.fit_transform(train_texts)
X_val_text = vectorizer.transform(val_texts)
X_test_text = vectorizer.transform(test_texts)
print(f"  Text feature dim: {X_train_text.shape[1]}")

# Numeric features
def build_numeric_features(feats_list):
    feats = []
    for f in feats_list:
        amount_log = math.log10(f['amount'] + 1) if f['amount'] > 0 else 0.0
        amount_bucket_map = {'金额_小额': 0, '金额_中额': 1, '金额_大额': 2, '金额_超大额': 3}
        ab = amount_bucket_map.get(f['amount_bucket'], 0)
        feats.append([amount_log, ab, f['industry']])
    return np.array(feats, dtype=np.float32)

X_train_num = build_numeric_features(train_feats)
X_val_num = build_numeric_features(val_feats)
X_test_num = build_numeric_features(test_feats)

# Normalize numeric features
num_mean = X_train_num.mean(axis=0)
num_std = X_train_num.std(axis=0) + 1e-8
X_train_num = (X_train_num - num_mean) / num_std
X_val_num = (X_val_num - num_mean) / num_std
X_test_num = (X_test_num - num_mean) / num_std

print(f"  Numeric feature dim: {X_train_num.shape[1]}")

y_train = np.array([f['case_type_label'] for f in train_feats])
y_val = np.array([f['case_type_label'] for f in val_feats])
y_test = np.array([f['case_type_label'] for f in test_feats])

# ============================================================
# STEP 3: Train classifier
# ============================================================
print("\n[Step 3] Training case_type classifier...")

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC
from scipy.sparse import hstack

# Combine text + numeric
X_train_all = hstack([X_train_text, X_train_num])
X_val_all = hstack([X_val_text, X_val_num])
X_test_all = hstack([X_test_text, X_test_num])

classifiers = {
    'LogisticRegression': LogisticRegression(max_iter=1000, C=1.0),
    'LinearSVC': LinearSVC(max_iter=2000, C=1.0, dual=False),
    'RandomForest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=SEED),
    'GradientBoosting': GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=SEED),
}

best_clf = None
best_val_acc = 0
best_name = ""

for name, clf in classifiers.items():
    clf.fit(X_train_all, y_train)
    val_pred = clf.predict(X_val_all)
    val_acc = accuracy_score(y_val, val_pred)
    print(f"  {name:20s}  Val Acc: {val_acc:.4f}")
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_clf = clf
        best_name = name

print(f"\n  Best: {best_name} (Val Acc: {best_val_acc:.4f})")

# Evaluate on test
test_pred = best_clf.predict(X_test_all)
test_acc = accuracy_score(y_test, test_pred)
test_f1 = f1_score(y_test, test_pred, average='macro')
print(f"  Test Acc: {test_acc:.4f}, Test F1 (macro): {test_f1:.4f}")
print(f"\n  Classification Report (Test):")
print(classification_report(y_test, test_pred, labels=range(len(case_type_list)), target_names=case_type_list, zero_division=0))

# ============================================================
# STEP 4: Generate predictions for ALL samples
# ============================================================
print("\n[Step 4] Generating predictions for all samples...")

all_texts = [f['scheme_description'] for f in all_features.values()]
all_cids = [f['cid'] for f in all_features.values()]
X_all_text = vectorizer.transform(all_texts)

all_num = build_numeric_features(list(all_features.values()))
all_num = (all_num - num_mean) / num_std
X_all_combined = hstack([X_all_text, all_num])

all_preds = best_clf.predict(X_all_combined)
all_probs = best_clf.predict_proba(X_all_combined)

predicted_case_type = {}
for cid, pred_idx, probs in zip(all_cids, all_preds, all_probs):
    predicted_case_type[cid] = {
        'predicted': case_type_list[pred_idx],
        'predicted_idx': int(pred_idx),
        'confidence': float(probs.max()),
        'probs': {case_type_list[i]: float(p) for i, p in enumerate(probs)},
    }

# Save predictions
pred_path = os.path.join(PATHS['master_results_dir'], 'case_type_predictions.json')
with open(pred_path, 'w') as f:
    json.dump(predicted_case_type, f, ensure_ascii=False, indent=2)
print(f"  Saved predictions to {pred_path}")

for split_name, ids in [('train', train_ids), ('val', val_ids), ('test', test_ids)]:
    correct = sum(1 for cid in ids if cid in all_features and cid in predicted_case_type
                  and predicted_case_type[cid]['predicted'] == all_features[cid]['case_type'])
    total = sum(1 for cid in ids if cid in all_features and cid in predicted_case_type)
    print(f"  {split_name}: {correct}/{total} = {correct/total*100:.2f}%")

# ============================================================
# STEP 5: Run ranking GNN with PREDICTED case_type
# ============================================================
print("\n[Step 5] Running ranking GNN with PREDICTED case_type...")

# Import master_experiments from same directory
_ME_DIR = os.path.dirname(os.path.abspath(__file__))
if _ME_DIR not in sys.path:
    sys.path.insert(0, _ME_DIR)
import master_experiments as me

# Save original function
original_json_graph_to_data = me.json_graph_to_data

# Create modified version that uses predicted case_type
def predicted_json_graph_to_data(graph_dict, graph_id=None, label=None, max_nodes=None):
    """Same as original, but uses predicted case_type for ViolationEvent nodes."""
    nodes = graph_dict.get('nodes', [])
    edges = graph_dict.get('edges', [])

    if max_nodes and len(nodes) > max_nodes:
        degree = defaultdict(int)
        nids = {n['id'] for n in nodes}
        for e in edges:
            if e['source'] in nids and e['target'] in nids:
                degree[e['source']] += 1; degree[e['target']] += 1
        nodes = sorted(nodes, key=lambda n: degree.get(n['id'], 0), reverse=True)[:max_nodes]
        keep_ids = {n['id'] for n in nodes}
        edges = [e for e in edges if e['source'] in keep_ids and e['target'] in keep_ids]

    degree = defaultdict(int)
    nids = {n['id'] for n in nodes}
    for e in edges:
        if e['source'] in nids and e['target'] in nids:
            degree[e['source']] += 1; degree[e['target']] += 1

    text_parts = []
    node_id_to_idx = {n['id']: i for i, n in enumerate(nodes)}
    node_types = torch.zeros(len(nodes), dtype=torch.long)
    node_attrs = torch.zeros(len(nodes), 6, dtype=torch.float32)

    # Get predicted case_type for this graph
    pred_ct = predicted_case_type.get(graph_id, {})
    pred_ct_idx = pred_ct.get('predicted_idx', 0)  # default to 0 (其他)

    for i, n in enumerate(nodes):
        nt = n.get('type', 'Company')
        node_types[i] = me.node_type_to_idx.get(nt, 0)
        props = n.get('properties', {})
        deg = degree[n['id']]
        db = me.degree_to_bucket(deg)

        if nt == 'Company':
            node_attrs[i, 0] = me.VOCAB_MAP['industry'].get(props.get('industry', ''), 0)
            node_attrs[i, 1] = me.VOCAB_MAP['province'].get(props.get('province', ''), 0)
            node_attrs[i, 2] = me.VOCAB_MAP['role'].get(props.get('role', ''), 0)
            node_attrs[i, 3] = db; node_attrs[i, 4] = 0.0; node_attrs[i, 5] = 0.0
        elif nt == 'Person':
            node_attrs[i, 0] = 0; node_attrs[i, 1] = 0
            node_attrs[i, 2] = me.VOCAB_MAP['role'].get(props.get('role', ''), 0)
            node_attrs[i, 3] = db; node_attrs[i, 4] = 0.0; node_attrs[i, 5] = 0.0
        elif nt == 'ViolationEvent':
            # USE PREDICTED case_type instead of ground truth or zero
            node_attrs[i, 0] = pred_ct_idx + 1  # +1 because 0 = <UNK>
            node_attrs[i, 1] = me.VOCAB_MAP['violation_item'].get(props.get('violation_item', ''), 0)
            node_attrs[i, 2] = 0; node_attrs[i, 3] = db
            node_attrs[i, 4] = 0.0; node_attrs[i, 5] = 0.0
            desc = props.get('scheme_description', '')
            if desc: text_parts.append(desc)
        elif nt == 'FinancialFlow':
            node_attrs[i, 0] = me.VOCAB_MAP['flow_type'].get(props.get('type', ''), 0)
            node_attrs[i, 1] = 0; node_attrs[i, 2] = 0; node_attrs[i, 3] = db
            raw_val = props.get('value', None)
            if raw_val is not None and raw_val > 0:
                node_attrs[i, 4] = (math.log10(raw_val + 1) - me.LOG_VALUE_MEAN) / me.LOG_VALUE_STD
            else:
                node_attrs[i, 4] = 0.0
            node_attrs[i, 5] = 0.0
        else:
            node_attrs[i, 0] = 0; node_attrs[i, 1] = 0; node_attrs[i, 2] = 0
            node_attrs[i, 3] = db; node_attrs[i, 4] = 0.0; node_attrs[i, 5] = 0.0

    if edges:
        src_list, dst_list, etype_list = [], [], []
        for e in edges:
            src = node_id_to_idx.get(e['source'])
            dst = node_id_to_idx.get(e['target'])
            if src is not None and dst is not None:
                src_list.append(src); dst_list.append(dst)
                et = me.edge_type_to_idx.get(e.get('relation', 'unknown'), 0)
                etype_list.append(et)
                src_list.append(dst); dst_list.append(src)
                etype_list.append(et)
        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        edge_types = torch.tensor(etype_list, dtype=torch.long)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_types = torch.zeros(0, dtype=torch.long)

    return me.GraphData(
        node_types=node_types, edge_index=edge_index, edge_types=edge_types,
        node_attrs=node_attrs, graph_id=graph_id, label=label,
        text=' '.join(text_parts), metadata={}
    )

# Override the function
me.json_graph_to_data = predicted_json_graph_to_data

# Clear cache to force reload with new function
me._data_cache = {}

print("=" * 70)
print("EXP B: Ranking GNN with PREDICTED case_type")
print(f"Classifier: {best_name}, Test Acc: {test_acc:.4f}")
print("=" * 70)

try:
    results = me.run_pure_gnn_baseline(seed=SEED)
except Exception as e:
    print(f"ERROR running ranking: {e}")
    import traceback
    traceback.print_exc()
    results = None

# ============================================================
# STEP 6: Compare all three settings
# ============================================================
print("\n" + "=" * 70)
print("EXP B: CASE TYPE RECOVERY COMPARISON")
print("=" * 70)

# Oracle: with ground-truth case_type (from previous experiments)
oracle = {5: {'MRR': 0.9783, 'Precision': 0.9722, 'Hit': 0.9888},
          10: {'MRR': 0.9785, 'Precision': 0.9462, 'Hit': 0.9901}}

# Zeroed: without case_type (from ablation)
zeroed = {5: {'MRR': 0.9030, 'Precision': 0.8207, 'Hit': 0.9367},
          10: {'MRR': 0.9057, 'Precision': 0.8148, 'Hit': 0.9571}}

print(f"\n  Classifier: {best_name}")
print(f"  Classifier Test Acc: {test_acc:.4f}  |  F1 (macro): {test_f1:.4f}")
print(f"\n  {'Metric':<12} {'Oracle':>8} {'Predicted':>10} {'Zeroed':>8}  {'Recovery%':>9}")
print(f"  {'-'*12} {'-'*8} {'-'*10} {'-'*8}  {'-'*9}")

if results:
    predicted = {5: results[5], 10: results[10]}
    for k in [5, 10]:
        r = results[k]
        o = oracle[k]
        z = zeroed[k]
        for m in ['MRR', 'Precision', 'Hit']:
            rec_pct = (r[m] - z[m]) / (o[m] - z[m]) * 100 if (o[m] - z[m]) != 0 else 0
            print(f"  {m}@{k:<7d} {o[m]:8.4f} {r[m]:10.4f} {z[m]:8.4f}  {rec_pct:+8.1f}%")

    print(f"\n  Recovery % = (Predicted - Zeroed) / (Oracle - Zeroed) x 100")
    print(f"  Higher = better recovery of lost performance")

    # Save results
    expB_results = {
        'classifier': best_name,
        'classifier_test_acc': float(test_acc),
        'classifier_test_f1_macro': float(test_f1),
        'case_type_list': case_type_list,
        'oracle': oracle,
        'zeroed': zeroed,
        'predicted': {str(k): {mk: float(mv) for mk, mv in v.items()} for k, v in predicted.items()},
        'timestamp': datetime.now().isoformat(),
    }
    with open(os.path.join(PATHS['master_results_dir'], 'expB_case_type_recovery.json'), 'w') as f:
        json.dump(expB_results, f, indent=2)
    print(f"\n  Results saved to expB_case_type_recovery.json")

print("\nDone.")
