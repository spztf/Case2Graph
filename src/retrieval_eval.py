"""
Task C Steps 7-9: Retrieval Evaluation + Baselines + Ablation + Results
Loads encoder_data.pkl and runs full evaluation pipeline.
"""
import json
import pickle
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import networkx as nx
from scipy.sparse.linalg import svds, eigsh

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

BASE_DIR = Path(r"C:\workspace\paper")
DATA_DIR = BASE_DIR / "data" / "kg"
MODEL_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

NODE_TYPES = ["COMPANY", "PERSON", "GOVERNMENT", "BANK_ACCOUNT", "ASSET", "POLICY"]

# ─── Load pre-computed data ─────────────────────────────────────────────────
print("Loading pre-computed encoder data...")
with open(MODEL_DIR / "encoder_data.pkl", "rb") as f:
    saved = pickle.load(f)

train_ids = saved["train_ids"]
val_ids = saved["val_ids"]
test_ids = saved["test_ids"]
ent_ids = saved["ent_ids"]
raw_dim = saved["raw_dim"]
case_embeddings = {k: np.array(v) for k, v in saved["case_embeddings"].items()}
ent_embeddings = {k: np.array(v) for k, v in saved["ent_embeddings"].items()}
case_types = saved["case_types"]
entity_alignments = saved["entity_alignments"]

# Build ent_matrix
ent_matrix = np.array([ent_embeddings[eid] for eid in ent_ids], dtype=np.float64)
print(f"  Test cases: {len(test_ids)}")
print(f"  Enterprise index: {ent_matrix.shape}")

# ─── Re-load graphs for baselines ───────────────────────────────────────────
def load_case_graphs():
    case_dir = DATA_DIR / "case_graphs"
    graphs = {}
    for fpath in sorted(case_dir.glob("case_*.json")):
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
        G = nx.DiGraph()
        for node in data["graph"]["nodes"]:
            G.add_node(node["node_id"], **node)
        for edge in data["graph"]["edges"]:
            G.add_edge(edge["source"], edge["target"], **edge)
        graphs[data["case_id"]] = {
            "graph": G,
            "case_type": data["case_type"],
            "entity_alignment": data.get("entity_alignment", {})
        }
    return graphs

def load_enterprise_subgraphs():
    ent_dir = DATA_DIR / "enterprise_subgraphs"
    subgraphs = {}
    for fpath in sorted(ent_dir.glob("COMP_*.json")):
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
        G = nx.DiGraph()
        for node in data["nodes"]:
            props = node.get("properties", {})
            G.add_node(node["node_id"], type=node.get("type", "UNKNOWN"), **props)
        for edge in data.get("edges", []):
            G.add_edge(edge["source"], edge["target"], **edge)
        subgraphs[data["center_node"]] = {"graph": G}
    return subgraphs

print("Loading graphs for baselines...")
case_graphs = load_case_graphs()
ent_subgraphs = load_enterprise_subgraphs()

# ─── Ground Truth Construction ──────────────────────────────────────────────
def build_ground_truth(case_graphs, test_ids, ent_ids):
    """
    Build ground truth relevance for each test case.
    Uses entity_alignment + subgraph center node matching.
    """
    gt = {}
    for case_id in test_ids:
        case_G = case_graphs[case_id]["graph"]
        alignment = case_graphs[case_id].get("entity_alignment", {})
        relevant = set()
        
        # Direct entity alignment mapping
        for local_id, ent_id in alignment.items():
            relevant.add(ent_id)
        
        # Also match by: if entity subgraph center node appears in case graph
        case_node_names = set()
        for node in case_G.nodes():
            name = case_G.nodes[node].get("name", "")
            if name:
                case_node_names.add(name)
        
        # Check if any enterprise center node name matches case node names
        for ent_id in ent_ids:
            if ent_id in ent_subgraphs:
                center_name = ent_subgraphs[ent_id]["graph"].nodes.get(ent_id, {}).get("name", "")
                if center_name and center_name in case_node_names:
                    relevant.add(ent_id)
        
        gt[case_id] = relevant
    
    return gt

print("Building ground truth...")
ground_truth = build_ground_truth(case_graphs, test_ids, ent_ids)
# Count cases with non-empty ground truth
gt_counts = {cid: len(gt) for cid, gt in ground_truth.items()}
print(f"  Ground truth sizes: min={min(gt_counts.values())}, max={max(gt_counts.values())}, "
      f"mean={np.mean(list(gt_counts.values())):.1f}")

# ─── Evaluation Functions ───────────────────────────────────────────────────
def cosine_search(query_vec, index_matrix, ent_ids_list, k=20):
    """Brute-force cosine similarity search."""
    similarities = index_matrix @ query_vec  # [N]
    top_k_idx = np.argsort(similarities)[::-1][:k]
    results = []
    for idx in top_k_idx:
        results.append({
            "enterprise_id": ent_ids_list[idx],
            "similarity": float(similarities[idx])
        })
    return results

def compute_metrics(retrieved_ids, relevant_ids, k):
    """Compute Precision@k, Recall@k, MRR, Hit@k."""
    hits = len(retrieved_ids & relevant_ids)
    precision = hits / k
    recall = hits / len(relevant_ids) if relevant_ids else 0.0
    return precision, recall

def evaluate_method(query_embeddings, index_matrix, ent_ids_list, ground_truth, 
                    test_ids, k_values=[5, 10, 20]):
    """Evaluate a method given query embeddings dict."""
    metrics = {k: {"precision": [], "recall": [], "mrr": [], "hits": []} for k in k_values}
    
    for case_id in test_ids:
        if case_id not in ground_truth or not ground_truth[case_id]:
            continue
        if case_id not in query_embeddings:
            continue
        
        relevant = ground_truth[case_id]
        query_vec = query_embeddings[case_id]
        
        for k in k_values:
            results = cosine_search(query_vec, index_matrix, ent_ids_list, k=k)
            retrieved_ids = set(r["enterprise_id"] for r in results)
            
            prec, rec = compute_metrics(retrieved_ids, relevant, k)
            metrics[k]["precision"].append(prec)
            metrics[k]["recall"].append(rec)
            
            # MRR and Hit
            found = False
            for rank, r in enumerate(results, 1):
                if r["enterprise_id"] in relevant:
                    metrics[k]["mrr"].append(1.0 / rank)
                    metrics[k]["hits"].append(1.0)
                    found = True
                    break
            if not found:
                metrics[k]["mrr"].append(0.0)
                metrics[k]["hits"].append(0.0)
    
    result = {}
    for k in k_values:
        m = metrics[k]
        result[f"Precision@{k}"] = round(np.mean(m["precision"]), 4) if m["precision"] else 0.0
        result[f"Recall@{k}"] = round(np.mean(m["recall"]), 4) if m["recall"] else 0.0
        result[f"MRR@{k}"] = round(np.mean(m["mrr"]), 4) if m["mrr"] else 0.0
        result[f"Hit@{k}"] = round(np.mean(m["hits"]), 4) if m["hits"] else 0.0
    
    return result

# ─── Feature functions (for baselines) ──────────────────────────────────────
def compute_graph_features(G):
    n = max(G.number_of_nodes(), 1)
    m = max(G.number_of_edges(), 1)
    features = []
    
    features.append(np.log1p(n) / 10.0)
    features.append(np.log1p(m) / 10.0)
    features.append(m / max(n, 1))
    try:
        features.append(nx.number_strongly_connected_components(G) / max(n, 1))
    except:
        features.append(1.0/max(n,1))
    try:
        features.append(nx.number_weakly_connected_components(G) / max(n, 1))
    except:
        features.append(1.0/max(n,1))
    features.append(nx.density(G) if n > 1 else 0.0)
    
    in_degrees = [d for _, d in G.in_degree()]
    out_degrees = [d for _, d in G.out_degree()]
    features.append(np.mean(in_degrees) if in_degrees else 0.0)
    features.append(np.std(in_degrees) if len(in_degrees) > 1 else 0.0)
    features.append(np.max(in_degrees) if in_degrees else 0.0)
    features.append(np.mean(out_degrees) if out_degrees else 0.0)
    features.append(np.std(out_degrees) if len(out_degrees) > 1 else 0.0)
    features.append(np.max(out_degrees) if out_degrees else 0.0)
    
    try:
        pr = nx.pagerank(G, alpha=0.85, max_iter=50)
        pr_vals = list(pr.values())
        features.append(np.mean(pr_vals))
        features.append(np.std(pr_vals) if len(pr_vals) > 1 else 0.0)
    except:
        features.extend([0.0, 0.0])
    try:
        features.append(nx.reciprocity(G))
    except:
        features.append(0.0)
    
    type_counts = defaultdict(int)
    for node in G.nodes():
        ntype = G.nodes[node].get("type", "UNKNOWN")
        type_counts[ntype] += 1
    for t in NODE_TYPES:
        features.append(type_counts.get(t, 0) / max(n, 1))
    
    try:
        if n <= 100:
            A = nx.adjacency_matrix(G.to_undirected()).astype(np.float64).toarray()
            eigenvals = np.linalg.eigvalsh(A)
            top5 = np.sort(np.abs(eigenvals))[::-1][:5]
            for i in range(5):
                features.append(top5[i] if i < len(top5) else 0.0)
        else:
            features.extend([0.0]*5)
    except:
        features.extend([0.0]*5)
    
    features.append(np.log1p(nx.number_of_selfloops(G)))
    
    feat = np.array(features, dtype=np.float64)
    norm = np.linalg.norm(feat)
    if norm > 0:
        feat = feat / norm
    return feat

def compute_node2vec_lite(G, dim=64, walk_length=8, num_walks=10):
    n = G.number_of_nodes()
    if n < 2:
        return np.zeros(dim, dtype=np.float64)
    nodes = list(G.nodes())
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    undirected = G.to_undirected()
    
    walks = []
    for _ in range(num_walks):
        for start_node in nodes:
            walk = [start_node]
            current = start_node
            for _ in range(walk_length - 1):
                neighbors = list(undirected.neighbors(current))
                if not neighbors:
                    break
                current = random.choice(neighbors)
                walk.append(current)
            walks.append([node_to_idx[n] for n in walk])
    
    from scipy import sparse
    window_size = 3
    cooc = sparse.lil_matrix((n, n), dtype=np.float64)
    for walk in walks:
        for i, node_i in enumerate(walk):
            start = max(0, i - window_size)
            end = min(len(walk), i + window_size + 1)
            for j in range(start, end):
                if i != j:
                    cooc[node_i, walk[j]] += 1.0
    cooc = cooc.tocsr()
    
    k = min(dim, n - 1)
    if k < 1:
        return np.zeros(dim, dtype=np.float64)
    
    try:
        U, s, Vt = svds(cooc, k=k, which="LM")
        node_embs = U @ np.diag(np.sqrt(np.abs(s)))
        graph_emb = node_embs.mean(axis=0)
        result = np.zeros(dim, dtype=np.float64)
        result[:len(graph_emb)] = graph_emb[:dim]
        norm = np.linalg.norm(result)
        if norm > 0:
            result /= norm
        return result
    except:
        return np.zeros(dim, dtype=np.float64)

def compute_spectral_embedding(G, dim=64):
    n = G.number_of_nodes()
    if n < 2:
        return np.zeros(dim, dtype=np.float64)
    try:
        L = nx.normalized_laplacian_matrix(G.to_undirected()).astype(np.float64)
        k = min(dim, n - 1)
        if k < 1:
            return np.zeros(dim, dtype=np.float64)
        eigenvalues, eigenvectors = eigsh(L, k=k, which="SM")
        graph_emb = eigenvectors.mean(axis=0)
        result = np.zeros(dim, dtype=np.float64)
        result[:len(graph_emb)] = graph_emb[:dim]
        norm = np.linalg.norm(result)
        if norm > 0:
            result /= norm
        return result
    except:
        return np.zeros(dim, dtype=np.float64)

# ─── Step 7 & 8: Main Evaluation ────────────────────────────────────────────
print("\n" + "=" * 60)
print("RETRIEVAL EVALUATION")
print("=" * 60)

# 1. Our method (SubGraphFeatureEncoder)
print("\n[Our Method] SubGraphFeatureEncoder...")
our_results = evaluate_method(case_embeddings, ent_matrix, ent_ids, ground_truth, test_ids)
for k, v in our_results.items():
    print(f"  {k}: {v}")

# 2. Node2Vec Baseline
print("\n[Baseline 1] Node2Vec+AvgPool...")
t0 = time.time()
n2v_embeddings = {}
for cid in case_graphs:
    n2v_embeddings[cid] = compute_node2vec_lite(case_graphs[cid]["graph"], dim=64)

n2v_ent_matrix = []
for ent_id in ent_ids:
    emb = compute_node2vec_lite(ent_subgraphs[ent_id]["graph"], dim=64)
    n2v_ent_matrix.append(emb)
n2v_ent_matrix = np.array(n2v_ent_matrix, dtype=np.float64)
n2v_results = evaluate_method(n2v_embeddings, n2v_ent_matrix, ent_ids, ground_truth, test_ids)
for k, v in n2v_results.items():
    print(f"  {k}: {v}")
print(f"  Time: {time.time()-t0:.1f}s")

# 3. Spectral Baseline
print("\n[Baseline 2] Spectral+MeanPool...")
t0 = time.time()
spec_embeddings = {}
for cid in case_graphs:
    spec_embeddings[cid] = compute_spectral_embedding(case_graphs[cid]["graph"], dim=64)

spec_ent_matrix = []
for ent_id in ent_ids:
    emb = compute_spectral_embedding(ent_subgraphs[ent_id]["graph"], dim=64)
    spec_ent_matrix.append(emb)
spec_ent_matrix = np.array(spec_ent_matrix, dtype=np.float64)
spec_results = evaluate_method(spec_embeddings, spec_ent_matrix, ent_ids, ground_truth, test_ids)
for k, v in spec_results.items():
    print(f"  {k}: {v}")
print(f"  Time: {time.time()-t0:.1f}s")

# 4. Structural Features Baseline
print("\n[Baseline 3] StructuralFeatures+Norm...")
t0 = time.time()
feat_embeddings = {}
for cid in case_graphs:
    feat_embeddings[cid] = compute_graph_features(case_graphs[cid]["graph"])

feat_ent_matrix = []
for ent_id in ent_ids:
    emb = compute_graph_features(ent_subgraphs[ent_id]["graph"])
    feat_ent_matrix.append(emb)
feat_ent_matrix = np.array(feat_ent_matrix, dtype=np.float64)
feat_results = evaluate_method(feat_embeddings, feat_ent_matrix, ent_ids, ground_truth, test_ids)
for k, v in feat_results.items():
    print(f"  {k}: {v}")
print(f"  Time: {time.time()-t0:.1f}s")

# 5. Rule-based Baseline
print("\n[Baseline 4] Rule-based (Industry Jaccard)...")
t0 = time.time()
# Pre-compute enterprise features
ent_features = {}
for ent_id in ent_ids:
    G = ent_subgraphs[ent_id]["graph"]
    industries = set()
    provinces = set()
    for node in G.nodes():
        props = G.nodes[node]
        if "industry_code" in props:
            industries.add(props["industry_code"])
        if "registered_province" in props:
            provinces.add(props["registered_province"])
    ent_features[ent_id] = {"industries": industries, "provinces": provinces}

rule_metrics = {k: {"precision": [], "recall": [], "mrr": [], "hits": []} for k in [5, 10, 20]}
for case_id in test_ids:
    if case_id not in ground_truth or not ground_truth[case_id]:
        continue
    relevant = ground_truth[case_id]
    case_G = case_graphs[case_id]["graph"]
    
    # Extract case industries
    case_industries = set()
    for node in case_G.nodes():
        props = case_G.nodes[node]
        if "industry_code" in props:
            case_industries.add(props["industry_code"])
    
    # Score enterprises by Jaccard similarity
    scores = []
    for ent_id in ent_ids:
        ef = ent_features[ent_id]
        if case_industries and ef["industries"]:
            jaccard = len(case_industries & ef["industries"]) / max(1, len(case_industries | ef["industries"]))
        else:
            jaccard = 0.0
        scores.append((ent_id, jaccard))
    scores.sort(key=lambda x: x[1], reverse=True)
    
    for k in [5, 10, 20]:
        top_k = scores[:k]
        retrieved_ids = set(eid for eid, _ in top_k)
        prec, rec = compute_metrics(retrieved_ids, relevant, k)
        rule_metrics[k]["precision"].append(prec)
        rule_metrics[k]["recall"].append(rec)
        
        found = False
        for rank, (eid, _) in enumerate(top_k, 1):
            if eid in relevant:
                rule_metrics[k]["mrr"].append(1.0/rank)
                rule_metrics[k]["hits"].append(1.0)
                found = True
                break
        if not found:
            rule_metrics[k]["mrr"].append(0.0)
            rule_metrics[k]["hits"].append(0.0)

rule_results = {}
for k in [5, 10, 20]:
    m = rule_metrics[k]
    rule_results[f"Precision@{k}"] = round(np.mean(m["precision"]), 4) if m["precision"] else 0.0
    rule_results[f"Recall@{k}"] = round(np.mean(m["recall"]), 4) if m["recall"] else 0.0
    rule_results[f"MRR@{k}"] = round(np.mean(m["mrr"]), 4) if m["mrr"] else 0.0
    rule_results[f"Hit@{k}"] = round(np.mean(m["hits"]), 4) if m["hits"] else 0.0
for k, v in rule_results.items():
    print(f"  {k}: {v}")
print(f"  Time: {time.time()-t0:.1f}s")

# ─── Ablation Studies ──────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("ABLATION STUDIES")
print("=" * 60)

# We need to re-build embeddings for ablation variants
# Ablation 1: w/o Spectral features (only struct + node2vec)
print("\n[Ablation 1] w/o Spectral Features...")
t0 = time.time()
abl1_case = {}
for cid in case_graphs:
    G = case_graphs[cid]["graph"]
    feat = compute_graph_features(G)
    rw = compute_node2vec_lite(G, dim=64)
    combined = np.concatenate([feat, rw])
    combined = np.nan_to_num(combined, nan=0.0, posinf=0.0, neginf=0.0)
    norm = np.linalg.norm(combined)
    if norm > 0:
        combined /= norm
    abl1_case[cid] = combined

abl1_ent = []
for ent_id in ent_ids:
    G = ent_subgraphs[ent_id]["graph"]
    feat = compute_graph_features(G)
    rw = compute_node2vec_lite(G, dim=64)
    combined = np.concatenate([feat, rw])
    combined = np.nan_to_num(combined, nan=0.0, posinf=0.0, neginf=0.0)
    norm = np.linalg.norm(combined)
    if norm > 0:
        combined /= norm
    abl1_ent.append(combined)
abl1_ent = np.array(abl1_ent, dtype=np.float64)

abl1_results = evaluate_method(abl1_case, abl1_ent, ent_ids, ground_truth, test_ids)
for k, v in abl1_results.items():
    print(f"  {k}: {v}")
print(f"  Time: {time.time()-t0:.1f}s")

# Ablation 2: w/o Node2Vec features (only struct + spectral)
print("\n[Ablation 2] w/o RandomWalk Features...")
t0 = time.time()
abl2_case = {}
for cid in case_graphs:
    G = case_graphs[cid]["graph"]
    feat = compute_graph_features(G)
    spec = compute_spectral_embedding(G, dim=64)
    combined = np.concatenate([feat, spec])
    combined = np.nan_to_num(combined, nan=0.0, posinf=0.0, neginf=0.0)
    norm = np.linalg.norm(combined)
    if norm > 0:
        combined /= norm
    abl2_case[cid] = combined

abl2_ent = []
for ent_id in ent_ids:
    G = ent_subgraphs[ent_id]["graph"]
    feat = compute_graph_features(G)
    spec = compute_spectral_embedding(G, dim=64)
    combined = np.concatenate([feat, spec])
    combined = np.nan_to_num(combined, nan=0.0, posinf=0.0, neginf=0.0)
    norm = np.linalg.norm(combined)
    if norm > 0:
        combined /= norm
    abl2_ent.append(combined)
abl2_ent = np.array(abl2_ent, dtype=np.float64)

abl2_results = evaluate_method(abl2_case, abl2_ent, ent_ids, ground_truth, test_ids)
for k, v in abl2_results.items():
    print(f"  {k}: {v}")
print(f"  Time: {time.time()-t0:.1f}s")

# Ablation 3: w/o Structural features (only node2vec + spectral)
print("\n[Ablation 3] w/o Structural Features...")
t0 = time.time()
abl3_case = {}
for cid in case_graphs:
    G = case_graphs[cid]["graph"]
    rw = compute_node2vec_lite(G, dim=64)
    spec = compute_spectral_embedding(G, dim=64)
    combined = np.concatenate([rw, spec])
    combined = np.nan_to_num(combined, nan=0.0, posinf=0.0, neginf=0.0)
    norm = np.linalg.norm(combined)
    if norm > 0:
        combined /= norm
    abl3_case[cid] = combined

abl3_ent = []
for ent_id in ent_ids:
    G = ent_subgraphs[ent_id]["graph"]
    rw = compute_node2vec_lite(G, dim=64)
    spec = compute_spectral_embedding(G, dim=64)
    combined = np.concatenate([rw, spec])
    combined = np.nan_to_num(combined, nan=0.0, posinf=0.0, neginf=0.0)
    norm = np.linalg.norm(combined)
    if norm > 0:
        combined /= norm
    abl3_ent.append(combined)
abl3_ent = np.array(abl3_ent, dtype=np.float64)

abl3_results = evaluate_method(abl3_case, abl3_ent, ent_ids, ground_truth, test_ids)
for k, v in abl3_results.items():
    print(f"  {k}: {v}")
print(f"  Time: {time.time()-t0:.1f}s")

# ─── Save Results ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SAVING RESULTS")
print("=" * 60)

results = {
    "experiment_id": "EXP_001",
    "date": time.strftime("%Y-%m-%d"),
    "environment": "Python 3.14 + NumPy 2.4 + NetworkX 3.6 + SciPy 1.17 (No PyTorch/FAISS)",
    "config": {
        "model": "SubGraphFeatureEncoder (SVD-projected combined features)",
        "raw_feat_dim": raw_dim,
        "output_dim": 128,
        "rw_dim": 64,
        "spectral_dim": 64,
        "method": "Structural + Node2Vec RandomWalk + Spectral concatenation + PCA projection",
        "train_cases": len(train_ids),
        "val_cases": len(val_ids),
        "test_cases": len(test_ids),
        "enterprise_subgraphs": len(ent_ids),
        "test_cases_with_gt": sum(1 for cid in test_ids if cid in ground_truth and ground_truth[cid]),
    },
    "results": {
        "SubGraphFeatureEncoder (Ours)": our_results,
        "Node2Vec+AvgPool": n2v_results,
        "Spectral+MeanPool": spec_results,
        "StructuralFeatures+Norm": feat_results,
        "Rule-based Baseline": rule_results
    },
    "ablation": {
        "w/o Spectral Features": {
            "Precision@10": abl1_results.get("Precision@10", 0),
            "MRR@10": abl1_results.get("MRR@10", 0)
        },
        "w/o RandomWalk Features": {
            "Precision@10": abl2_results.get("Precision@10", 0),
            "MRR@10": abl2_results.get("MRR@10", 0)
        },
        "w/o Structural Features": {
            "Precision@10": abl3_results.get("Precision@10", 0),
            "MRR@10": abl3_results.get("MRR@10", 0)
        }
    },
    "honest_note": (
        "This experiment was run without PyTorch/torch_geometric/FAISS (Python 3.14 incompatibility). "
        "The 'SubGraphFeatureEncoder' uses PCA on concatenated graph features "
        "(structural stats + skip-gram random walk embeddings + Laplacian spectral embeddings). "
        "Key challenges: (1) Case graphs are tiny (2-4 nodes) vs enterprise subgraphs (120-202 nodes) "
        "creating a severe size/scale mismatch for embedding methods. "
        "(2) Most case graphs have very sparse entity alignment (0-1 aligned enterprises), "
        "making ground truth construction challenging. "
        "(3) Without contrastive learning or GNN message-passing, the model relies on "
        "hand-crafted graph statistics which may not capture the subtle topological patterns "
        "that distinguish fraud-related subgraphs. "
        "Results should be interpreted as LOWER BOUNDS. "
        "A properly trained SubGNN with contrastive learning on GPU would likely achieve higher scores."
    )
}

# Save main results
output_path = RESULTS_DIR / "retrieval_metrics.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f"Saved: {output_path}")

# Save model metadata
model_data = {
    "model_type": "SubGraphFeatureEncoder",
    "raw_feat_dim": raw_dim,
    "output_dim": 128,
    "train_case_ids": train_ids,
    "val_case_ids": val_ids,
    "test_case_ids": test_ids,
    "fitted": True,
    "note": "Pure NumPy/SciPy model. No torch checkpoint available.",
    "date": time.strftime("%Y-%m-%d"),
}
model_path = MODEL_DIR / "subgnn_checkpoint.json"
with open(model_path, "w", encoding="utf-8") as f:
    json.dump(model_data, f, ensure_ascii=False, indent=2)
print(f"Saved: {model_path}")

# ─── Summary ───────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("EXPERIMENT COMPLETE - RESULTS SUMMARY")
print("=" * 60)
print(f"\n{'Method':<30} {'P@10':>8} {'R@10':>8} {'MRR@10':>8} {'Hit@10':>8}")
print("-" * 65)
for method, res in results["results"].items():
    name = method[:28]
    print(f"{name:<30} {res.get('Precision@10', 0):>8.4f} {res.get('Recall@10', 0):>8.4f} "
          f"{res.get('MRR@10', 0):>8.4f} {res.get('Hit@10', 0):>8.4f}")

print(f"\nAblation (P@10 / MRR@10):")
for abl_name, abl_res in results["ablation"].items():
    print(f"  {abl_name}: P@10={abl_res['Precision@10']:.4f}, MRR@10={abl_res['MRR@10']:.4f}")
