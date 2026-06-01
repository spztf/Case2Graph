#!/usr/bin/env python3
"""
Phase 3 — Master Experiment Suite for ICSC2026
================================================
Runs all experiments needed for paper narrative:
  1. BM25 baseline (text-based retrieval)
  2. Pure GNN baseline (CE-only, no contrastive)
  3. Multi-seed stability (V13 joint × 3 seeds)
  4. Lambda sweep (CE:NCE weight)
  5. Temperature ablation
  6. t-SNE visualization
  7. Full retrieval evaluation

Usage:
  python3 master_experiments.py                          # run ALL
  python3 master_experiments.py --exp bm25               # BM25 only
  python3 master_experiments.py --exp baseline           # baselines (BM25+random+pureGNN)
  python3 master_experiments.py --exp seeds              # multi-seed
  python3 master_experiments.py --exp lambda             # lambda sweep
  python3 master_experiments.py --exp tau                # temperature sweep
  python3 master_experiments.py --exp tsne               # visualization
  python3 master_experiments.py --exp full_eval          # full retrieval
  python3 master_experiments.py --exp summary            # just print summary
  python3 master_experiments.py --gpu 0                  # select GPU

Output: /root/fd/workspace/paper/results/task_c/master_results/
"""

import json, os, sys, time, math, copy, random, gc, argparse, warnings
from collections import defaultdict, Counter
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

warnings.filterwarnings('ignore')

# ============================================================
# GLOBAL CONFIG
# ============================================================
PATHS = {
    'case_graphs_dir': '/root/fd/workspace/paper/data/kg/case_graphs',
    'enterprise_subgraphs_dir': '/root/fd/workspace/paper/data/kg/enterprise_subgraphs',
    'output_dir': '/root/fd/workspace/paper/results/task_c',
    'model_dir': '/root/fd/workspace/paper/models',
    'split_meta_path': '/root/fd/workspace/paper/results/task_c/split_meta_v9.json',
    'vocab_path': '/root/fd/workspace/paper/V10/node_vocabs.json',
    'master_results_dir': '/root/fd/workspace/paper/results/task_c/master_results',
}

os.makedirs(PATHS['master_results_dir'], exist_ok=True)
os.makedirs(PATHS['output_dir'], exist_ok=True)
os.makedirs(PATHS['model_dir'], exist_ok=True)

NODE_TYPES = ['Company', 'Person', 'Authority', 'ViolationEvent',
              'FinancialFlow', 'Location', 'Industry']
EDGE_TYPES = ['located_in', 'in_industry', 'penalized_by', 'committed',
              'controlled_by', 'involves_amount', 'co_penalized']
node_type_to_idx = {t: i for i, t in enumerate(NODE_TYPES)}
edge_type_to_idx = {t: i for i, t in enumerate(EDGE_TYPES)}

# ============================================================
# VOCAB LOADING (shared, loaded once)
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
    'province': build_vocab_map(VOCABS['province']),
    'city': build_vocab_map(VOCABS['city']),
    'role': build_vocab_map(VOCABS['role']),
    'case_type': build_vocab_map(VOCABS['case_type']),
    'violation_item': build_vocab_map(VOCABS['violation_item']),
    'flow_type': build_vocab_map(VOCABS['flow_type']),
}
VOCAB_SIZES = {k: len(v) for k, v in VOCAB_MAP.items()}

LOG_VALUE_MEAN = 2.022
LOG_VALUE_STD = 1.197

def degree_to_bucket(deg):
    if deg <= 0: return 0
    if deg == 1: return 1
    if deg <= 3: return 2
    if deg <= 5: return 3
    if deg <= 7: return 4
    if deg <= 15: return 5
    if deg <= 31: return 6
    return 7

# ============================================================
# DATA STRUCTURES
# ============================================================
class GraphData:
    __slots__ = ('node_types', 'edge_index', 'edge_types', 'node_attrs',
                 'num_nodes', 'num_edges', 'graph_id', 'label', 'text', 'metadata')
    def __init__(self, node_types, edge_index, edge_types, node_attrs,
                 graph_id=None, label=None, text='', metadata=None):
        self.node_types = node_types
        self.edge_index = edge_index
        self.edge_types = edge_types
        self.node_attrs = node_attrs
        self.num_nodes = len(node_types)
        self.num_edges = edge_index.shape[1] if edge_index.numel() > 0 else 0
        self.graph_id = graph_id
        self.label = label
        self.text = text
        self.metadata = metadata or {}

def json_graph_to_data(graph_dict, graph_id=None, label=None, max_nodes=None):
    """Convert JSON graph to GraphData with node features + text extraction."""
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

    for i, n in enumerate(nodes):
        nt = n.get('type', 'Company')
        node_types[i] = node_type_to_idx.get(nt, 0)
        props = n.get('properties', {})
        deg = degree[n['id']]
        db = degree_to_bucket(deg)

        if nt == 'Company':
            node_attrs[i, 0] = VOCAB_MAP['industry'].get(props.get('industry', ''), 0)
            node_attrs[i, 1] = VOCAB_MAP['province'].get(props.get('province', ''), 0)
            node_attrs[i, 2] = VOCAB_MAP['role'].get(props.get('role', ''), 0)
            node_attrs[i, 3] = db; node_attrs[i, 4] = 0.0; node_attrs[i, 5] = 0.0
        elif nt == 'Person':
            node_attrs[i, 0] = 0; node_attrs[i, 1] = 0
            node_attrs[i, 2] = VOCAB_MAP['role'].get(props.get('role', ''), 0)
            node_attrs[i, 3] = db; node_attrs[i, 4] = 0.0; node_attrs[i, 5] = 0.0
        elif nt == 'ViolationEvent':
            node_attrs[i, 0] = VOCAB_MAP['case_type'].get(props.get('case_type', ''), 0)
            node_attrs[i, 1] = VOCAB_MAP['violation_item'].get(props.get('violation_item', ''), 0)
            node_attrs[i, 2] = 0; node_attrs[i, 3] = db
            node_attrs[i, 4] = 0.0; node_attrs[i, 5] = 0.0
            desc = props.get('scheme_description', '')
            if desc: text_parts.append(desc)
        elif nt == 'FinancialFlow':
            node_attrs[i, 0] = VOCAB_MAP['flow_type'].get(props.get('type', ''), 0)
            node_attrs[i, 1] = 0; node_attrs[i, 2] = 0; node_attrs[i, 3] = db
            raw_val = props.get('value', None)
            if raw_val is not None and raw_val > 0:
                node_attrs[i, 4] = (math.log10(raw_val + 1) - LOG_VALUE_MEAN) / LOG_VALUE_STD
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
                et = edge_type_to_idx.get(e.get('relation', 'unknown'), 0)
                etype_list.append(et)
                src_list.append(dst); dst_list.append(src)
                etype_list.append(et)
        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        edge_types = torch.tensor(etype_list, dtype=torch.long)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_types = torch.zeros(0, dtype=torch.long)

    return GraphData(
        node_types=node_types, edge_index=edge_index, edge_types=edge_types,
        node_attrs=node_attrs, graph_id=graph_id, label=label,
        text=' '.join(text_parts))

# ============================================================
# MODEL COMPONENTS (from V13 — proven architecture)
# ============================================================
class NodeFeatureEncoder(nn.Module):
    def __init__(self, hidden_dim=128, feat_dim=32):
        super().__init__()
        fd = feat_dim
        max_emb0 = max(VOCAB_SIZES['industry'], VOCAB_SIZES['case_type'], VOCAB_SIZES['flow_type'])
        max_emb1 = max(VOCAB_SIZES['province'], VOCAB_SIZES['violation_item'])
        self.emb_0 = nn.Embedding(max_emb0, fd)
        self.emb_1 = nn.Embedding(max_emb1, fd)
        self.emb_2 = nn.Embedding(VOCAB_SIZES['role'], fd)
        self.emb_3 = nn.Embedding(8, fd // 2)
        self.value_enc = nn.Sequential(
            nn.Linear(1, fd // 2), nn.ReLU(inplace=True), nn.Linear(fd // 2, fd // 2))
        total_feat = 4 * fd
        self.project = nn.Sequential(
            nn.Linear(total_feat, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(inplace=True))

    def forward(self, node_attrs):
        B, N, _ = node_attrs.shape; device = node_attrs.device
        idx_0 = node_attrs[:, :, 0].long().clamp(0, self.emb_0.num_embeddings - 1)
        idx_1 = node_attrs[:, :, 1].long().clamp(0, self.emb_1.num_embeddings - 1)
        idx_2 = node_attrs[:, :, 2].long().clamp(0, self.emb_2.num_embeddings - 1)
        idx_3 = node_attrs[:, :, 3].long().clamp(0, 7)
        e0 = self.emb_0(idx_0); e1 = self.emb_1(idx_1); e2 = self.emb_2(idx_2); e3 = self.emb_3(idx_3)
        val = node_attrs[:, :, 4:5]; e4 = self.value_enc(val)
        feats = torch.cat([e0, e1, e2, e3, e4], dim=-1)
        return self.project(feats)

class EdgeTypeGINConv(nn.Module):
    def __init__(self, hidden_dim, train_eps=True):
        super().__init__()
        self.eps = nn.Parameter(torch.tensor(0.0))
        self.edge_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim))
        self.bn = nn.BatchNorm1d(hidden_dim)

    def forward(self, x, adj, edge_emb_padded):
        h_self = (1 + self.eps) * x
        neighbor_base = torch.bmm(adj, x)
        edge_mod = torch.einsum('b ijh, b jh -> b ih', adj.unsqueeze(-1) * edge_emb_padded, x)
        neighbor_msgs = neighbor_base + 0.1 * edge_mod
        h = h_self + neighbor_msgs
        h = self.mlp(h)
        h = self.bn(h.transpose(1, 2)).transpose(1, 2)
        return F.relu(h)

class GINEncoder(nn.Module):
    """Generic GIN Encoder — with/without classifier for different experiments."""
    def __init__(self, num_node_types, num_edge_types, num_classes,
                 hidden_dim=128, num_layers=3, dropout=0.3, with_classifier=True):
        super().__init__()
        self.node_embed = nn.Embedding(num_node_types, hidden_dim)
        self.edge_embed = nn.Embedding(num_edge_types, hidden_dim)
        self.feat_encoder = NodeFeatureEncoder(hidden_dim=hidden_dim, feat_dim=32)
        self.convs = nn.ModuleList([EdgeTypeGINConv(hidden_dim) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2), nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim), nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim))
        self.with_classifier = with_classifier
        if with_classifier:
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(inplace=True),
                nn.Dropout(dropout), nn.Linear(hidden_dim // 2, num_classes))

    def _build_adj_and_edge_emb(self, edge_index, edge_types, B, N, device):
        adj = torch.zeros(B, N, N, device=device)
        edge_emb_padded = torch.zeros(B, N, N, self.edge_embed.embedding_dim, device=device)
        for b in range(B):
            ei = edge_index[b]; et = edge_types[b]
            valid = (ei[0] >= 0) & (ei[1] >= 0) & (et >= 0)
            if valid.any():
                src = ei[0][valid]; dst = ei[1][valid]; etypes = et[valid]
                adj[b, src, dst] = 1.0
                edge_emb_padded[b, src, dst] = self.edge_embed(etypes)
        deg = adj.sum(dim=-1, keepdim=True).clamp(min=1)
        adj = adj / deg
        return adj, edge_emb_padded

    def forward(self, node_types, edge_index, edge_types, node_attrs, mask=None, return_logits=False):
        B, N = node_types.shape; device = node_types.device
        x_type = self.node_embed(node_types)
        x_feat = self.feat_encoder(node_attrs)
        x = x_type + x_feat
        adj, eep = self._build_adj_and_edge_emb(edge_index, edge_types, B, N, device)
        for conv in self.convs:
            x = conv(x, adj, eep); x = self.dropout(x)
        if mask is not None:
            mf = mask.float().unsqueeze(-1); x = x * mf
            pooled = x.sum(dim=1) / mf.sum(dim=1).clamp(min=1)
        else:
            pooled = x.mean(dim=1)
        graph_emb = self.projection(pooled)
        graph_emb_norm = F.normalize(graph_emb, p=2, dim=1)
        if return_logits and self.with_classifier:
            return graph_emb_norm, self.classifier(graph_emb_norm)
        return graph_emb_norm

    @torch.no_grad()
    def encode_batch(self, graphs, device='cpu', batch_size=64):
        self.eval()
        all_embs = []
        for i in range(0, len(graphs), batch_size):
            batch = graphs[i:i + batch_size]
            nt, ei, et, na, mask = collate_raw(batch, device)
            emb = self.forward(nt, ei, et, na, mask)
            all_embs.append(emb.cpu())
        return torch.cat(all_embs, dim=0)

# ============================================================
# DATA UTILITIES
# ============================================================
def collate_raw(batch, device='cpu'):
    max_nodes = max(g.num_nodes for g in batch)
    max_edges = max(g.num_edges for g in batch) if any(g.num_edges > 0 for g in batch) else 0
    B = len(batch)
    node_types = torch.zeros(B, max_nodes, dtype=torch.long, device=device)
    node_attrs = torch.zeros(B, max_nodes, 6, dtype=torch.float32, device=device)
    mask = torch.zeros(B, max_nodes, dtype=torch.bool, device=device)
    for i, g in enumerate(batch):
        n = g.num_nodes
        node_types[i, :n] = g.node_types.to(device); mask[i, :n] = True
        node_attrs[i, :n] = g.node_attrs.to(device)
    if max_edges > 0:
        edge_index = torch.full((B, 2, max_edges), -1, dtype=torch.long, device=device)
        edge_types = torch.full((B, max_edges), -1, dtype=torch.long, device=device)
        for i, g in enumerate(batch):
            e = g.num_edges
            if e > 0:
                edge_index[i, :, :e] = g.edge_index.to(device)
                edge_types[i, :e] = g.edge_types.to(device)
    else:
        edge_index = torch.full((B, 2, 0), -1, dtype=torch.long, device=device)
        edge_types = torch.full((B, 0), -1, dtype=torch.long, device=device)
    return node_types, edge_index, edge_types, node_attrs, mask

def collate_fn(batch):
    nt, ei, et, na, mask = collate_raw(batch, 'cpu')
    labels = torch.tensor([g.label for g in batch], dtype=torch.long)
    return {'node_types': nt, 'edge_index': ei, 'edge_types': et,
            'node_attrs': na, 'mask': mask, 'labels': labels,
            'graph_ids': [g.graph_id for g in batch]}

def info_nce_loss(embeddings, labels, temperature=0.5):
    B = embeddings.shape[0]; device = embeddings.device
    sim = torch.mm(embeddings, embeddings.t()) / temperature
    pos_mask = (labels.view(-1, 1) == labels.view(1, -1)).float()
    pos_mask.fill_diagonal_(0)
    pos_count = pos_mask.sum(dim=1); valid = pos_count > 0
    if not valid.any():
        return torch.tensor(0.0, device=device, requires_grad=True)
    exp_sim = torch.exp(sim)
    pos_sum = (exp_sim * pos_mask).sum(dim=1)
    all_sum = exp_sim.sum(dim=1) - torch.exp(sim.diag())
    loss = -torch.log(pos_sum[valid] / all_sum[valid].clamp(min=1e-8))
    return loss.mean()

# ============================================================
# GLOBAL DATA LOADING (cached once)
# ============================================================
_data_cache = {}

def load_all_data(verbose=True):
    """Load all graphs, splits, labels — cached after first call."""
    if _data_cache:
        return (_data_cache['train_graphs'], _data_cache['val_graphs'],
                _data_cache['test_graphs'], _data_cache['all_graphs'],
                _data_cache['num_classes'], _data_cache['unique_types'],
                _data_cache['train_ids'], _data_cache['val_ids'],
                _data_cache['test_ids'], _data_cache['case_data'],
                _data_cache['ent_graphs_loaded'], _data_cache['ent_ids'],
                _data_cache['ent_case_types'], _data_cache['gt_relaxed'])

    if verbose: print("[Data] Loading split_meta_v9.json...")
    with open(PATHS['split_meta_path']) as f:
        meta = json.load(f)
    train_ids = meta['train_ids']; val_ids = meta['val_ids']; test_ids = meta['test_ids']
    case_data = meta['case_data']; unique_types = meta['unique_types']
    type_to_label = {t: i for i, t in enumerate(unique_types)}
    num_classes = len(unique_types)
    if verbose:
        print(f"  Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}, Classes: {num_classes}")

    # Load case graphs
    if verbose: print("[Data] Loading case graphs...")
    cg_dir = PATHS['case_graphs_dir']; all_fnames = set(os.listdir(cg_dir))
    all_graphs = {}; labels_map = {}; skipped = 0
    for cid in train_ids + val_ids + test_ids:
        fname = None
        for cand in [cid + ".json", cid.lower() + ".json",
                     cid.replace("CASE_", "case_").lower() + ".json"]:
            if cand in all_fnames: fname = cand; break
        if fname is None: skipped += 1; continue
        with open(os.path.join(cg_dir, fname), encoding='utf-8') as f:
            raw = json.load(f)
        sub_type = case_data.get(cid, {}).get('sub_type', raw.get('case_type', 'Unknown'))
        label = type_to_label.get(sub_type, 0)
        g = json_graph_to_data(raw.get('graph', raw), graph_id=cid, label=label, max_nodes=200)
        all_graphs[cid] = g; labels_map[cid] = label
    if verbose: print(f"  Loaded {len(all_graphs)} case graphs (skipped {skipped})")

    train_graphs = [all_graphs[c] for c in train_ids if c in all_graphs]
    val_graphs = [all_graphs[c] for c in val_ids if c in all_graphs]
    test_graphs = [all_graphs[c] for c in test_ids if c in all_graphs]

    # Load enterprise subgraphs
    if verbose: print("[Data] Loading enterprise subgraphs...")
    ent_dir = PATHS['enterprise_subgraphs_dir']
    ent_graphs_loaded = {}; ent_case_types = {}
    for fname in sorted(os.listdir(ent_dir)):
        eid = fname.replace('.json', '')
        try:
            with open(os.path.join(ent_dir, fname), encoding='utf-8') as f:
                raw = json.load(f)
        except: continue
        ct_counts = Counter()
        text_parts = []
        for node in raw.get('nodes', []):
            if node.get('type') == 'ViolationEvent':
                ct = node.get('properties', {}).get('case_type', '')
                desc = node.get('properties', {}).get('scheme_description', '')
                if ct and ct != 'Unknown': ct_counts[ct] += 1
                if desc: text_parts.append(desc)
        primary_ct = ct_counts.most_common(1)[0][0] if ct_counts else 'Unknown'
        ent_case_types[eid] = primary_ct
        ent_graphs_loaded[eid] = json_graph_to_data(raw, graph_id=eid, max_nodes=500)
        ent_graphs_loaded[eid].text = ' '.join(text_parts)

    ent_ids = sorted(ent_graphs_loaded.keys())
    if verbose: print(f"  Loaded {len(ent_ids)} enterprise subgraphs")

    # Build ground truth (relaxed: same coarse case_type)
    def sub_type_to_case_type(st):
        for ct in ['虚开增值税专用发票', '虚开发票', '隐匿收入', '骗取出口退税', '转让定价', '其他']:
            if ct in st: return ct
        return st

    gt_relaxed = {}
    for cid in test_ids:
        if cid not in all_graphs: continue
        coarse_ct = sub_type_to_case_type(case_data.get(cid, {}).get('sub_type', 'Unknown'))
        matches = {eid for eid, ect in ent_case_types.items() if ect == coarse_ct}
        if matches:
            gt_relaxed[cid] = matches

    _data_cache.update({
        'train_graphs': train_graphs, 'val_graphs': val_graphs,
        'test_graphs': test_graphs, 'all_graphs': all_graphs,
        'num_classes': num_classes, 'unique_types': unique_types,
        'train_ids': train_ids, 'val_ids': val_ids, 'test_ids': test_ids,
        'case_data': case_data, 'ent_graphs_loaded': ent_graphs_loaded,
        'ent_ids': ent_ids, 'ent_case_types': ent_case_types,
        'gt_relaxed': gt_relaxed
    })
    return (train_graphs, val_graphs, test_graphs, all_graphs,
            num_classes, unique_types, train_ids, val_ids, test_ids,
            case_data, ent_graphs_loaded, ent_ids, ent_case_types, gt_relaxed)


# ============================================================
# RETRIEVAL EVALUATION (shared)
# ============================================================
@torch.no_grad()
def evaluate_retrieval_gnn(model, test_graphs_dict, ent_graphs_dict, ent_ids,
                           test_case_ids, ground_truth, device, k_values=[5, 10, 20],
                           batch_size=64):
    """Full retrieval evaluation with GNN embeddings."""
    model.eval()
    case_list = [test_graphs_dict[cid] for cid in test_case_ids]
    case_embs = model.encode_batch(case_list, device, batch_size=batch_size)
    ent_list = [ent_graphs_dict[eid] for eid in ent_ids]
    ent_embs = model.encode_batch(ent_list, device, batch_size=max(batch_size // 2, 8))
    case_embs = F.normalize(case_embs, p=2, dim=1)
    ent_embs = F.normalize(ent_embs, p=2, dim=1)
    sim = torch.mm(case_embs, ent_embs.t())
    metrics = {k: {'precision': [], 'recall': [], 'hit': [], 'mrr': []} for k in k_values}
    for i, cid in enumerate(test_case_ids):
        sims = sim[i]; ranked = torch.argsort(sims, descending=True)
        gt = ground_truth.get(cid, set())
        if not gt: continue
        for k in k_values:
            top_k = {ent_ids[idx] for idx in ranked[:k].tolist()}
            hits = len(top_k & gt)
            metrics[k]['precision'].append(hits / k)
            metrics[k]['recall'].append(hits / len(gt))
            metrics[k]['hit'].append(1.0 if hits > 0 else 0.0)
            mrr = 0.0
            for rank, idx in enumerate(ranked[:k].tolist(), 1):
                if ent_ids[idx] in gt: mrr = 1.0 / rank; break
            metrics[k]['mrr'].append(mrr)
    results = {}
    for k in k_values:
        m = metrics[k]
        results[k] = {
            'Precision': float(np.mean(m['precision'])),
            'Recall': float(np.mean(m['recall'])),
            'MRR': float(np.mean(m['mrr'])),
            'Hit': float(np.mean(m['hit'])),
            'Precision_std': float(np.std(m['precision'])),
            'MRR_std': float(np.std(m['mrr'])),
        }
    return results

def evaluate_retrieval_bm25(test_graphs_dict, ent_graphs_dict, ent_ids,
                            test_case_ids, ground_truth, k_values=[5, 10, 20]):
    """BM25 retrieval using scheme_description text."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    ent_texts = []
    for eid in ent_ids:
        g = ent_graphs_dict.get(eid)
        ent_texts.append(g.text if g and g.text else '')
    case_texts = []
    valid_case_ids = []
    for cid in test_case_ids:
        g = test_graphs_dict.get(cid)
        if g:
            case_texts.append(g.text if g.text else '')
            valid_case_ids.append(cid)
    vectorizer = TfidfVectorizer(max_features=5000, sublinear_tf=True)
    all_texts = ent_texts + case_texts
    tfidf = vectorizer.fit_transform(all_texts)
    ent_vecs = tfidf[:len(ent_texts)]
    case_vecs = tfidf[len(ent_texts):]
    from sklearn.metrics.pairwise import cosine_similarity
    sim = cosine_similarity(case_vecs, ent_vecs)
    metrics = {k: {'precision': [], 'recall': [], 'hit': [], 'mrr': []} for k in k_values}
    for i, cid in enumerate(valid_case_ids):
        ranked = np.argsort(-sim[i])
        gt = ground_truth.get(cid, set())
        if not gt: continue
        for k in k_values:
            top_k = {ent_ids[idx] for idx in ranked[:k]}
            hits = len(top_k & gt)
            metrics[k]['precision'].append(hits / k)
            metrics[k]['recall'].append(hits / len(gt))
            metrics[k]['hit'].append(1.0 if hits > 0 else 0.0)
            mrr = 0.0
            for rank, idx in enumerate(ranked[:k], 1):
                if ent_ids[idx] in gt: mrr = 1.0 / rank; break
            metrics[k]['mrr'].append(mrr)
    results = {}
    for k in k_values:
        m = metrics[k]
        results[k] = {'Precision': float(np.mean(m['precision'])),
                      'Recall': float(np.mean(m['recall'])),
                      'MRR': float(np.mean(m['mrr'])),
                      'Hit': float(np.mean(m['hit']))}
    return results

def evaluate_random_baseline(ent_ids, test_case_ids, ground_truth, dim=128, k_values=[5, 10, 20]):
    """Random embedding baseline."""
    rng = np.random.RandomState(42)
    case_mat = rng.randn(len(test_case_ids), dim).astype(np.float32)
    ent_mat = rng.randn(len(ent_ids), dim).astype(np.float32)
    case_mat = case_mat / (np.linalg.norm(case_mat, axis=1, keepdims=True) + 1e-8)
    ent_mat = ent_mat / (np.linalg.norm(ent_mat, axis=1, keepdims=True) + 1e-8)
    sim = np.dot(case_mat, ent_mat.T)
    metrics = {k: {'precision': [], 'recall': [], 'hit': [], 'mrr': []} for k in k_values}
    for i, cid in enumerate(test_case_ids):
        ranked = np.argsort(-sim[i])
        gt = ground_truth.get(cid, set())
        if not gt: continue
        for k in k_values:
            top_k = {ent_ids[idx] for idx in ranked[:k]}
            hits = len(top_k & gt)
            metrics[k]['precision'].append(hits / k)
            metrics[k]['recall'].append(hits / len(gt))
            metrics[k]['hit'].append(1.0 if hits > 0 else 0.0)
            mrr = 0.0
            for rank, idx in enumerate(ranked[:k], 1):
                if ent_ids[idx] in gt: mrr = 1.0 / rank; break
            metrics[k]['mrr'].append(mrr)
    results = {}
    for k in k_values:
        m = metrics[k]
        results[k] = {'Precision': float(np.mean(m['precision'])),
                      'Recall': float(np.mean(m['recall'])),
                      'MRR': float(np.mean(m['mrr'])),
                      'Hit': float(np.mean(m['hit']))}
    return results


# ============================================================
# EXPERIMENT: BM25 Baseline
# ============================================================
def run_bm25_baseline():
    print("\n" + "=" * 70)
    print("[EXP 1] BM25 Text Baseline")
    print("=" * 70)
    _, _, _, all_graphs, _, _, _, _, test_ids, _, ent_graphs, ent_ids, _, gt_relaxed = load_all_data(verbose=False)
    test_case_graphs = {cid: all_graphs[cid] for cid in test_ids if cid in all_graphs and cid in gt_relaxed}
    test_case_ids = list(test_case_graphs.keys())
    print(f"  Test cases: {len(test_case_ids)}, Enterprises: {len(ent_ids)}")
    t0 = time.time()
    results = evaluate_retrieval_bm25(test_case_graphs, ent_graphs, ent_ids, test_case_ids, gt_relaxed)
    print(f"  Done in {t0:.1f}s")
    for k in [5, 10, 20]:
        r = results[k]
        print(f"  @{k:2d}: P={r['Precision']:.4f}  MRR={r['MRR']:.4f}  Hit={r['Hit']:.4f}")
    out = {'experiment': 'BM25_baseline', 'timestamp': str(datetime.now()),
           'results': {f'@{k}': results[k] for k in [5, 10, 20]}}
    with open(os.path.join(PATHS['master_results_dir'], 'bm25_baseline.json'), 'w') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    return results

# ============================================================
# EXPERIMENT: Random Baseline
# ============================================================
def run_random_baseline():
    print("\n" + "=" * 70)
    print("[EXP 2] Random Embedding Baseline")
    print("=" * 70)
    _, _, _, all_graphs, _, _, _, _, test_ids, _, _, ent_ids, _, gt_relaxed = load_all_data(verbose=False)
    test_case_ids = [cid for cid in test_ids if cid in all_graphs and cid in gt_relaxed]
    results = evaluate_random_baseline(ent_ids, test_case_ids, gt_relaxed)
    for k in [5, 10, 20]:
        r = results[k]
        print(f"  @{k:2d}: P={r['Precision']:.4f}  MRR={r['MRR']:.4f}  Hit={r['Hit']:.4f}")
    out = {'experiment': 'random_baseline', 'timestamp': str(datetime.now()),
           'results': {f'@{k}': results[k] for k in [5, 10, 20]}}
    with open(os.path.join(PATHS['master_results_dir'], 'random_baseline.json'), 'w') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    return results

# ============================================================
# EXPERIMENT: Pure GNN (CE-only, no contrastive)
# ============================================================
def run_pure_gnn_baseline(seed=42):
    print("\n" + "=" * 70)
    print(f"[EXP 3] Pure GNN Baseline (CE-only, seed={seed})")
    print("=" * 70)
    set_seed(seed)
    (train_graphs, val_graphs, test_graphs, all_graphs,
     num_classes, unique_types, train_ids, val_ids, test_ids,
     case_data, ent_graphs, ent_ids, ent_case_types, gt_relaxed) = load_all_data(verbose=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")

    train_labels_list = [g.label for g in train_graphs]
    label_counts = Counter(train_labels_list)
    class_weights = {lbl: 1.0 / max(cnt, 1) for lbl, cnt in label_counts.items()}
    sample_weights = [class_weights[lbl] for lbl in train_labels_list]
    balanced_sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_graphs), replacement=True)

    train_loader = DataLoader(train_graphs, batch_size=128, sampler=balanced_sampler,
                              collate_fn=collate_fn, drop_last=True)
    val_loader = DataLoader(val_graphs, batch_size=128, shuffle=False, collate_fn=collate_fn)

    model = GINEncoder(num_node_types=len(NODE_TYPES), num_edge_types=len(EDGE_TYPES),
                       num_classes=num_classes, hidden_dim=128, num_layers=3,
                       dropout=0.3, with_classifier=True).to(device)
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=15)
    best_acc, best_epoch, patience = 0.0, 0, 0

    for epoch in range(1, 201):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for batch in train_loader:
            nt = batch['node_types'].to(device); ei = batch['edge_index'].to(device)
            et = batch['edge_types'].to(device); na = batch['node_attrs'].to(device)
            mask = batch['mask'].to(device); labels = batch['labels'].to(device)
            optimizer.zero_grad()
            _, logits = model(nt, ei, et, na, mask, return_logits=True)
            loss = F.cross_entropy(logits, labels, label_smoothing=0.05)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            pred = logits.argmax(dim=1)
            correct += (pred == labels).sum().item(); total += labels.size(0)

        model.eval()
        v_correct, v_total = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                nt = batch['node_types'].to(device); ei = batch['edge_index'].to(device)
                et = batch['edge_types'].to(device); na = batch['node_attrs'].to(device)
                mask = batch['mask'].to(device); labels = batch['labels'].to(device)
                _, logits = model(nt, ei, et, na, mask, return_logits=True)
                pred = logits.argmax(dim=1)
                v_correct += (pred == labels).sum().item(); v_total += labels.size(0)
        v_acc = v_correct / max(v_total, 1)
        t_acc = correct / max(total, 1)
        scheduler.step(v_acc)

        if v_acc > best_acc:
            best_acc = v_acc; best_epoch = epoch; patience = 0
            torch.save(model.state_dict(), os.path.join(PATHS['model_dir'], 'master_pure_gnn_best.pt'))
        else:
            patience += 1

        if epoch % 20 == 0 or epoch == 1:
            lr = optimizer.param_groups[0]['lr']
            print(f"  Epoch {epoch:3d} | Loss: {total_loss/max(len(train_loader),1):.4f} | "
                  f"Acc: {t_acc:.4f}/{v_acc:.4f} | Best: {best_acc:.4f} @{best_epoch} | LR: {lr:.2e}")

        if patience >= 40:
            print(f"  Early stop @ {epoch}")
            break

    model.load_state_dict(torch.load(os.path.join(PATHS['model_dir'], 'master_pure_gnn_best.pt'), map_location=device))
    print(f"  Best Val Acc: {best_acc:.4f} @ epoch {best_epoch}")

    test_case_graphs = {cid: all_graphs[cid] for cid in test_ids if cid in all_graphs and cid in gt_relaxed}
    test_case_ids_list = list(test_case_graphs.keys())
    results = evaluate_retrieval_gnn(model, test_case_graphs, ent_graphs, ent_ids,
                                     test_case_ids_list, gt_relaxed, device)
    for k in [5, 10, 20]:
        r = results[k]
        print(f"  @{k:2d}: P={r['Precision']:.4f}  MRR={r['MRR']:.4f}  Hit={r['Hit']:.4f}")

    out = {'experiment': 'pure_gnn_ce_only', 'seed': seed,
           'best_val_acc': best_acc, 'best_epoch': best_epoch,
           'results': {f'@{k}': results[k] for k in [5, 10, 20]}}
    with open(os.path.join(PATHS['master_results_dir'], 'pure_gnn_baseline.json'), 'w') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    return results


# ============================================================
# CORE: Joint Training (V13-style, configurable)
# ============================================================
def run_joint_training(name, ce_weight=0.5, nce_weight=0.5, temperature=0.5,
                       label_smoothing=0.05, seed=42, epochs=350, patience=80,
                       lr=1e-3, use_cosine=True, cosine_T0=60, cosine_Tmult=2,
                       verbose=True):
    """V13-style joint training with CE + InfoNCE."""
    set_seed(seed)
    (train_graphs, val_graphs, test_graphs, all_graphs,
     num_classes, unique_types, train_ids, val_ids, test_ids,
     case_data, ent_graphs, ent_ids, ent_case_types, gt_relaxed) = load_all_data(verbose=verbose)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if verbose: print(f"  Device: {device} | CE={ce_weight} | NCE={nce_weight} | tau={temperature} | seed={seed}")

    train_labels_list = [g.label for g in train_graphs]
    label_counts = Counter(train_labels_list)
    class_weights = {lbl: 1.0 / max(cnt, 1) for lbl, cnt in label_counts.items()}
    sample_weights = [class_weights[lbl] for lbl in train_labels_list]
    balanced_sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_graphs), replacement=True)

    train_loader = DataLoader(train_graphs, batch_size=128, sampler=balanced_sampler,
                              collate_fn=collate_fn, drop_last=True)
    val_loader = DataLoader(val_graphs, batch_size=128, shuffle=False, collate_fn=collate_fn)

    model = GINEncoder(num_node_types=len(NODE_TYPES), num_edge_types=len(EDGE_TYPES),
                       num_classes=num_classes, hidden_dim=128, num_layers=3,
                       dropout=0.3, with_classifier=True).to(device)
    if verbose: print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    if use_cosine:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=cosine_T0, T_mult=cosine_Tmult, eta_min=1e-6)
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20)

    best_nce, best_epoch, best_vacc, patience_counter = float('inf'), 0, 0.0, 0

    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            nt = batch['node_types'].to(device); ei = batch['edge_index'].to(device)
            et = batch['edge_types'].to(device); na = batch['node_attrs'].to(device)
            mask = batch['mask'].to(device); labels = batch['labels'].to(device)
            optimizer.zero_grad()
            embeddings, logits = model(nt, ei, et, na, mask, return_logits=True)
            ce_loss = F.cross_entropy(logits, labels, label_smoothing=label_smoothing)
            nce_loss = info_nce_loss(embeddings, labels, temperature)
            total_loss = ce_weight * ce_loss + nce_weight * nce_loss
            if torch.isfinite(total_loss):
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        model.eval()
        v_nce_sum, v_correct, v_total, v_batches = 0.0, 0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                nt = batch['node_types'].to(device); ei = batch['edge_index'].to(device)
                et = batch['edge_types'].to(device); na = batch['node_attrs'].to(device)
                mask = batch['mask'].to(device); labels = batch['labels'].to(device)
                embeddings, logits = model(nt, ei, et, na, mask, return_logits=True)
                nce_l = info_nce_loss(embeddings, labels, temperature)
                if torch.isfinite(nce_l):
                    v_nce_sum += nce_l.item(); v_batches += 1
                pred = logits.argmax(dim=1)
                v_correct += (pred == labels).sum().item(); v_total += labels.size(0)
        v_nce = v_nce_sum / max(v_batches, 1)
        v_acc = v_correct / max(v_total, 1)

        lr_now = optimizer.param_groups[0]['lr']
        if use_cosine:
            scheduler.step()
        else:
            scheduler.step(v_nce)

        is_best = v_nce < best_nce
        if is_best:
            best_nce = v_nce; best_epoch = epoch; best_vacc = v_acc; patience_counter = 0
            torch.save(model.state_dict(), os.path.join(PATHS['model_dir'], f'master_{name}_best.pt'))
        else:
            patience_counter += 1

        if verbose and (epoch % 20 == 0 or epoch == 1):
            print(f"  Epoch {epoch:3d} | NCE: {v_nce:.4f} | Acc: {v_acc:.4f} | "
                  f"Best: {best_nce:.4f} @{best_epoch} | LR: {lr_now:.2e}")

        if patience_counter >= patience:
            if verbose: print(f"  Early stop @ {epoch}")
            break

    model.load_state_dict(torch.load(os.path.join(PATHS['model_dir'], f'master_{name}_best.pt'), map_location=device))

    test_case_graphs = {cid: all_graphs[cid] for cid in test_ids if cid in all_graphs and cid in gt_relaxed}
    test_case_ids_list = list(test_case_graphs.keys())
    results = evaluate_retrieval_gnn(model, test_case_graphs, ent_graphs, ent_ids,
                                     test_case_ids_list, gt_relaxed, device)
    if verbose:
        for k in [5, 10, 20]:
            r = results[k]
            print(f"  Retrieval @{k:2d}: P={r['Precision']:.4f}  MRR={r['MRR']:.4f}  Hit={r['Hit']:.4f}")

    return {
        'name': name, 'seed': seed, 'ce_weight': ce_weight, 'nce_weight': nce_weight,
        'temperature': temperature, 'best_nce': best_nce, 'best_vacc': best_vacc,
        'best_epoch': best_epoch,
        'retrieval': {f'@{k}': results[k] for k in [5, 10, 20]}
    }

# ============================================================
# EXPERIMENT: Multi-Seed Stability
# ============================================================
def run_seed_stability(seeds=[42, 123, 456]):
    print("\n" + "=" * 70)
    print("[EXP 4] Multi-Seed Stability (V13 Joint x3)")
    print("=" * 70)
    all_results = []
    for s in seeds:
        print(f"\n--- Seed {s} ---")
        r = run_joint_training(f"seed_{s}", ce_weight=0.5, nce_weight=0.5,
                               temperature=0.5, seed=s, epochs=350, patience=80,
                               use_cosine=True, verbose=True)
        all_results.append(r)

    mrrs10 = [r['retrieval']['@10']['MRR'] for r in all_results]
    p5s = [r['retrieval']['@5']['Precision'] for r in all_results]
    print("\n=== Seed Stability Summary ===")
    print(f"  MRR@10: {np.mean(mrrs10):.4f} +- {np.std(mrrs10):.4f}")
    print(f"  P@5:    {np.mean(p5s):.4f} +- {np.std(p5s):.4f}")
    for i, s in enumerate(seeds):
        print(f"  Seed {s}: MRR@10={mrrs10[i]:.4f}")

    out = {'experiment': 'seed_stability', 'seeds': seeds,
           'mean_mrr10': float(np.mean(mrrs10)), 'std_mrr10': float(np.std(mrrs10)),
           'mean_p5': float(np.mean(p5s)), 'std_p5': float(np.std(p5s)),
           'individual': all_results}
    with open(os.path.join(PATHS['master_results_dir'], 'seed_stability.json'), 'w') as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=str)
    return out

# ============================================================
# EXPERIMENT: Lambda Sweep
# ============================================================
def run_lambda_sweep(lambdas=[0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]):
    print("\n" + "=" * 70)
    print("[EXP 5] Lambda Sweep (CE:NCE weight)")
    print("=" * 70)
    all_results = []
    for lam in lambdas:
        ce_w = lam; nce_w = 1.0 - lam
        label = f"lambda_{lam:.1f}"
        lstyle = 'Pure InfoNCE' if lam == 0.0 else ('Pure CE' if lam == 1.0 else f'Joint CE={lam:.1f}')
        print(f"\n--- {lstyle} ---")

        if lam == 1.0:
            # Pure CE — reuse pure GNN if available
            cache_path = os.path.join(PATHS['master_results_dir'], 'pure_gnn_baseline.json')
            if os.path.exists(cache_path):
                with open(cache_path) as f:
                    prev = json.load(f)
                r = {'name': label, 'seed': 42, 'ce_weight': 1.0, 'nce_weight': 0.0,
                     'temperature': 0.5, 'best_nce': float('inf'),
                     'best_vacc': prev['best_val_acc'], 'best_epoch': prev['best_epoch'],
                     'retrieval': prev['results']}
            else:
                r = run_joint_training(label, ce_weight=1.0, nce_weight=0.0,
                                       temperature=0.5, seed=42, epochs=200, patience=60,
                                       use_cosine=False, verbose=True)
        else:
            use_cos = (0 < lam < 1.0)
            ep, pat = (350, 80) if use_cos else (200, 60)
            r = run_joint_training(label, ce_weight=ce_w, nce_weight=nce_w,
                                   temperature=0.5, seed=42, epochs=ep, patience=pat,
                                   use_cosine=use_cos, verbose=True)
        all_results.append(r)

    print("\n=== Lambda Sweep Summary ===")
    print(f"  {'lambda_ce':>8s}  {'MRR@5':>8s}  {'MRR@10':>8s}  {'P@5':>8s}  {'P@10':>8s}  {'Hit@10':>8s}")
    print("  " + "-" * 60)
    best_mrr10 = max(r['retrieval']['@10']['MRR'] for r in all_results)
    for r in all_results:
        m5 = r['retrieval']['@5']['MRR']; m10 = r['retrieval']['@10']['MRR']
        p5 = r['retrieval']['@5']['Precision']; p10 = r['retrieval']['@10']['Precision']
        h10 = r['retrieval']['@10']['Hit']
        mark = ' <-- BEST' if abs(m10 - best_mrr10) < 1e-6 else ''
        print(f"  {r['ce_weight']:8.1f}  {m5:8.4f}  {m10:8.4f}  {p5:8.4f}  {p10:8.4f}  {h10:8.4f}{mark}")

    out = {'experiment': 'lambda_sweep', 'lambdas': lambdas, 'results': all_results}
    with open(os.path.join(PATHS['master_results_dir'], 'lambda_sweep.json'), 'w') as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=str)
    return all_results

# ============================================================
# EXPERIMENT: Temperature Ablation
# ============================================================
def run_temperature_sweep(taus=[0.03, 0.05, 0.07, 0.1, 0.3, 0.5]):
    print("\n" + "=" * 70)
    print("[EXP 6] Temperature Ablation")
    print("=" * 70)
    all_results = []
    for tau in taus:
        print(f"\n--- tau={tau} ---")
        r = run_joint_training(f"tau_{tau}", ce_weight=0.5, nce_weight=0.5,
                               temperature=tau, seed=42, epochs=350, patience=80,
                               use_cosine=True, verbose=True)
        all_results.append(r)

    print("\n=== Temperature Sweep Summary ===")
    print(f"  {'tau':>8s}  {'MRR@5':>8s}  {'MRR@10':>8s}  {'P@5':>8s}  {'P@10':>8s}  {'BestNCE':>10s}")
    print("  " + "-" * 60)
    for r in all_results:
        m5 = r['retrieval']['@5']['MRR']; m10 = r['retrieval']['@10']['MRR']
        p5 = r['retrieval']['@5']['Precision']; p10 = r['retrieval']['@10']['Precision']
        print(f"  {r['temperature']:8.3f}  {m5:8.4f}  {m10:8.4f}  {p5:8.4f}  {p10:8.4f}  {r['best_nce']:10.4f}")

    out = {'experiment': 'temperature_sweep', 'taus': taus, 'results': all_results}
    with open(os.path.join(PATHS['master_results_dir'], 'temperature_sweep.json'), 'w') as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=str)
    return all_results


# ============================================================
# EXPERIMENT: t-SNE Visualization
# ============================================================
def run_tsne_visualization():
    print("\n" + "=" * 70)
    print("[EXP 7] t-SNE Visualization")
    print("=" * 70)
    try:
        from sklearn.manifold import TSNE
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  sklearn/matplotlib not available, skipping.")
        return

    (train_graphs, val_graphs, test_graphs, all_graphs,
     num_classes, unique_types, _, _, test_ids,
     _, _, _, _, gt_relaxed) = load_all_data(verbose=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("  Training Joint (CE+NCE) model...")
    r_joint = run_joint_training("tsne_joint", ce_weight=0.5, nce_weight=0.5,
                                 temperature=0.5, seed=42, epochs=200, patience=60,
                                 use_cosine=True, verbose=False)
    joint_model = GINEncoder(num_node_types=len(NODE_TYPES), num_edge_types=len(EDGE_TYPES),
                             num_classes=num_classes, hidden_dim=128, num_layers=3,
                             dropout=0.3, with_classifier=True).to(device)
    joint_model.load_state_dict(torch.load(os.path.join(PATHS['model_dir'], 'master_tsne_joint_best.pt'), map_location=device))
    joint_model.eval()

    print("  Training Pure NCE model...")
    r_nce = run_joint_training("tsne_pure_nce", ce_weight=0.0, nce_weight=1.0,
                               temperature=0.5, seed=42, epochs=200, patience=60,
                               use_cosine=False, verbose=False)
    nce_model = GINEncoder(num_node_types=len(NODE_TYPES), num_edge_types=len(EDGE_TYPES),
                           num_classes=num_classes, hidden_dim=128, num_layers=3,
                           dropout=0.3, with_classifier=False).to(device)
    nce_model.load_state_dict(torch.load(os.path.join(PATHS['model_dir'], 'master_tsne_pure_nce_best.pt'), map_location=device), strict=False)
    nce_model.eval()

    test_case_ids_list = [cid for cid in test_ids if cid in all_graphs]
    test_graphs_list = [all_graphs[cid] for cid in test_case_ids_list]
    labels_arr = [all_graphs[cid].label for cid in test_case_ids_list]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, num_classes))

    models_plot = [('Joint (CE+NCE)', joint_model, r_joint),
                   ('Pure InfoNCE', nce_model, r_nce)]

    for ax_idx, (name, model, r) in enumerate(models_plot):
        embs = model.encode_batch(test_graphs_list, device, batch_size=128).numpy()
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
        embs_2d = tsne.fit_transform(embs)
        ax = axes[ax_idx]
        for lbl in range(num_classes):
            idx = [i for i, l in enumerate(labels_arr) if l == lbl]
            if idx:
                ax.scatter(embs_2d[idx, 0], embs_2d[idx, 1], c=[colors[lbl]],
                           label=unique_types[lbl][:20], alpha=0.6, s=8)
        ax.set_title(f'{name}\nMRR@10={r["retrieval"]["@10"]["MRR"]:.4f}', fontsize=12)
        ax.legend(loc='lower right', fontsize=6, ncol=2)

    plt.tight_layout()
    out_path = os.path.join(PATHS['master_results_dir'], 'tsne_joint_vs_pure_nce.png')
    plt.savefig(out_path, dpi=150)
    print(f"  Saved t-SNE to {out_path}")


# ============================================================
# FINAL SUMMARY
# ============================================================
def print_final_summary():
    print("\n\n")
    print("=" * 78)
    print("  MASTER EXPERIMENT RESULTS SUMMARY")
    print(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 78)
    print()
    print(f"  {'Experiment':<32s} {'MRR@5':>8s} {'MRR@10':>8s} {'P@5':>8s} {'P@10':>8s} {'Hit@10':>8s}")
    print("  " + "-" * 72)

    results_dir = PATHS['master_results_dir']

    for fname, label in [('bm25_baseline.json', 'BM25 (TF-IDF)'),
                          ('random_baseline.json', 'Random Embedding'),
                          ('pure_gnn_baseline.json', 'Pure GNN (CE only)')]:
        try:
            with open(os.path.join(results_dir, fname)) as f:
                d = json.load(f)
            mrr5 = d['results']['@5']['MRR']; mrr10 = d['results']['@10']['MRR']
            p5 = d['results']['@5']['Precision']; p10 = d['results']['@10']['Precision']
            h10 = d['results']['@10']['Hit']
            print(f"  {label:<32s} {mrr5:8.4f} {mrr10:8.4f} {p5:8.4f} {p10:8.4f} {h10:8.4f}")
        except Exception as e: print(f"  {label:<32s} --- ({e})")

    try:
        with open(os.path.join(results_dir, 'seed_stability.json')) as f:
            d = json.load(f)
        mm = [r['retrieval']['@10']['MRR'] for r in d['individual']]
        pp5 = [r['retrieval']['@5']['Precision'] for r in d['individual']]
        pp10 = [r['retrieval']['@10']['Precision'] for r in d['individual']]
        hh10 = [r['retrieval']['@10']['Hit'] for r in d['individual']]
        label = f'V13 Joint x{len(d["seeds"])} (mean)'
        print(f"  {label:<32s} {np.mean(mm):8.4f} {np.mean(mm):8.4f} {np.mean(pp5):8.4f} {np.mean(pp10):8.4f} {np.mean(hh10):8.4f}")
        label2 = '  (+- std)'
        print(f"  {label2:<32s} {np.std(mm):8.4f} {np.std(mm):8.4f} {np.std(pp5):8.4f} {np.std(pp10):8.4f} {np.std(hh10):8.4f}")
    except: pass

    try:
        with open(os.path.join(results_dir, 'lambda_sweep.json')) as f:
            d = json.load(f)
        print("\n  Lambda Sweep:")
        print(f"  {'  lambda_ce':>12s}  {'MRR@5':>8s} {'MRR@10':>8s} {'P@5':>8s} {'P@10':>8s}")
        best_m = max(r['retrieval']['@10']['MRR'] for r in d['results'])
        for r in d['results']:
            m5 = r['retrieval']['@5']['MRR']; m10 = r['retrieval']['@10']['MRR']
            p5 = r['retrieval']['@5']['Precision']; p10 = r['retrieval']['@10']['Precision']
            mk = ' <--' if abs(m10 - best_m) < 1e-6 else ''
            print(f"  {r['ce_weight']:12.1f}  {m5:8.4f} {m10:8.4f} {p5:8.4f} {p10:8.4f}{mk}")
    except: pass

    try:
        with open(os.path.join(results_dir, 'temperature_sweep.json')) as f:
            d = json.load(f)
        print("\n  Temperature Sweep:")
        print(f"  {'  tau':>12s}  {'MRR@5':>8s} {'MRR@10':>8s} {'P@5':>8s} {'P@10':>8s}")
        for r in d['results']:
            m5 = r['retrieval']['@5']['MRR']; m10 = r['retrieval']['@10']['MRR']
            p5 = r['retrieval']['@5']['Precision']; p10 = r['retrieval']['@10']['Precision']
            print(f"  {r['temperature']:12.3f}  {m5:8.4f} {m10:8.4f} {p5:8.4f} {p10:8.4f}")
    except: pass

    print()
    print(f"  Results saved to: {results_dir}/")
    print()


# ============================================================
# MAIN
# ============================================================
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    parser = argparse.ArgumentParser(description='Master Experiment Suite for ICSC2026')
    parser.add_argument('--exp', type=str, default='all',
                        choices=['all', 'bm25', 'random', 'baseline', 'pure_gnn',
                                 'seeds', 'lambda', 'tau', 'tsne', 'full_eval', 'summary'],
                        help='Experiment to run')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    args = parser.parse_args()

    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        print(f"GPU {args.gpu}: {torch.cuda.get_device_name(args.gpu)}")
    print(f"Experiment: {args.exp}")

    exp = args.exp

    if exp in ('all', 'bm25', 'baseline'):
        run_bm25_baseline()
        run_random_baseline()

    if exp in ('all', 'pure_gnn', 'baseline'):
        run_pure_gnn_baseline(seed=42)

    if exp in ('all', 'seeds'):
        run_seed_stability([42, 123, 456])

    if exp in ('all', 'lambda'):
        run_lambda_sweep([0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0])

    if exp in ('all', 'tau'):
        run_temperature_sweep([0.03, 0.05, 0.07, 0.1, 0.3, 0.5])

    if exp in ('all', 'tsne'):
        run_tsne_visualization()

    if exp in ('all', 'full_eval'):
        # Just train V13 and do full retrieval
        print("\n[EXP 8] Full Retrieval (V13 Joint)")
        (train_graphs, val_graphs, test_graphs, all_graphs,
         num_classes, unique_types, train_ids, val_ids, test_ids,
         case_data, ent_graphs, ent_ids, ent_case_types, gt_relaxed) = load_all_data(verbose=True)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        r = run_joint_training("full_eval", ce_weight=0.5, nce_weight=0.5,
                               temperature=0.5, seed=42, epochs=350, patience=80,
                               use_cosine=True, verbose=True)
        model = GINEncoder(num_node_types=len(NODE_TYPES), num_edge_types=len(EDGE_TYPES),
                           num_classes=num_classes, hidden_dim=128, num_layers=3,
                           dropout=0.3, with_classifier=True).to(device)
        model.load_state_dict(torch.load(os.path.join(PATHS['model_dir'], 'master_full_eval_best.pt'), map_location=device))
        model.eval()
        test_case_graphs = {cid: all_graphs[cid] for cid in test_ids if cid in all_graphs and cid in gt_relaxed}
        test_case_ids_list = list(test_case_graphs.keys())
        results = evaluate_retrieval_gnn(model, test_case_graphs, ent_graphs, ent_ids,
                                         test_case_ids_list, gt_relaxed, device)
        print("\n  Full Retrieval on all enterprises:")
        for k in [5, 10, 20]:
            r = results[k]
            print(f"  @{k:2d}: P={r['Precision']:.4f}+-{r['Precision_std']:.4f}  "
                  f"MRR={r['MRR']:.4f}+-{r['MRR_std']:.4f}  Hit={r['Hit']:.4f}")
        out = {'experiment': 'full_retrieval', 'n_enterprises': len(ent_ids),
               'n_test_cases': len(test_case_ids_list),
               'results': {f'@{k}': results[k] for k in [5, 10, 20]}}
        with open(os.path.join(PATHS['master_results_dir'], 'full_retrieval.json'), 'w') as f:
            json.dump(out, f, ensure_ascii=False, indent=2)

    if exp in ('all', 'summary'):
        print_final_summary()

    if exp == 'all':
        print_final_summary()

if __name__ == '__main__':
    main()
