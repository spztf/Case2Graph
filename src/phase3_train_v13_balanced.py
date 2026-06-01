#!/usr/bin/env python3
"""
Phase 3 v13 — Balanced Joint Training (CE + InfoNCE) + Cosine Annealing
=========================================================================
Key changes over V12:
  - λ_ce=0.5, λ_nce=0.5 (balanced classification + contrastive objectives)
  - CosineAnnealingWarmRestarts replaces ReduceLROnPlateau (periodic restarts
    to escape local minima — V12 NCE plateaued from epoch 11 onward)
  - SMOTE disabled (V12 showed no benefit; implementation had gradient-flow bug)
  - Label smoothing reduced: 0.1 → 0.05 (preserve classification accuracy)
  - Extended patience 50→80, epochs 300→350 to match cosine cycle (T₀=60)
  - LR cycles: 1e-3 → 1e-6 cosine decay over 60 epochs, then restart

Rationale: V12 showed InfoNCE converging too early (epoch 11) with CE signal
too weak to guide. V13 balances both losses equally and uses cosine restarts
to periodically "shake" the model out of sub-optimal plateaus.
"""

import json, os, sys, time, math, copy, random, gc
from collections import defaultdict, Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

# ═══════════════════════════════════════════════════════════
CONFIG = {
    'case_graphs_dir': '/root/fd/workspace/paper/data/kg/case_graphs',
    'enterprise_subgraphs_dir': '/root/fd/workspace/paper/data/kg/enterprise_subgraphs',
    'output_dir': '/root/fd/workspace/paper/results/task_c',
    'model_dir': '/root/fd/workspace/paper/models',
    'split_meta_path': '/root/fd/workspace/paper/results/task_c/split_meta_v9.json',
    'vocab_path': '/root/fd/workspace/paper/V10/node_vocabs.json',
    'device': 'cuda',
    'seed': 42,
    'hidden_dim': 128,
    'num_layers': 3,
    'dropout': 0.3,
    'max_nodes': 200,
    'ent_max_nodes': 500,
    # ── Joint training ──
    'joint_epochs': 350,
    'joint_lr': 1e-3,
    'joint_batch_size': 128,
    'joint_patience': 80,
    # ── Loss weights (balanced 0.5:0.5) ──
    'ce_weight': 0.5,
    'nce_weight': 0.5,
    'label_smoothing': 0.05,   # reduced from 0.1 to preserve acc
    # ── InfoNCE ──
    'temperature': 0.5,
    # ── Cosine Annealing ──
    'cosine_T0': 60,           # first restart after 60 epochs
    'cosine_Tmult': 2,         # each subsequent cycle 2× longer
    'cosine_eta_min': 1e-6,
    # ── SMOTE (DISABLED in V13 — set smote_interval=0) ──
    'smote_interval': 0,
    'smote_k': 5,
    'smote_alpha': 0.3,
    'smote_finetune_steps': 10,
    # ── Optimizer ──
    'weight_decay': 1e-4,
    'grad_clip': 1.0,
    # Normalization stats for log_value (precomputed)
    'log_value_mean': 2.022,
    'log_value_std': 1.197,
}

os.makedirs(CONFIG['output_dir'], exist_ok=True)
os.makedirs(CONFIG['model_dir'], exist_ok=True)
np.random.seed(CONFIG['seed'])
torch.manual_seed(CONFIG['seed'])
random.seed(CONFIG['seed'])
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(CONFIG['seed'])

NODE_TYPES = ['Company', 'Person', 'Authority', 'ViolationEvent',
              'FinancialFlow', 'Location', 'Industry']
EDGE_TYPES = ['located_in', 'in_industry', 'penalized_by', 'committed',
              'controlled_by', 'involves_amount', 'co_penalized']
node_type_to_idx = {t: i for i, t in enumerate(NODE_TYPES)}
edge_type_to_idx = {t: i for i, t in enumerate(EDGE_TYPES)}

# ═══════════════════════════════════════════════════════════
# Load vocabularies
with open(CONFIG['vocab_path']) as f:
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
print(f"Loaded vocabularies: { {k: v for k, v in VOCAB_SIZES.items()} }")

def degree_to_bucket(deg):
    if deg <= 0: return 0
    if deg == 1: return 1
    if deg <= 3: return 2
    if deg <= 5: return 3
    if deg <= 7: return 4
    if deg <= 15: return 5
    if deg <= 31: return 6
    return 7

# ═══════════════════════════════════════════════════════════
class GraphData:
    __slots__ = ('node_types', 'edge_index', 'edge_types', 'node_attrs',
                 'num_nodes', 'num_edges', 'graph_id', 'label', 'metadata')
    def __init__(self, node_types, edge_index, edge_types, node_attrs,
                 graph_id=None, label=None, metadata=None):
        self.node_types = node_types
        self.edge_index = edge_index
        self.edge_types = edge_types
        self.node_attrs = node_attrs
        self.num_nodes = len(node_types)
        self.num_edges = edge_index.shape[1] if edge_index.numel() > 0 else 0
        self.graph_id = graph_id
        self.label = label
        self.metadata = metadata or {}

def json_graph_to_data_v13(graph_dict, graph_id=None, label=None, max_nodes=None):
    """
    Convert JSON graph to GraphData with node feature attributes.
    (Same as V12 — renamed for version clarity)
    """
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
            node_attrs[i, 3] = db
            node_attrs[i, 4] = 0.0
            node_attrs[i, 5] = 0.0
        elif nt == 'Person':
            node_attrs[i, 0] = 0
            node_attrs[i, 1] = 0
            node_attrs[i, 2] = VOCAB_MAP['role'].get(props.get('role', ''), 0)
            node_attrs[i, 3] = db
            node_attrs[i, 4] = 0.0
            node_attrs[i, 5] = 0.0
        elif nt == 'ViolationEvent':
            node_attrs[i, 0] = VOCAB_MAP['case_type'].get(props.get('case_type', ''), 0)
            node_attrs[i, 1] = VOCAB_MAP['violation_item'].get(props.get('violation_item', ''), 0)
            node_attrs[i, 2] = 0
            node_attrs[i, 3] = db
            node_attrs[i, 4] = 0.0
            node_attrs[i, 5] = 0.0
        elif nt == 'FinancialFlow':
            node_attrs[i, 0] = VOCAB_MAP['flow_type'].get(props.get('type', ''), 0)
            node_attrs[i, 1] = 0
            node_attrs[i, 2] = 0
            node_attrs[i, 3] = db
            raw_val = props.get('value', None)
            if raw_val is not None and raw_val > 0:
                log_val = np.log10(raw_val + 1)
                node_attrs[i, 4] = (log_val - CONFIG['log_value_mean']) / CONFIG['log_value_std']
            else:
                node_attrs[i, 4] = 0.0
            node_attrs[i, 5] = 0.0
        else:  # Location, Industry, Authority, Unknown
            node_attrs[i, 0] = 0
            node_attrs[i, 1] = 0
            node_attrs[i, 2] = 0
            node_attrs[i, 3] = db
            node_attrs[i, 4] = 0.0
            node_attrs[i, 5] = 0.0

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
        node_attrs=node_attrs, graph_id=graph_id, label=label
    )

# ═══════════════════════════════════════════════════════════
# Node Feature Encoder (same as V12)
class NodeFeatureEncoder(nn.Module):
    def __init__(self, hidden_dim=128, feat_dim=32):
        super().__init__()
        fd = feat_dim
        self.emb_0 = nn.Embedding(max(VOCAB_SIZES['industry'], VOCAB_SIZES['case_type'], VOCAB_SIZES['flow_type']), fd)
        self.emb_1 = nn.Embedding(max(VOCAB_SIZES['province'], VOCAB_SIZES['violation_item']), fd)
        self.emb_2 = nn.Embedding(VOCAB_SIZES['role'], fd)
        self.emb_3 = nn.Embedding(8, fd // 2)
        self.value_enc = nn.Sequential(
            nn.Linear(1, fd // 2), nn.ReLU(inplace=True), nn.Linear(fd // 2, fd // 2)
        )
        total_feat = 4 * fd
        self.project = nn.Sequential(
            nn.Linear(total_feat, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(inplace=True)
        )

    def forward(self, node_attrs):
        B, N, _ = node_attrs.shape
        device = node_attrs.device

        idx_0 = node_attrs[:, :, 0].long().clamp(0, self.emb_0.num_embeddings - 1)
        idx_1 = node_attrs[:, :, 1].long().clamp(0, self.emb_1.num_embeddings - 1)
        idx_2 = node_attrs[:, :, 2].long().clamp(0, self.emb_2.num_embeddings - 1)
        idx_3 = node_attrs[:, :, 3].long().clamp(0, 7)

        e0 = self.emb_0(idx_0)
        e1 = self.emb_1(idx_1)
        e2 = self.emb_2(idx_2)
        e3 = self.emb_3(idx_3)

        val = node_attrs[:, :, 4:5]
        e4 = self.value_enc(val)

        feats = torch.cat([e0, e1, e2, e3, e4], dim=-1)
        return self.project(feats)

# ═══════════════════════════════════════════════════════════
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

class GINEncoderV13(nn.Module):
    def __init__(self, num_node_types, num_edge_types, num_classes,
                 hidden_dim=128, num_layers=3, dropout=0.3):
        super().__init__()
        self.node_embed = nn.Embedding(num_node_types, hidden_dim)
        self.edge_embed = nn.Embedding(num_edge_types, hidden_dim)
        self.feat_encoder = NodeFeatureEncoder(hidden_dim=hidden_dim, feat_dim=32)
        self.convs = nn.ModuleList([
            EdgeTypeGINConv(hidden_dim) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2), nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim), nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim))
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
        graph_emb = F.normalize(graph_emb, p=2, dim=1)
        if return_logits:
            return graph_emb, self.classifier(graph_emb)
        return graph_emb

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

# ═══════════════════════════════════════════════════════════
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

# ═══════════════════════════════════════════════════════════
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

# ═══════════════════════════════════════════════════════════
# JOINT TRAINING: CE + InfoNCE
def train_joint(model, dataloader, optimizer, device):
    """
    Single training step: compute both CE and InfoNCE losses.
    Returns (avg_total_loss, avg_ce_loss, avg_nce_loss, ce_accuracy)
    """
    model.train()
    total_loss_sum, ce_sum, nce_sum = 0.0, 0.0, 0.0
    correct, total_samples = 0, 0
    num_batches = 0

    for batch in dataloader:
        nt = batch['node_types'].to(device); ei = batch['edge_index'].to(device)
        et = batch['edge_types'].to(device); na = batch['node_attrs'].to(device)
        mask = batch['mask'].to(device); labels = batch['labels'].to(device)

        optimizer.zero_grad()

        # Forward: get both embeddings and logits
        embeddings, logits = model(nt, ei, et, na, mask, return_logits=True)

        # CE loss with label smoothing
        ce_loss = F.cross_entropy(logits, labels, label_smoothing=CONFIG['label_smoothing'])

        # InfoNCE loss
        nce_loss = info_nce_loss(embeddings, labels, CONFIG['temperature'])

        # Weighted sum
        total_loss = CONFIG['ce_weight'] * ce_loss + CONFIG['nce_weight'] * nce_loss

        if torch.isfinite(total_loss):
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['grad_clip'])
            optimizer.step()

            total_loss_sum += total_loss.item()
            ce_sum += ce_loss.item()
            nce_sum += nce_loss.item()
            num_batches += 1

        # Track CE accuracy
        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total_samples += labels.size(0)

    if num_batches == 0:
        return 0.0, 0.0, 0.0, 0.0

    n = max(num_batches, 1)
    return (total_loss_sum / n, ce_sum / n, nce_sum / n,
            correct / max(total_samples, 1))


@torch.no_grad()
def validate_joint(model, dataloader, device):
    """
    Validation: compute CE accuracy + InfoNCE loss.
    Returns (avg_nce_loss, ce_accuracy, avg_ce_loss)
    """
    model.eval()
    ce_sum, nce_sum = 0.0, 0.0
    correct, total_samples = 0, 0
    num_batches = 0

    for batch in dataloader:
        nt = batch['node_types'].to(device); ei = batch['edge_index'].to(device)
        et = batch['edge_types'].to(device); na = batch['node_attrs'].to(device)
        mask = batch['mask'].to(device); labels = batch['labels'].to(device)

        embeddings, logits = model(nt, ei, et, na, mask, return_logits=True)

        ce_loss = F.cross_entropy(logits, labels, label_smoothing=CONFIG['label_smoothing'])
        nce_loss = info_nce_loss(embeddings, labels, CONFIG['temperature'])

        if torch.isfinite(nce_loss):
            nce_sum += nce_loss.item()
            ce_sum += ce_loss.item()
            num_batches += 1

        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total_samples += labels.size(0)

    if num_batches == 0:
        return float('inf'), 0.0, 0.0

    n = max(num_batches, 1)
    return (nce_sum / n, correct / max(total_samples, 1), ce_sum / n)


# ═══════════════════════════════════════════════════════════
@torch.no_grad()
def evaluate_retrieval(model, case_graphs, ent_graphs, ent_ids,
                       case_ids, ground_truth, device, k_values=[5, 10, 20]):
    model.eval()
    case_list = [case_graphs[cid] for cid in case_ids]
    case_embs = model.encode_batch(case_list, device, batch_size=64)
    ent_list = [ent_graphs[eid] for eid in ent_ids]
    ent_embs = model.encode_batch(ent_list, device, batch_size=32)
    case_embs = F.normalize(case_embs, p=2, dim=1)
    ent_embs = F.normalize(ent_embs, p=2, dim=1)
    sim = torch.mm(case_embs, ent_embs.t())
    metrics = {k: {'precision': [], 'recall': [], 'hit': [], 'mrr': []} for k in k_values}
    for i, cid in enumerate(case_ids):
        sims = sim[i]; ranked = torch.argsort(sims, descending=True)
        gt = ground_truth.get(cid, set())
        for k in k_values:
            top_k = {ent_ids[idx] for idx in ranked[:k].tolist()}
            hits = len(top_k & gt)
            precision = hits / k
            recall = hits / len(gt) if len(gt) > 0 else 0
            hit = 1.0 if hits > 0 else 0.0
            mrr = 0.0
            for rank, idx in enumerate(ranked[:k].tolist(), 1):
                if ent_ids[idx] in gt:
                    mrr = 1.0 / rank; break
            metrics[k]['precision'].append(precision)
            metrics[k]['recall'].append(recall)
            metrics[k]['hit'].append(hit); metrics[k]['mrr'].append(mrr)
    results = {}
    for k in k_values:
        results[k] = {
            'Precision': float(np.mean(metrics[k]['precision'])),
            'Recall': float(np.mean(metrics[k]['recall'])),
            'MRR': float(np.mean(metrics[k]['mrr'])),
            'Hit': float(np.mean(metrics[k]['hit']))}
    return results, metrics


# ═══════════════════════════════════════════════════════════
def main():
    print("=" * 70)
    print("Phase 3 v13 — Balanced Joint Training + Cosine Annealing")
    print("=" * 70)
    device = torch.device(CONFIG['device'] if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)} "
              f"({torch.cuda.get_device_properties(0).total_memory / 1024**3:.0f} GB)")
    print(f"Device: {device}")
    print(f"Loss weights: CE={CONFIG['ce_weight']}, NCE={CONFIG['nce_weight']}")
    print(f"Label smoothing: {CONFIG['label_smoothing']}")
    print(f"Cosine Annealing: T0={CONFIG['cosine_T0']}, Tmult={CONFIG['cosine_Tmult']}, "
          f"eta_min={CONFIG['cosine_eta_min']}")
    print(f"SMOTE: {'ENABLED' if CONFIG['smote_interval'] > 0 else 'DISABLED'}")

    # ── Load split_meta_v9 ──
    print("\n[1] Loading split_meta_v9.json...")
    with open(CONFIG['split_meta_path']) as f:
        meta = json.load(f)
    train_ids = meta['train_ids']; val_ids = meta['val_ids']; test_ids = meta['test_ids']
    case_data = meta['case_data']; unique_types = meta['unique_types']
    type_to_label = {t: i for i, t in enumerate(unique_types)}
    num_classes = len(unique_types)
    print(f"  Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")
    print(f"  Classes: {num_classes}")
    for i, t in enumerate(unique_types):
        cnt = sum(1 for c in train_ids if case_data.get(c, {}).get('sub_type', '') == t)
        print(f"    [{i}] {t}: train={cnt}")

    # ── Load case graphs ──
    print("\n[2] Loading case graphs (with node features)...")
    cg_dir = CONFIG['case_graphs_dir']; all_fnames = set(os.listdir(cg_dir))
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
        g = json_graph_to_data_v13(raw.get('graph', raw), graph_id=cid,
                                   label=label, max_nodes=CONFIG['max_nodes'])
        all_graphs[cid] = g; labels_map[cid] = label
    print(f"  Loaded {len(all_graphs)} graphs (skipped {skipped})")

    train_graphs = [all_graphs[c] for c in train_ids if c in all_graphs]
    val_graphs = [all_graphs[c] for c in val_ids if c in all_graphs]
    test_graphs = [all_graphs[c] for c in test_ids if c in all_graphs]

    na_samples = [g.node_attrs for g in train_graphs[:100]]
    nonzero_counts = [(na.abs() > 1e-6).sum(dim=-1).float().mean() for na in na_samples]
    avg_nz = sum(nonzero_counts) / len(nonzero_counts)
    print(f"  Node attrs: {len(na_samples)} graphs, avg nonzero slots/node: {avg_nz:.1f}/6")

    # ── Balanced sampler ──
    train_labels_list = [g.label for g in train_graphs]
    label_counts = Counter(train_labels_list)
    class_weights = {lbl: 1.0 / max(cnt, 1) for lbl, cnt in label_counts.items()}
    sample_weights = [class_weights[lbl] for lbl in train_labels_list]
    balanced_sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_graphs), replacement=True)

    train_loader = DataLoader(train_graphs, batch_size=CONFIG['joint_batch_size'],
                              sampler=balanced_sampler, collate_fn=collate_fn, drop_last=True)
    val_loader = DataLoader(val_graphs, batch_size=CONFIG['joint_batch_size'],
                            shuffle=False, collate_fn=collate_fn)

    # ── Build model ──
    print("\n[3] Building GIN Encoder v13 (Balanced Joint Training)...")
    model = GINEncoderV13(num_node_types=len(NODE_TYPES), num_edge_types=len(EDGE_TYPES),
                          num_classes=num_classes, hidden_dim=CONFIG['hidden_dim'],
                          num_layers=CONFIG['num_layers'], dropout=CONFIG['dropout']).to(device)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ═══ Joint Training (CE + InfoNCE) ═══
    print("\n" + "=" * 70)
    print("[Joint Training] Balanced CE + InfoNCE + Cosine Annealing (No SMOTE)")
    print(f"  λ_ce={CONFIG['ce_weight']}, λ_nce={CONFIG['nce_weight']}, "
          f"label_smooth={CONFIG['label_smoothing']}")
    print(f"  Cosine: T0={CONFIG['cosine_T0']}, Tmult={CONFIG['cosine_Tmult']}, "
          f"eta_min={CONFIG['cosine_eta_min']}")
    print("=" * 70)

    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['joint_lr'],
                                 weight_decay=CONFIG['weight_decay'])
    # CosineAnnealingWarmRestarts: periodic restarts to escape local minima
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=CONFIG['cosine_T0'], T_mult=CONFIG['cosine_Tmult'],
        eta_min=CONFIG['cosine_eta_min'])

    best_nce_loss = float('inf')
    best_epoch = 0
    best_vacc = 0.0
    patience_counter = 0

    hist = {'train_total': [], 'train_ce': [], 'train_nce': [], 'train_acc': [],
            'val_nce': [], 'val_ce': [], 'val_acc': [], 'lr': []}

    for epoch in range(1, CONFIG['joint_epochs'] + 1):
        # ── Train ──
        t_total, t_ce, t_nce, t_acc = train_joint(
            model, train_loader, optimizer, device)

        # ── Validate ──
        v_nce, v_acc, v_ce = validate_joint(model, val_loader, device)

        # ── Log ──
        hist['train_total'].append(t_total)
        hist['train_ce'].append(t_ce)
        hist['train_nce'].append(t_nce)
        hist['train_acc'].append(t_acc)
        hist['val_nce'].append(v_nce)
        hist['val_ce'].append(v_ce)
        hist['val_acc'].append(v_acc)
        lr_now = optimizer.param_groups[0]['lr']
        hist['lr'].append(lr_now)

        # ── SMOTE (DISABLED in V13) ──
        # V12 showed SMOTE had no benefit on NCE; also had gradient-flow issues
        # (synthetic embeddings passed through projection layer incorrectly).
        # Will revisit as a separate ablation in V14+.

        # ── LR schedule (Cosine: step every epoch) ──
        scheduler.step()

        # ── Best model tracking (by InfoNCE validation loss) ──
        is_best = v_nce < best_nce_loss
        if is_best:
            best_nce_loss = v_nce
            best_epoch = epoch
            best_vacc = v_acc
            patience_counter = 0
            torch.save(model.state_dict(),
                       os.path.join(CONFIG['model_dir'], 'phase3_v13_balanced_best.pt'))
        else:
            patience_counter += 1

        # ── Print progress ──
        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | "
                  f"CE: {t_ce:.4f}/{v_ce:.4f} | "
                  f"NCE: {t_nce:.4f}/{v_nce:.4f} | "
                  f"Acc: {t_acc:.4f}/{v_acc:.4f} | "
                  f"Best NCE: {best_nce_loss:.4f} @{best_epoch} | "
                  f"LR: {lr_now:.2e}")

        # ── Cosine restart notification ──
        if epoch > 1 and lr_now > hist['lr'][-2] * 1.5:
            print(f"  🔄 Cosine restart @ epoch {epoch} (LR: {hist['lr'][-2]:.2e} → {lr_now:.2e})")

        # ── Early stop ──
        if patience_counter >= CONFIG['joint_patience']:
            print(f"  Early stop @ {epoch} (patience={CONFIG['joint_patience']})")
            break

    # ── Load best model ──
    model.load_state_dict(torch.load(
        os.path.join(CONFIG['model_dir'], 'phase3_v13_balanced_best.pt'),
        map_location=device))
    print(f"\n  Joint training done.")
    print(f"  Best Val NCE: {best_nce_loss:.4f} @ epoch {best_epoch}")
    print(f"  Best Val Acc:  {best_vacc:.4f}")

    # ═══ Phase C: Encode + Save ═══
    print("\n" + "=" * 70)
    print("[Encode] Encoding all case graphs...")
    print("=" * 70)
    model.eval()
    all_case_list = [all_graphs[c] for c in train_ids + val_ids + test_ids if c in all_graphs]
    all_embs = model.encode_batch(all_case_list, device, batch_size=128)
    emb_path = os.path.join(CONFIG['output_dir'], 'embeddings_v13_balanced.npz')
    np.savez_compressed(emb_path, embeddings=all_embs.numpy())
    print(f"  Saved {all_embs.shape} to {emb_path}")

    # ═══ Mini retrieval eval (100 ents) ═══
    print("\n[4] Quick retrieval check (100 enterprises)...")
    ent_dir = CONFIG['enterprise_subgraphs_dir']
    ent_fnames = sorted(os.listdir(ent_dir))[:100]
    ent_graphs_loaded = {}; ent_case_types = {}
    for fname in ent_fnames:
        eid = fname.replace('.json', '')
        try:
            with open(os.path.join(ent_dir, fname), encoding='utf-8') as f:
                raw = json.load(f)
        except: continue
        ct_counts = Counter()
        for node in raw.get('nodes', []):
            if node.get('type') == 'ViolationEvent':
                ct = node.get('properties', {}).get('case_type', '')
                if ct and ct != 'Unknown': ct_counts[ct] += 1
        primary_ct = ct_counts.most_common(1)[0][0] if ct_counts else 'Unknown'
        ent_case_types[eid] = primary_ct
        ent_graphs_loaded[eid] = json_graph_to_data_v13(raw, graph_id=eid,
                                                         max_nodes=CONFIG['ent_max_nodes'])

    ent_ids = list(ent_graphs_loaded.keys())
    def sub_type_to_case_type(st):
        for ct in ['虚开增值税专用发票', '虚开发票', '隐匿收入', '骗取出口退税', '转让定价', '其他']:
            if ct in st: return ct
        return st

    gt_relaxed = {}
    for cid in test_ids:
        if cid not in all_graphs: continue
        coarse_ct = sub_type_to_case_type(case_data.get(cid, {}).get('sub_type', 'Unknown'))
        matches = set()
        for eid, ect in ent_case_types.items():
            if ect == coarse_ct: matches.add(eid)
        gt_relaxed[cid] = matches

    test_case_graphs = {cid: all_graphs[cid] for cid in test_ids
                        if cid in all_graphs and cid in gt_relaxed}
    test_case_ids = list(test_case_graphs.keys())
    if test_case_ids and ent_ids:
        results, _ = evaluate_retrieval(model, test_case_graphs, ent_graphs_loaded,
                                        ent_ids, test_case_ids, gt_relaxed, device)
        print("  Quick Retrieval (Relaxed GT, 100 ents):")
        for k in [5, 10, 20]:
            r = results[k]
            print(f"    @{k:2d}: P={r['Precision']:.4f}  MRR={r['MRR']:.4f}  Hit={r['Hit']:.4f}")
    else:
        results = {k: {'Precision': 0, 'MRR': 0, 'Hit': 0} for k in [5, 10, 20]}

    # ── Save results ──
    final_results = {
        'experiment': 'phase3_v13_balanced_joint',
        'version': 'v13',
        'description': 'Balanced joint training (CE=0.5, NCE=0.5) with CosineAnnealingWarmRestarts, no SMOTE',
        'training': {
            'best_val_nce': best_nce_loss,
            'best_val_acc': best_vacc,
            'best_epoch': best_epoch,
            'history': hist
        },
        'retrieval': {f'@{k}': results[k] for k in [5, 10, 20]},
        'config': {k: str(v) if not isinstance(v, (int, float, bool, list, type(None)))
                   else v for k, v in CONFIG.items()},
        'unique_types': unique_types,
        'vocab_sizes': VOCAB_SIZES
    }
    with open(os.path.join(CONFIG['output_dir'], 'phase3_v13_balanced_results.json'), 'w') as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)
    torch.save({'model_state_dict': model.state_dict(), 'config': CONFIG,
                'unique_types': unique_types, 'type_to_label': type_to_label},
               os.path.join(CONFIG['model_dir'], 'phase3_v13_balanced_final.pt'))
    print("\n✅ Phase 3 v13 Complete!")
    print(f"  Best Val NCE: {best_nce_loss:.4f}")
    print(f"  Best Val Acc: {best_vacc:.4f}")
    print(f"  MRR@10: {results[10]['MRR']:.4f}")


if __name__ == '__main__':
    main()
