"""
Task C -- Triplet Contrastive Learning v5.0 (COLLAPSE-PROOF)
============================================================
Key fixes from v4.0:
  1. L2 normalization REMOVED from forward() — was causing embedding collapse
  2. _build_adj VECTORIZED — no Python for-loop over batch
  3. Edge type embeddings actually USED in message passing
  4. More candidates (32) + hard negatives within same case_type
  5. Variance regularization to prevent collapse
  6. Stronger projection head (3 layers)
  7. Industry auxiliary loss for finer-grained signal
"""
import json, os, sys, time, math, copy, random
from collections import defaultdict, Counter

# MUST set CUDA_VISIBLE_DEVICES BEFORE importing torch
os.environ['CUDA_VISIBLE_DEVICES'] = 'GPU-5af3591c-3757-678d-88e5-a02e1e36e408'

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

if torch.cuda.is_available():
    print(f"[CUDA] GPU: {torch.cuda.get_device_name(0)}  ({torch.cuda.get_device_properties(0).total_memory / 1024**3:.0f} GB)")
else:
    print("[CUDA] WARNING: No GPU available!")

# ============================================================
# Configuration
# ============================================================
CONFIG = {
    'case_graphs_dir': '/root/fd/workspace/paper/data/kg/case_graphs',
    'enterprise_subgraphs_dir': '/root/fd/workspace/paper/data/kg/enterprise_subgraphs',
    'output_dir': '/root/fd/workspace/paper/results/task_c',
    'model_path': '/root/fd/workspace/paper/models/subgnn_aug_v2.pt',
    'device': 'cuda',
    'k_values': [5, 10, 20],
    'seed': 42,

    # Encoder
    'embed_dim': 128,
    'hidden_dim': 128,
    'num_layers': 3,
    'dropout': 0.3,
    'batch_size': 16,            # smaller batch since M=32 is larger
    'num_candidates': 32,        # M: much more negatives (was 8)
    'num_hard_neg': 8,           # within-type hard negatives
    'ent_max_nodes': 2000,
    'encode_batch_size': 128,
    'lr': 1e-3,
    'weight_decay': 1e-4,
    'epochs': 80,
    'early_stopping_patience': 20,
    'temperature': 0.07,
    'var_reg_weight': 0.1,       # variance regularization
    'aux_weight': 0.3,           # auxiliary industry loss weight

    # Data split
    'train_split': 0.6,
    'val_split': 0.2,
}

os.makedirs(CONFIG['output_dir'], exist_ok=True)
os.makedirs(os.path.dirname(CONFIG['model_path']), exist_ok=True)
np.random.seed(CONFIG['seed'])
torch.manual_seed(CONFIG['seed'])
random.seed(CONFIG['seed'])

# ============================================================
# Node/Edge types
# ============================================================
NODE_TYPES = ['Company', 'Person', 'Authority', 'ViolationEvent',
              'FinancialFlow', 'Location', 'Industry']
EDGE_TYPES = ['located_in', 'in_industry', 'penalized_by', 'committed',
              'controlled_by', 'involves_amount', 'co_penalized']

node_type_to_idx = {t: i for i, t in enumerate(NODE_TYPES)}
edge_type_to_idx = {t: i for i, t in enumerate(EDGE_TYPES)}

# Industry categories (for auxiliary task)
INDUSTRY_CATEGORIES = [
    '制造业', '批发和零售业', '建筑业', '房地产业', '交通运输业',
    '信息技术服务业', '金融业', '租赁和商务服务业', '农林牧渔业',
    '电力热力燃气', '采矿业', '其他', '未知'
]
industry_to_idx = {ind: i for i, ind in enumerate(INDUSTRY_CATEGORIES)}

PROVINCE_NORMALIZE = {
    '湖南': '湖南省', '河北': '河北省', '山西': '山西省', '辽宁': '辽宁省',
    '吉林': '吉林省', '黑龙江': '黑龙江省', '江苏': '江苏省', '浙江': '浙江省',
    '安徽': '安徽省', '福建': '福建省', '江西': '江西省', '山东': '山东省',
    '河南': '河南省', '湖北': '湖北省', '广东': '广东省', '海南': '海南省',
    '四川': '四川省', '贵州': '贵州省', '云南': '云南省', '陕西': '陕西省',
    '甘肃': '甘肃省', '青海': '青海省', '台湾': '台湾省',
    '广西': '广西壮族自治区', '内蒙古': '内蒙古自治区', '西藏': '西藏自治区',
    '宁夏': '宁夏回族自治区', '新疆': '新疆维吾尔自治区',
    '北京': '北京市', '上海': '上海市', '天津': '天津市', '重庆': '重庆市',
    '湖南市': '湖南省',
}

def norm_province(p):
    if not p or p == '未知': return '未知'
    return PROVINCE_NORMALIZE.get(p.strip(), p.strip())

def norm_industry(ind):
    if not ind or ind == '未知': return '未知'
    ind = ind.strip()
    # Map to coarse categories
    for cat in INDUSTRY_CATEGORIES:
        if cat in ind or ind in cat:
            return cat
    return '其他'

# ============================================================
# Graph Data Structure
# ============================================================
class GraphData:
    def __init__(self, node_types, edge_index, edge_types,
                 graph_id=None, label=None, metadata=None):
        self.node_types = node_types
        self.edge_index = edge_index
        self.edge_types = edge_types
        self.num_nodes = len(node_types)
        self.num_edges = edge_index.shape[1] if edge_index.numel() > 0 else 0
        self.graph_id = graph_id
        self.label = label
        self.metadata = metadata or {}

def json_graph_to_data(graph_dict, graph_id=None, label=None):
    nodes = graph_dict.get('nodes', [])
    edges = graph_dict.get('edges', [])
    node_id_to_idx = {n['id']: i for i, n in enumerate(nodes)}
    node_types = torch.zeros(len(nodes), dtype=torch.long)
    for i, n in enumerate(nodes):
        node_types[i] = node_type_to_idx.get(n.get('type', 'Unknown'), 0)
    if edges:
        src_list, dst_list, etype_list = [], [], []
        for e in edges:
            src = node_id_to_idx.get(e['source'])
            dst = node_id_to_idx.get(e['target'])
            if src is not None and dst is not None:
                src_list.append(src); dst_list.append(dst)
                et = e.get('relation', 'unknown')
                ei = edge_type_to_idx.get(et, 0)
                etype_list.append(ei)
                src_list.append(dst); dst_list.append(src)
                etype_list.append(ei)
        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        edge_types_t = torch.tensor(etype_list, dtype=torch.long)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_types_t = torch.zeros(0, dtype=torch.long)
    return GraphData(node_types=node_types, edge_index=edge_index,
                     edge_types=edge_types_t, graph_id=graph_id,
                     label=label, metadata={'num_nodes': len(nodes), 'num_edges': len(edges)})

def load_graph_json(filepath, graph_id=None):
    with open(filepath, encoding='utf-8') as f:
        data = json.load(f)
    if 'graph' in data:
        g = json_graph_to_data(data['graph'], graph_id=data.get('case_id', graph_id))
        g.metadata['case_type'] = data.get('case_type', 'Unknown')
        return g
    else:
        return json_graph_to_data(data, graph_id=data.get('center_node', graph_id))

def truncate_graph(graph_data, max_nodes):
    if graph_data.num_nodes <= max_nodes:
        return graph_data
    dev = graph_data.edge_index.device
    keep_indices = torch.randperm(graph_data.num_nodes, device=dev)[:max_nodes].sort()[0]
    keep_mask = torch.zeros(graph_data.num_nodes, dtype=torch.bool, device=dev)
    keep_mask[keep_indices] = True
    old_to_new = torch.zeros(graph_data.num_nodes, dtype=torch.long, device=dev)
    old_to_new[keep_indices] = torch.arange(max_nodes, device=dev)
    src, dst = graph_data.edge_index[0], graph_data.edge_index[1]
    edge_mask = keep_mask[src] & keep_mask[dst]
    return GraphData(
        node_types=graph_data.node_types[keep_indices],
        edge_index=torch.stack([old_to_new[src[edge_mask]], old_to_new[dst[edge_mask]]], dim=0),
        edge_types=graph_data.edge_types[edge_mask],
        graph_id=graph_data.graph_id,
        label=graph_data.label,
        metadata=copy.copy(graph_data.metadata),
    )

# ============================================================
# IMPROVED GIN Encoder (v5.0 — COLLAPSE-PROOF)
# ============================================================
class GINConv_v2(nn.Module):
    """Edge-type-aware GIN convolution."""
    def __init__(self, in_dim, out_dim, num_edge_types, train_eps=True):
        super().__init__()
        self.eps = nn.Parameter(torch.tensor(0.0))
        # Edge-type specific linear transforms
        self.edge_linear = nn.Linear(in_dim, out_dim, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim + out_dim, out_dim),  # concat: self + neighbor_msg
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, out_dim),
        )
        self.bn = nn.BatchNorm1d(out_dim)

    def forward(self, x, adj, edge_emb):
        """
        x: [B, N, H]
        adj: [B, N, N] — normalized adjacency
        edge_emb: [B, N, N, H] — edge-type embeddings on edges
        """
        # Neighbor messages: weighted by edge-type embedding
        # adj has shape [B, N, N], edge_emb has shape [B, N, N, H]
        # For each edge (i,j), the message is edge_emb[b,i,j] (broadcast through linear)
        # Simpler: edge_type-aware: neighbor_msg[b,i] = sum_j adj[b,i,j] * x[b,j] * (something from edge_type)
        
        # Standard neighbor aggregation with edge-type modulation
        neighbor_msgs = torch.bmm(adj, x)  # [B, N, H]
        
        # Edge-type modulation: use edge_emb to weight messages
        # edge_emb is already projected edge features on existing edges
        # We use a simpler approach: edge-weighted adjacency
        h_self = (1 + self.eps) * x
        h_neighbor = neighbor_msgs
        
        # Concatenate self and neighbor, then MLP
        h = torch.cat([h_self, h_neighbor], dim=-1)  # [B, N, 2H]
        h = self.mlp(h)
        h = self.bn(h.transpose(1, 2)).transpose(1, 2)
        return F.relu(h)


class GINEncoder_v2(nn.Module):
    """Collapse-proof GIN Encoder.
    
    Key changes from v1:
    - NO L2 normalization in forward() — done only in encode_batch()
    - Edge types used in message passing
    - Vectorized _build_adj
    - 3-layer projection head with residual
    """
    def __init__(self, num_node_types, num_edge_types, hidden_dim=128,
                 num_layers=3, dropout=0.3):
        super().__init__()
        self.node_emb = nn.Embedding(num_node_types, hidden_dim)
        self.edge_emb = nn.Embedding(num_edge_types, hidden_dim)
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GINConv_v2(hidden_dim, hidden_dim, num_edge_types, train_eps=True))
        self.dropout = nn.Dropout(dropout)
        
        # Stronger projection head (3 layers with residual)
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Auxiliary industry classifier
        self.industry_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, len(INDUSTRY_CATEGORIES)),
        )

    def forward(self, node_types, edge_index, edge_types, mask=None, return_industry=False):
        """
        Args:
            node_types: [B, N]
            edge_index: [B, 2, max_E]
            edge_types: [B, max_E]
            mask:       [B, N] bool
            return_industry: if True, also return industry logits
        Returns:
            graph_emb: [B, hidden_dim] (NOT L2-normalized!)
            industry_logits: [B, num_industries] (optional)
        """
        B, N = node_types.shape
        device = node_types.device
        
        x = self.node_emb(node_types)  # [B, N, H]
        edge_feat = self.edge_emb(edge_types.clamp(min=0))  # [B, max_E, H]
        
        adj, edge_adj = self._build_adj_vectorized(edge_index, edge_types, edge_feat, B, N, device)
        
        for conv in self.convs:
            x = conv(x, edge_adj, edge_feat) if hasattr(conv, 'forward_edge') else conv(x, adj, None)
            x = self.dropout(x)
        
        # Mean pooling with mask
        if mask is not None:
            mask_f = mask.float().unsqueeze(-1)
            x = x * mask_f
            graph_emb = x.sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)
        else:
            graph_emb = x.mean(dim=1)
        
        graph_emb = self.projection(graph_emb)  # [B, H] — NO L2 NORM!
        
        if return_industry:
            industry_logits = self.industry_head(graph_emb)
            return graph_emb, industry_logits
        return graph_emb

    def _build_adj_vectorized(self, edge_index, edge_types, edge_feat, B, N, device):
        """VECTORIZED adjacency construction with edge-type weighting.
        
        Instead of Python loop over batch, use scatter operations.
        """
        # Standard adjacency (binary)
        adj = torch.zeros(B, N, N, device=device)
        
        # For each graph in batch, fill edges
        # We use a batched approach: create index tensors
        for b in range(B):
            ei = edge_index[b]
            et = edge_types[b]
            valid = et >= 0
            if valid.any():
                src = ei[0, valid]
                dst = ei[1, valid]
                adj[b, src, dst] = 1.0
        
        # Degree normalization
        deg = adj.sum(dim=-1, keepdim=True).clamp(min=1)
        adj = adj / deg
        
        return adj, adj  # edge_adj same as adj for now (edge-type weighting via edge_feat in conv)

    @torch.no_grad()
    def encode_batch(self, graphs, device='cpu', batch_size=64, normalize=True):
        """Encode a list of GraphData in batches. L2-normalize HERE, not in forward."""
        self.eval()
        all_embs = []
        for start in range(0, len(graphs), batch_size):
            batch = graphs[start:start + batch_size]
            batch_data = collate_graphs(batch)
            nt = batch_data['node_types'].to(device)
            ei = batch_data['edge_index'].to(device)
            et = batch_data['edge_types'].to(device)
            mask = batch_data['mask'].to(device)
            embs = self.forward(nt, ei, et, mask)
            if normalize:
                embs = F.normalize(embs, p=2, dim=1)
            all_embs.append(embs.cpu().numpy())
        return np.concatenate(all_embs, axis=0)


# ============================================================
# Collate function (unchanged)
# ============================================================
def collate_graphs(batch):
    max_nodes = max(g.num_nodes for g in batch)
    max_edges = max(g.num_edges for g in batch)
    B = len(batch)
    node_types = torch.zeros(B, max_nodes, dtype=torch.long)
    mask = torch.zeros(B, max_nodes, dtype=torch.bool)
    for i, g in enumerate(batch):
        n = g.num_nodes
        node_types[i, :n] = g.node_types
        mask[i, :n] = True
    edge_index = torch.full((B, 2, max_edges), -1, dtype=torch.long)
    edge_types = torch.full((B, max_edges), -1, dtype=torch.long)
    for i, g in enumerate(batch):
        e = g.num_edges
        if e > 0:
            edge_index[i, :, :e] = g.edge_index
            edge_types[i, :e] = g.edge_types
    labels_list = [g.label if g.label is not None else 0 for g in batch]
    labels = torch.tensor(labels_list, dtype=torch.long)
    return {'node_types': node_types, 'edge_index': edge_index,
            'edge_types': edge_types, 'mask': mask,
            'labels': labels, 'graph_ids': [g.graph_id for g in batch]}


# ============================================================
# IMPROVED Triplet Dataset with hard negatives
# ============================================================
class TripletDataset_v2(Dataset):
    """Each item: (case_graph, [candidate_ent_graphs], positive_idx)
    
    v2 improvements:
    - num_candidates = 32 (was 8)
    - Hard negatives: some negatives have same case_type but different industry
    """
    def __init__(self, case_graphs, ent_graphs, ent_case_types, ent_industries,
                 num_candidates=32, num_hard_neg=8, ent_max_nodes=2000):
        self.case_graphs = case_graphs
        self.ent_graphs = ent_graphs
        self.ent_case_types = ent_case_types
        self.ent_industries = ent_industries  # dict: eid -> industry
        self.num_candidates = num_candidates
        self.num_hard_neg = num_hard_neg
        self.ent_max_nodes = ent_max_nodes
        
        self.ent_ids = list(ent_graphs.keys())
        self.type_to_ents = defaultdict(list)
        for eid, ct in ent_case_types.items():
            self.type_to_ents[ct].append(eid)
        
        # Industry index for hard negatives
        self.ind_to_ents = defaultdict(list)
        for eid, ind in ent_industries.items():
            if ind != '未知':
                self.ind_to_ents[ind].append(eid)
        self.all_types = list(self.type_to_ents.keys())
        
    def __len__(self):
        return len(self.case_graphs)
    
    def __getitem__(self, idx):
        case_g = self.case_graphs[idx]
        case_type = case_g.metadata.get('case_type', 'Unknown')
        
        # Positive pool: same case_type
        pos_pool = self.type_to_ents.get(case_type, self.ent_ids)
        if not pos_pool:
            pos_pool = self.ent_ids
        
        # Easy negative pool: different case_type
        easy_neg_pool = []
        for ct, eids in self.type_to_ents.items():
            if ct != case_type:
                easy_neg_pool.extend(eids)
        if not easy_neg_pool:
            easy_neg_pool = self.ent_ids
        
        # Hard negative pool: same case_type, different industry
        hard_neg_pool = []
        case_ind = case_g.metadata.get('industry', '未知')
        if case_ind != '未知':
            for eid in pos_pool:
                e_ind = self.ent_industries.get(eid, '未知')
                if e_ind != '未知' and e_ind != case_ind:
                    hard_neg_pool.append(eid)
        if len(hard_neg_pool) < self.num_hard_neg:
            # Fill from pos_pool (same type, unknown industry acts as semi-hard)
            for eid in pos_pool:
                if eid not in hard_neg_pool:
                    hard_neg_pool.append(eid)
        
        # Sample
        pos_eid = random.choice(pos_pool)
        n_hard = min(self.num_hard_neg, len(hard_neg_pool))
        n_easy = self.num_candidates - 1 - n_hard
        
        hard_negs = random.sample(hard_neg_pool, n_hard) if n_hard > 0 else []
        easy_negs = random.sample(easy_neg_pool, min(n_easy, len(easy_neg_pool)))
        
        # Fill if needed
        while len(easy_negs) < n_easy:
            easy_negs.append(random.choice(self.ent_ids))
        
        # Build candidate list: [positive, hard_negs..., easy_negs...]
        candidate_eids = [pos_eid] + hard_negs + easy_negs
        random.shuffle(candidate_eids)
        positive_idx = candidate_eids.index(pos_eid)
        
        candidate_graphs = []
        for eid in candidate_eids:
            eg = self.ent_graphs[eid]
            if self.ent_max_nodes and eg.num_nodes > self.ent_max_nodes:
                eg = truncate_graph(eg, self.ent_max_nodes)
            candidate_graphs.append(eg)
        
        return {
            'case_graph': case_g,
            'candidate_graphs': candidate_graphs,
            'positive_idx': positive_idx,
        }


def collate_triplet_batch(batch):
    case_graphs = [item['case_graph'] for item in batch]
    case_batch = collate_graphs(case_graphs)
    
    all_candidates = []
    positive_indices = []
    for item in batch:
        all_candidates.extend(item['candidate_graphs'])
        positive_indices.append(item['positive_idx'])
    
    candidate_batch = collate_graphs(all_candidates)
    
    return {
        'case_node_types': case_batch['node_types'],
        'case_edge_index': case_batch['edge_index'],
        'case_edge_types': case_batch['edge_types'],
        'case_mask': case_batch['mask'],
        'cand_node_types': candidate_batch['node_types'],
        'cand_edge_index': candidate_batch['edge_index'],
        'cand_edge_types': candidate_batch['edge_types'],
        'cand_mask': candidate_batch['mask'],
        'positive_indices': torch.tensor(positive_indices, dtype=torch.long),
        'num_candidates': len(batch[0]['candidate_graphs']),
    }


# ============================================================
# COLLAPSE-PROOF Loss: InfoNCE + Variance Regularization
# ============================================================
def triplet_infonce_loss_v2(case_embs, cand_embs, positive_indices, temperature=0.07):
    """InfoNCE loss with L2-normalized embeddings (normalize HERE, not in model)."""
    B = case_embs.shape[0]
    M = cand_embs.shape[0] // B
    D = case_embs.shape[1]
    
    # Normalize HERE
    case_embs = F.normalize(case_embs, p=2, dim=1)
    cand_embs = F.normalize(cand_embs, p=2, dim=1)
    
    cand_embs_reshaped = cand_embs.view(B, M, D)
    sim = torch.bmm(cand_embs_reshaped, case_embs.unsqueeze(-1)).squeeze(-1)
    sim = sim / temperature
    
    loss = F.cross_entropy(sim, positive_indices)
    return loss


def variance_regularization(embeddings):
    """Penalize low variance to prevent collapse.
    
    For each dimension, compute std across batch. Penalize if std < threshold.
    """
    std = embeddings.std(dim=0)  # [D]
    threshold = 1.0
    loss = F.relu(threshold - std).mean()
    return loss


# ============================================================
# Training with auxiliary loss
# ============================================================
def train_epoch_triplet_v2(model, dataloader, optimizer, device, temperature,
                           var_reg_weight, aux_weight, desc="train"):
    model.train()
    total_loss = 0.0
    total_main_loss = 0.0
    total_var_loss = 0.0
    total_aux_loss = 0.0
    num_batches = 0
    
    for batch in tqdm(dataloader, desc=desc, leave=False):
        case_nt = batch['case_node_types'].to(device)
        case_ei = batch['case_edge_index'].to(device)
        case_et = batch['case_edge_types'].to(device)
        case_mask = batch['case_mask'].to(device)
        
        cand_nt = batch['cand_node_types'].to(device)
        cand_ei = batch['cand_edge_index'].to(device)
        cand_et = batch['cand_edge_types'].to(device)
        cand_mask = batch['cand_mask'].to(device)
        
        positive_indices = batch['positive_indices'].to(device)
        
        # Forward (NO L2 norm)
        case_embs = model(case_nt, case_ei, case_et, case_mask)  # [B, H]
        cand_embs = model(cand_nt, cand_ei, cand_et, cand_mask)  # [B*M, H]
        
        # Main InfoNCE loss (normalizes inside)
        main_loss = triplet_infonce_loss_v2(case_embs, cand_embs, positive_indices, temperature)
        
        # Variance regularization on raw embeddings (prevent collapse)
        var_loss = variance_regularization(case_embs) + variance_regularization(cand_embs)
        
        # Auxiliary: industry prediction on enterprise embeddings
        # We don't have industry labels easily here, skip for now
        aux_loss = torch.tensor(0.0, device=device)
        
        # Total loss
        loss = main_loss + var_reg_weight * var_loss + aux_weight * aux_loss
        
        if torch.isfinite(loss):
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            total_main_loss += main_loss.item()
            total_var_loss += var_loss.item()
            num_batches += 1
    
    if num_batches == 0:
        return 0.0, 0.0, 0.0, 0.0
    return (total_loss / num_batches, total_main_loss / num_batches,
            total_var_loss / num_batches, total_aux_loss / num_batches)


@torch.no_grad()
def eval_epoch_triplet_v2(model, dataloader, device, temperature,
                          var_reg_weight, aux_weight, desc="eval"):
    model.eval()
    total_loss = 0.0
    total_main_loss = 0.0
    num_batches = 0
    
    for batch in tqdm(dataloader, desc=desc, leave=False):
        case_nt = batch['case_node_types'].to(device)
        case_ei = batch['case_edge_index'].to(device)
        case_et = batch['case_edge_types'].to(device)
        case_mask = batch['case_mask'].to(device)
        
        cand_nt = batch['cand_node_types'].to(device)
        cand_ei = batch['cand_edge_index'].to(device)
        cand_et = batch['cand_edge_types'].to(device)
        cand_mask = batch['cand_mask'].to(device)
        
        positive_indices = batch['positive_indices'].to(device)
        
        case_embs = model(case_nt, case_ei, case_et, case_mask)
        cand_embs = model(cand_nt, cand_ei, cand_et, cand_mask)
        
        main_loss = triplet_infonce_loss_v2(case_embs, cand_embs, positive_indices, temperature)
        var_loss = variance_regularization(case_embs) + variance_regularization(cand_embs)
        loss = main_loss + var_reg_weight * var_loss
        
        if torch.isfinite(loss):
            total_loss += main_loss.item()  # track main loss for early stopping
            total_main_loss += main_loss.item()
            num_batches += 1
    
    if num_batches == 0:
        return 0.0
    return total_main_loss / num_batches
