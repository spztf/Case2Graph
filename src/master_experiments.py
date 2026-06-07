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

Output: ./final_results/
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
# GLOBAL CONFIG — paths via environment variables
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

# ... (remaining model + experiment code unchanged - 1177 lines total)
