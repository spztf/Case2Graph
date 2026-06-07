# From Case to Graph: Graph Contrastive Learning for Case-Driven Tax Evasion Pattern Retrieval

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

Official implementation of the paper submitted to **IEEE ICSC 2026**.

> **Abstract:** Given a confirmed tax evasion case, can we automatically find other enterprises employing the *same modus operandi*? We cast case-driven pattern retrieval as a graph contrastive learning problem. Each case is represented as a heterogeneous subgraph of the enterprise knowledge graph; a GIN-based encoder with edge-type-aware message passing maps these subgraphs into a shared embedding space where cases with identical topology-based evasion patterns cluster together. Trained with a hybrid InfoNCE + cross-entropy objective on 7,588 real-world Chinese tax cases, Pure GNN (CE-only) achieves MRR@5=0.9783, P@5=0.9722, Hit@10=0.9901 on a held-out test set—a **+65.7% relative gain** over BM25 text retrieval (MRR@5=0.5905). Our joint contrastive model further attains MRR@5=0.9504 (±0.0577 over 3 seeds) and scales to full-database retrieval over 7,594 enterprises at P@5=0.7938.

---

## 📁 Repository Structure

```
.
├── README.md                          # This file
├── LICENSE                            # MIT License
├── requirements.txt                   # Python dependencies
├── CITATION.cff                       # Citation metadata
├── CONTRIBUTING.md                    # Contribution guidelines
├── .gitignore
├── src/
│   ├── master_experiments.py          # 🎯 Main orchestrator: all experiments
│   ├── expB_case_type_recovery.py     # 🔬 Case type recovery (Exp B)
│   ├── task_c_common_v2.py            # Shared data loading / model definitions
│   ├── phase3_train_v12_joint.py      # Joint training (CE + InfoNCE)
│   ├── phase3_train_v13_balanced.py   # Balanced sampling variant
│   ├── phase3_train_v14_pure_nce.py   # Pure InfoNCE variant
│   ├── build_kg_v3.py                 # Knowledge graph construction pipeline
│   ├── retrieval_eval.py              # Standalone retrieval evaluation
│   └── generate_dataset.py            # Dataset generation from raw data
├── data/
│   ├── case_graphs.tar.gz             # 7,588 case subgraphs (compressed)
│   ├── enterprise_graph.json          # Full enterprise knowledge graph
│   ├── seed_cases.json                # Annotated seed cases
│   ├── seed_cases_full.json           # Full seed case metadata
│   ├── node_vocabs.json               # Node attribute vocabularies
│   ├── type_analysis.txt              # Case type distribution
│   └── data_kg_readme.md              # KG building statistics
├── final_results/
│   ├── bm25_baseline.json             # BM25 text retrieval baseline
│   ├── random_baseline.json           # Random embedding baseline
│   ├── pure_gnn_baseline.json         # CE-only GNN baseline
│   ├── lambda_sweep.json              # λ (CE weight) sweep
│   ├── temperature_sweep.json         # τ (temperature) sweep
│   ├── seed_stability.json            # Multi-seed variance
│   ├── full_retrieval.json            # Full 7,594-enterprise retrieval
│   ├── split_meta_v9.json             # Train/val/test split metadata
│   ├── embeddings_v13_balanced.npz    # 128-dim case embeddings
│   └── expB_case_type_recovery.json   # Case type recovery results (generated at runtime)
├── final_models/
│   ├── master_lambda_0.0_best.pt      # λ=0 (InfoNCE-only)
│   ├── master_tau_0.3_best.pt         # τ=0.3 optimal
│   ├── master_pure_gnn_best.pt        # CE-only GNN
│   └── master_full_eval_best.pt       # Full-retrieval model
├── figures/
│   ├── system_architecture.svg        # System architecture diagram
│   ├── case1_sequence.svg             # Case 1 event sequence
│   ├── case2_sequence.svg             # Case 2 event sequence
│   ├── case3_sequence.svg             # Case 3 event sequence
│   └── embedding_tsne_combined.png    # t-SNE visualization
└── docs/
    ├── DATA_DESCRIPTION.md            # Detailed data schema & provenance
    └── CHANGELOG.md                   # Version history
```

---

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Create conda environment
conda create -n tax-retrieval python=3.10 -y
conda activate tax-retrieval

# Install dependencies
pip install -r requirements.txt

# Extract case graphs
tar xzf data/case_graphs.tar.gz -C data/
```

### 2. Reproduce Main Results

```bash
# Run all baselines (BM25, Random, Pure GNN) — ~10 min on A100
python src/master_experiments.py --exp baseline --gpu 0

# Run lambda sweep (8 values × ~3 min each) — ~25 min
python src/master_experiments.py --exp lambda --gpu 0

# Run temperature sweep (6 values) — ~18 min
python src/master_experiments.py --exp tau --gpu 0

# Run multi-seed stability — ~9 min
python src/master_experiments.py --exp seeds --gpu 0

# Print results summary
python src/master_experiments.py --exp summary
```

All results are saved to `final_results/` (JSON) and `final_models/` (PyTorch checkpoints).

### 3. Run a Single Experiment

```bash
# BM25 baseline only
python src/master_experiments.py --exp bm25 --gpu 0

# Full retrieval over all 7,594 enterprises
python src/master_experiments.py --exp full_eval --gpu 0
```

### 4. Custom Data Paths

All paths can be overridden via environment variables:

```bash
export CASE_GRAPHS_DIR=/path/to/case_graphs
export ENTERPRISE_SUBGRAPHS_DIR=/path/to/enterprise_subgraphs
export SPLIT_META_PATH=/path/to/split_meta_v9.json
export VOCAB_PATH=/path/to/node_vocabs.json
export MODEL_DIR=/path/to/models
export MASTER_RESULTS_DIR=/path/to/results

python src/master_experiments.py --exp baseline
```

---

## 🔬 Experiment B: Case Type Recovery

**Question:** If we *don't know* the case type for retrieval, can we predict it from text features and still achieve good ranking performance?

We train a LogisticRegression classifier on scheme description text (TF-IDF char bigrams) + numeric features (amount, industry) to predict case_type. Then we use the *predicted* case_type in the GNN ranking model and compare against Oracle (ground-truth) and Zeroed (no case_type) settings.

### Classification Performance
| Model | Val Acc | Test Acc | Test F1 (macro) |
|-------|---------|----------|-----------------|
| LogisticRegression | 0.7869 | **0.7711** | **0.7528** |
| LinearSVC | 0.7723 | — | — |
| RandomForest | 0.7545 | — | — |
| GradientBoosting | 0.7697 | — | — |

### Ranking Recovery

| Metric | Oracle | Predicted | Zeroed | Recovery % |
|--------|--------|-----------|--------|------------|
| **MRR@5** | 0.9783 | 0.9600 | 0.9030 | **+75.7%** |
| **Precision@5** | 0.9722 | 0.9268 | 0.8207 | **+70.0%** |
| **Hit@5** | 0.9888 | 0.9888 | 0.9367 | **+100.0%** |
| **MRR@10** | 0.9785 | 0.9608 | 0.9057 | **+75.7%** |
| **Precision@10** | 0.9462 | 0.9067 | 0.8148 | **+69.9%** |
| **Hit@10** | 0.9901 | 0.9941 | 0.9571 | **+112.0%** |

> **Key finding:** Even with only 77% classification accuracy, predicted case types recover **70–100%** of the performance gap between Oracle and Zeroed settings, demonstrating strong robustness.

```bash
# Run the case type recovery experiment
python src/expB_case_type_recovery.py
```

---

## 📊 Key Results

| Experiment | MRR@5 | MRR@10 | P@5 | P@10 | Hit@10 |
|-----------|-------|--------|-----|------|--------|
| **BM25 (TF-IDF)** | 0.5905 | 0.6001 | 0.4768 | 0.4640 | 0.7704 |
| Random Embedding | 0.4106 | 0.4235 | 0.3365 | 0.3346 | 0.6075 |
| **Pure GNN (CE only)** | **0.9783** | 0.9785 | **0.9722** | 0.9462 | **0.9901** |
| Joint ×3 (mean±σ) | 0.9504±0.058 | 0.9504±0.058 | 0.9166±0.057 | 0.9178±0.054 | 0.9835±0.016 |

### Lambda Sweep (CE weight λ)

| λ | MRR@5 | P@5 |
|---|-------|-----|
| 0.0 ★ | **1.0000** | 0.9496 |
| 0.1 | 0.9683 | 0.9447 |
| 0.2 | 0.9703 | 0.9187 |
| 0.3 | 0.8667 | 0.8732 | ← collapse
| 1.0 | 0.9291 | 0.9065 |

### Temperature Sweep (τ)

| τ | MRR@5 | P@5 |
|---|-------|-----|
| 0.03 | 0.9324 | 0.9078 |
| 0.07 | 0.9764 | 0.9615 |
| 0.30 ★ | **0.9944** | 0.9761 |
| 0.50 | 0.9617 | 0.9294 |

### Full-Scale Retrieval (7,594 enterprises)

| Metric | Value |
|--------|-------|
| P@5 | 0.7938 |
| P@10 | 0.7114 |
| MRR@5 | 0.8430 |
| Hit@20 | 0.9888 |
| Retrieval latency | ~200ms (FAISS) |

---

## 🏗️ Model Architecture

```
Input: Case Subgraph (heterogeneous, 7 node types × 7 edge types)
  │
  ▼
Node Feature Encoder ─── GINConv (3 layers, edge-type-aware)
  │  • Company: industry embedding + location embedding
  │  • Person: role embedding
  │  • FinancialFlow: amount bucket embedding
  │  • Authority/ViolationEvent/Location/Industry: type embedding
  │
  ▼
Global Mean Pooling ────→ d=128 graph-level embedding
  │
  ▼
Projection Head ──────── MLP(128→64→128)
  │
  ▼
Dual Objective:
  ├── L_CE: Cross-entropy over 9 sub-types
  └── L_NCE: InfoNCE with in-batch negatives
       L = λ·L_CE + (1-λ)·L_NCE
```

---

## 📦 Data

The dataset consists of **7,588 anonymized Chinese tax evasion cases** from provincial tax bureau records (2018–2024), represented as heterogeneous knowledge graph subgraphs.

### Graph Schema

| Node Type | Count | Description |
|-----------|-------|-------------|
| Company | 7,594 | Tax-registered enterprise |
| Person | 46 | Legal representative / controller |
| Authority | 707 | Tax bureau branch |
| ViolationEvent | 7,588 | Recorded violation instance |
| FinancialFlow | 12,927 | Invoice / payment record |
| Location | 40 | Province/City |
| Industry | 25 | Industry category |

| Edge Type | Count | Semantics |
|-----------|-------|-----------|
| `located_in` | 6,061 | Company → Location |
| `in_industry` | 6,043 | Company → Industry |
| `penalized_by` | 7,594 | ViolationEvent → Authority |
| `committed` | 7,594 | Company → ViolationEvent |
| `controlled_by` | 41 | Company → Person |
| `involves_amount` | 12,927 | ViolationEvent → FinancialFlow |
| `co_penalized` | 12 | Company ↔ Company |

### Case Type Distribution

| Type | Count |
|------|-------|
| 虚开增值税专用发票 (False VAT invoices) | 2,830 |
| 虚开发票 (False invoices) | 1,449 |
| 隐匿收入 (Concealed income) | 1,160 |
| 骗取出口退税 (Export tax rebate fraud) | 1,157 |
| 其他 (Other) | 968 |
| 转让定价 (Transfer pricing) | 24 |

See [`docs/DATA_DESCRIPTION.md`](docs/DATA_DESCRIPTION.md) for full schema documentation.

---

## 🔬 Reproducibility

All experiments use fixed random seeds. Key hyperparameters:

| Parameter | Value |
|-----------|-------|
| Embedding dimension | 128 |
| GNN layers | 3 (GINConv) |
| Optimizer | Adam (lr=1e-3, wd=1e-4) |
| Batch size | 128 |
| Training epochs | 350 (early stop patience=80) |
| Temperature τ | 0.3 (optimal) |
| CE weight λ | 0.0 (InfoNCE-only, optimal) |
| GPU | NVIDIA A100 80GB |

---

## 📝 Citation

If you use this code or data in your research, please cite:

```bibtex
@inproceedings{li2026casegraph,
  title     = {From Case to Graph: Graph Contrastive Learning for
               Case-Driven Tax Evasion Pattern Retrieval},
  author    = {Li, {First Author} and {Second Author} and {Third Author}},
  booktitle = {Proceedings of the IEEE International Conference on
               Social Computing (ICSC)},
  year      = {2026},
  note      = {Under review}
}
```

See [`CITATION.cff`](CITATION.cff) for machine-readable metadata.

---

## 📄 License

This project is licensed under the MIT License — see [`LICENSE`](LICENSE) for details.

The underlying tax case data is anonymized and used under a data-sharing agreement with the relevant provincial tax authority. **The data is provided for academic research purposes only.** Commercial use, re-identification attempts, or redistribution outside research contexts is prohibited.

---

## 🤝 Contact

For questions about the paper, code, or data, please open a GitHub issue or contact the corresponding author.
