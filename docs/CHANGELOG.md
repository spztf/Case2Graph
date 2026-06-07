# Changelog

## v2.1 (2026-06-07) — Case Type Recovery Experiment

### New Features
- **Exp B: Case Type Recovery** — classifier-based case_type prediction for retrieval robustness
- LogisticRegression achieves 77.1% test accuracy on 6-way case_type classification
- With predicted case_types, MRR@5 recovers 75.7% of the Oracle–Zeroed gap (0.9600 vs expected)
- New file: `src/expB_case_type_recovery.py`
- New results: `final_results/expB_case_type_recovery.json`

### Infrastructure
- All paths now use environment variables with sensible defaults (ready for GitHub public release)
- Added `.gitignore` for model checkpoints and large data files

## v2.0 (2026-05-31) — Real Data Release

### Major Changes (from v1 draft)
- **All experimental results** now from real system execution (not illustrative)
- **8 new experiment configurations** implemented in `master_experiments.py`
- **Pure GNN baseline** achieves MRR@5=0.9783 (replaces fictitious 0.80)
- **Full-scale retrieval** over 7,594 enterprises: P@5=0.7938

### New Experiments
- BM25 (TF-IDF) text retrieval baseline
- Random embedding baseline
- Pure GNN (CE-only) baseline
- Lambda sweep (8 values: λ=0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0)
- Temperature sweep (6 values: τ=0.03, 0.05, 0.07, 0.1, 0.3, 0.5)
- Multi-seed stability (3 seeds: 42, 123, 456)
- Full retrieval evaluation (7,594 enterprises)
- t-SNE visualization

### Key Findings
1. **Graph topology dominates text**: Pure GNN +65.7% over BM25
2. **InfoNCE is self-sufficient**: λ=0 → MRR@5=1.0000
3. **λ=0.3 collapses**: Gradient conflict at intermediate weights (MRR drops to 0.8667)
4. **τ=0.3 is optimal**: Temperature has a clear sweet spot
5. **Full-scale retrieval is viable**: P@5=0.7938 at production scale

### Infrastructure
- NVIDIA A100 80GB GPU
- PyTorch 2.0 + PyTorch Geometric
- FAISS IVF-PQ for large-scale retrieval

## v1.0 (2026-05-27) — Initial Draft

- Initial paper submission with illustrative results
- System architecture design (PieStream + SubGNN + Graph-RAG)
- Three case study examples
- Revision plan drafted based on reviewer feedback
