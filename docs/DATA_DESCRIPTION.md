# Data Description

## Overview

The dataset consists of **7,588 anonymized Chinese tax evasion cases** (2018–2024) from a provincial tax bureau, represented as a heterogeneous knowledge graph. All personally identifiable information (PII) has been removed: enterprise names are replaced with hashed IDs, person names with role labels, and exact monetary amounts with logarithmic buckets.

---

## File Inventory

### `data/enterprise_graph.json` (14 MB)
The full enterprise knowledge graph in a single JSON file.

```json
{
  "nodes": [
    {
      "id": "e_00001",
      "type": "Company",
      "attrs": {
        "industry": "制造业",
        "province": "广东",
        "city": "深圳",
        "reg_capital_bucket": 3
      }
    },
    ...
  ],
  "edges": [
    {
      "src": "e_00001",
      "dst": "loc_003",
      "type": "located_in"
    },
    ...
  ]
}
```

**Statistics:**
- Total nodes: 28,927
- Total edges: 40,272
- Node types: 7 (Company, Person, Authority, ViolationEvent, FinancialFlow, Location, Industry)
- Edge types: 7 (located_in, in_industry, penalized_by, committed, controlled_by, involves_amount, co_penalized)

### `data/case_graphs.tar.gz` (2.8 MB compressed, ~53 MB extracted)
7,588 individual case subgraph files, each stored as `case_graphs/case_XXXX.json`.

Each file contains the k-hop ego-subgraph (k=3) around a ViolationEvent node:

```json
{
  "case_id": "case_0001",
  "case_type": "虚开增值税专用发票",
  "nodes": { ... },
  "edges": [ ... ],
  "node_features": { ... }
}
```

### `data/seed_cases.json` (5.4 KB)
Metadata for 50 manually annotated seed cases used as the labeled training set.

### `data/seed_cases_full.json` (274 KB)
Extended metadata including full case descriptions, legal references, and penalty amounts.

### `data/node_vocabs.json` (6.8 KB)
Vocabulary mappings for categorical node attributes:
- `industry`: 25 unique industry categories
- `province`: 31 province-level divisions
- `city`: ~300 city-level divisions
- `amount_bucket`: 8 log-scale buckets

### `data/type_analysis.txt` (2 KB)
Human-readable distribution of case types and causal chain patterns.

---

## Node Type Schema

### Company (7,594 nodes)
| Attribute | Type | Description |
|-----------|------|-------------|
| `industry` | categorical (25) | Industry classification |
| `province` | categorical (31) | Province of registration |
| `city` | categorical (~300) | City of registration |
| `reg_capital_bucket` | int (0-7) | Registered capital (log bucket) |

### Person (46 nodes)
| Attribute | Type | Description |
|-----------|------|-------------|
| `role` | categorical | Legal representative / controller |

### Authority (707 nodes)
| Attribute | Type | Description |
|-----------|------|-------------|
| `level` | categorical | Provincial / Municipal / District tax bureau |

### ViolationEvent (7,588 nodes)
| Attribute | Type | Description |
|-----------|------|-------------|
| `case_type` | categorical (6) | Type of tax violation |
| `year` | int | Year of adjudication |
| `penalty_amount_bucket` | int (0-7) | Penalty amount (log bucket) |

### FinancialFlow (12,927 nodes)
| Attribute | Type | Description |
|-----------|------|-------------|
| `amount_bucket` | int (0-7) | Transaction amount (log bucket) |
| `flow_type` | categorical | Invoice / Payment / Transfer |

### Location (40 nodes)
| Attribute | Type | Description |
|-----------|------|-------------|
| `province` | categorical (31) | Province |
| `region` | categorical | East / Central / West / Northeast |

### Industry (25 nodes)
| Attribute | Type | Description |
|-----------|------|-------------|
| `category` | categorical | Industry name |
| `sector` | categorical | Primary / Secondary / Tertiary |

---

## Edge Type Schema

| Edge Type | Source → Target | Semantics |
|-----------|----------------|-----------|
| `located_in` | Company → Location | Enterprise registration location |
| `in_industry` | Company → Industry | Enterprise industry classification |
| `penalized_by` | ViolationEvent → Authority | Which tax bureau issued the penalty |
| `committed` | Company → ViolationEvent | Which enterprise committed the violation |
| `controlled_by` | Company → Person | Legal representative / controller relationship |
| `involves_amount` | ViolationEvent → FinancialFlow | Monetary amount involved in violation |
| `co_penalized` | Company ↔ Company | Co-defendants in the same case |

---

## Case Type Distribution

| Case Type (Chinese) | English | Count | % |
|---------------------|---------|-------|---|
| 虚开增值税专用发票 | False VAT invoices | 2,830 | 37.3% |
| 虚开发票 | False invoices | 1,449 | 19.1% |
| 隐匿收入 | Concealed income | 1,160 | 15.3% |
| 骗取出口退税 | Export tax rebate fraud | 1,157 | 15.2% |
| 其他 | Other | 968 | 12.8% |
| 转让定价 | Transfer pricing | 24 | 0.3% |
| **Total** | | **7,588** | **100%** |

---

## Train / Validation / Test Split

Cases are split by stratified sampling on case type:

| Split | Count | % |
|-------|-------|---|
| Train | 5,311 | 70% |
| Validation | 1,138 | 15% |
| Test | 1,139 | 15% |

Split metadata is stored in `final_results/split_meta_v9.json`.

---

## Data Provenance

1. **Raw source:** Administrative penalty decisions (行政处罚决定书) from a provincial tax bureau, 2018–2024.
2. **Extraction:** Entity and relation extraction using a fine-tuned BERT-based IE pipeline (PieStream).
3. **Graph construction:** Extracted entities and relations are merged into a heterogeneous graph with entity resolution across cases.
4. **Anonymization:** All enterprise names, person names, and identifiers replaced with hash IDs. Monetary amounts bucketed into 8 log-scale bins.
5. **Quality control:** 50 seed cases manually verified by three tax auditors (inter-annotator agreement κ=0.87).

---

## Usage Notes

- **Academic use only.** The data is provided under a data-sharing agreement with the tax authority. Commercial use, re-identification attempts, or redistribution outside academic research is prohibited.
- **Extract case graphs:** `tar xzf data/case_graphs.tar.gz -C data/`
- **Enterprise subgraphs** (24 GB) are not included due to size. They can be regenerated from `enterprise_graph.json` using `src/build_kg_v3.py`.
- **GPU requirement:** Training requires ~16 GB GPU memory (NVIDIA A100 or V100 recommended).

---

## Citation

If you use this data, please cite the accompanying paper and acknowledge the data-sharing agreement:

```
The tax case data used in this research was provided by [Provincial Tax Bureau] 
under data-sharing agreement [ID]. All data has been anonymized and is used 
for academic research purposes only.
```
