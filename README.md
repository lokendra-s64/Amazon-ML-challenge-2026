# CodeNova — Amazon ML Challenge 2026: Business Entity Resolution

**Team:** CodeNova  
**Challenge:** Amazon ML Challenge 2026 — Business Entity Resolution  
**Metric:** Macro-averaged F₀.₅

## Team Members

| # | Name | Role |
|---|------|------|
| 1 | Lokendra Sonwani | B.Tech Student|
| 2 | Akash Prajapati | B.Tech Student |
| 3 | Nitin Singh | B.Tech Student |

## Overview

This repository contains our complete solution for the Business Entity Resolution challenge. Given business records from 3 independent sources (S1, S2, S3) with noisy/inconsistent fields, the task is to find all matching records from S2 and S3 for each S1 entity.

## Solution Architecture

### 1. Normalization (`src/normalize.py`)
- Unicode NFKD accent folding
- Legal suffix removal (Ltd, Pvt, LLC, Inc, Corp, etc.)
- Address abbreviation expansion (Road→rd, Street→st, etc.)
- Landmark stripping ("near X" removal)
- Postal code extraction (5-6 digit PIN/ZIP)

### 2. Blocking / Candidate Generation (`src/blocking.py`, `src/pipeline.py`)
**Hybrid multi-signal approach:**
- Token-based inverted indices (name + address tokens)
- Exact postal code matching
- First-3-char prefix index (catches typos)
- **No hard country filtering** — handles unseen countries (France in test)

### 3. Pairwise Features (`src/features.py`) — 15 features
| Category | Features |
|----------|----------|
| Name | exact match, Levenshtein, Jaccard, token-sort, char-3-gram, first-token |
| Address | Levenshtein, Jaccard, char-3-gram |
| Postal | exact match, both present |
| Country | same label |
| Structural | name/address length diff |

### 4. Model (`src/model.py`)
**HistGradientBoostingClassifier** (scikit-learn)
- BSD-licensed, classical GBM (≪8B params)
- `max_depth=6`, `max_iter=300`, `learning_rate=0.06`
- `class_weight="balanced"` for rare matches (~1-5%)

### 5. Threshold Tuning
- GroupShuffleSplit on S1 entities (no leakage)
- Sweep probability threshold 0.05–0.95
- **Directly optimize macro F₀.₅** (competition metric)
- Retrain on full data at best threshold

## Performance

| Metric | Result |
|--------|--------|
| Validation macro F₀.₅ | >0.70 |
| Blocking recall ceiling | ≥0.90 |
| Test entities processed | 1,732,544 |
| Matched entities | 1,624,987 |

## Submission

```
CodeNova_submission.zip
├── output/
│   ├── matching_results.tsv    (1.7M rows, validated ✅)
│   └── candidate_pairs.tsv     (1.7M rows, validated ✅)
├── code/
│   └── business_entity_resolution/
│       ├── src/                (14 pipeline modules)
│       ├── models/             (model.pkl + threshold)
│       ├── README.md
│       └── requirements.txt
└── Documentation_template.md
```

## Quick Start

```bash
# Install deps
pip install -r code/business_entity_resolution/requirements.txt

# Train (sampled for speed)
cd code/business_entity_resolution/src
python train.py --train-dir ../../../dataset/train --model-out ../models/model.pkl --sample-s1 50000

# Predict (chunked for memory)
python predict_chunked.py --test-dir ../../../dataset/test --model ../models/model.pkl --output-dir ../../../output

# Validate
cd ../../..
python validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test
```

## Compliance

- ✅ No external data/APIs/geocoding
- ✅ Model: HistGradientBoostingClassifier (BSD, classical ML)
- ✅ Country as open set (France works)
- ✅ Output format exact (tab-separated, correct headers)
- ✅ All S1 entities have exactly one row
- ✅ No S1 self-matches, no duplicates, valid prefixes
- ✅ Matched IDs ⊆ Candidate IDs
- ✅ `validate_submission.py` → **PASS**

## License

MIT License — compliant with challenge requirements.

---

**Team CodeNova** | Amazon ML Challenge 2026
