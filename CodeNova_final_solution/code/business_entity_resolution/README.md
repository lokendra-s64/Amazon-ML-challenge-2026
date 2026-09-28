# Amazon ML Challenge 2026: Business Entity Resolution — Complete Solution

## Problem Summary

Given business records from 3 independent sources (S1, S2, S3) with noisy/inconsistent fields, find all matching records from S2 and S3 for each S1 entity. Evaluation metric: macro-averaged F₀.₅ (precision-weighted).

**Key constraints:**
- No external data/APIs/geocoding (fair-play rules)
- Test set adds unseen country "France" — cannot hard-code countries
- Output format: `matching_results.tsv` + `candidate_pairs.tsv` (tab-separated)
- Model must be MIT/Apache 2.0 licensed, ≤8B params

---

## Solution Architecture

### 1. Normalization (`normalize.py`)
Removes systematic noise so downstream similarities compare like-with-like:

**Business Name:**
- Unicode NFKD accent folding (café → cafe)
- Lowercase, punctuation removal
- Legal suffix canonicalization & removal: Corp/Corporation/Inc/Incorporated/Ltd/Limited/Pvt/Private/LLC/Co/Company/GmbH/SA/SAS/SRL/BV/PLC/Group/Holdings/Services/Solutions/Technologies/Tech/Intl/Global/National/Assoc/Bros/Ent/Ind → removed entirely
- Stopword removal: "the", "and", "of", "for", "in", "on", "at", "to", "a", "an", "&", "co"
- Generic location words removed: "india", "usa", "uk", etc.

**Address:**
- Same base cleaning
- Abbreviation expansion: Rd/Road, St/Street, Ave/Avenue, Blvd/Boulevard, Ln/Lane, Dr/Drive, Ct/Court, Cir/Circle, Hwy/Highway, Apt/Apartment, Fl/Floor, Ste/Suite, Sec/Sector, Blk/Block, Col/Colony, Ngr/Nagar, X/Cross
- Directional normalization: North/N, South/S, East/E, West/W
- Landmark stripping: "near X", "opp X", "behind X" removed from core address
- Postal code extraction (5-6 digit PIN/ZIP)
- Address stopwords: "no", "rd", "st", "dr", "ave", "new", "fl", "city", "near", "nan", "null", "unit", "plot", single digits/letters

### 2. Blocking / Candidate Generation (`pipeline.py` - `generate_candidates`)

**Hybrid approach for high recall + speed:**

| Signal | Method | Weight | Purpose |
|--------|--------|--------|---------|
| TF-IDF cosine (word+bigram) | Vectorized sparse matrix | Scaled score | Dense semantic similarity, catches fuzzy matches |
| Shared name tokens | Inverted index | 3 | Exact token overlap on normalized names |
| Shared address tokens | Inverted index | 1 | Exact token overlap on normalized addresses |
| Exact postal code | Hash index | 10 | Very strong match signal |
| First-3-char prefix | Hash index | 1 | Catches typo'd first token |

**Parameters:**
- `TOP_K_TFIDF = 50` per S1 entity (from combined S2+S3 pool)
- `TOP_K_TOKEN = 50` per source (S2/S3 separately)
- TF-IDF: `min_df=2`, `max_df=0.90`, sublinear TF, (1,2)-grams
- Batched cosine similarity (batch_size=5000) for memory efficiency
- Union of all signals, deduplicated, ranked by composite score

**Why not hard-filter by country?** Test set includes "France" (unseen in train). Country is used only as a feature (`same_country`) for the model.

### 3. Pairwise Features (`pipeline.py` - `build_features`)

16 features per (S1, candidate) pair:

| Feature | Description |
|---------|-------------|
| `block_score` | Composite blocking score |
| `name_exact_norm` | Exact normalized name match (1/0) |
| `name_lev_ratio` | Levenshtein ratio on normalized names |
| `name_token_jaccard` | Jaccard on name token sets |
| `name_token_sort_ratio` | Levenshtein ratio on sorted tokens (word-order invariant) |
| `name_char3gram_jaccard` | Jaccard on character 3-grams (typo-robust) |
| `name_first_token_match` | First token exact match |
| `addr_lev_ratio` | Levenshtein ratio on normalized addresses |
| `addr_token_jaccard` | Jaccard on address token sets |
| `addr_char3gram_jaccard` | Jaccard on address char 3-grams |
| `postal_match` | Exact postal code match (1/0) |
| `postal_both_present` | Both have postal codes (1/0) |
| `same_country` | Same country label (1/0) |
| `name_len_diff` | Absolute length difference (normalized names) |
| `addr_len_diff` | Absolute length difference (normalized addresses) |

All features computed with pure Python/NumPy; `rapidfuzz` used optionally for Levenshtein speed.

### 4. Matching Model (`pipeline.py`)

**Classifier:** `HistGradientBoostingClassifier` (scikit-learn)
- Native gradient boosting, BSD-licensed, no GPU needed
- `max_depth=6`, `max_iter=300`, `learning_rate=0.06`, `l2_regularization=1.0`
- `class_weight="balanced"` (matches are rare minority)

**Threshold Tuning (Critical for F₀.₅):**
- GroupShuffleSplit on S1 entities (no pair-level leakage)
- Sweep probability threshold 0.05–0.95 on validation split
- Select threshold maximizing **macro F₀.₅** (competition metric)
- Retrain on full training data at chosen threshold
- Save model + threshold together (`.pkl` + `.meta.json`)

### 5. Inference & Output

1. Load test sources
2. Run blocking → candidate pairs
3. Build features → score with trained model
4. Apply tuned threshold → final matches
5. Write both outputs in exact required format:
   - `matching_results.tsv`: `source1_entity_id\tmatched_entity_ids` (one row per S1, empty = singleton)
   - `candidate_pairs.tsv`: `source1_entity_id\tcandidate_entity_ids` (all blocking candidates)

---

## How to Run

### Setup
```bash
cd code/business_entity_resolution
pip install -r requirements.txt --break-system-packages
# or use a virtual environment
```

### Data Layout
Place challenge files at:
```
student_resource/
  dataset/
    train/
      train_source1.tsv
      train_source2.tsv
      train_source3.tsv
      train_ground_truth.tsv
    test/
      test_source1.tsv
      test_source2.tsv
      test_source3.tsv
```

### Train
```bash
cd src
python pipeline.py train --train-dir ../../../dataset/train --model-out ../models/model.pkl
```
Outputs: `models/model.pkl` + `models/model.pkl.meta.json` (contains tuned threshold)

### Predict
```bash
python pipeline.py predict --test-dir ../../../dataset/test --model ../models/model.pkl --output-dir ../../../output
```
Outputs: `output/matching_results.tsv` + `output/candidate_pairs.tsv`

### Validate Before Submitting
```bash
cd ../../..  # back to student_resource/
python utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test
```

---

## Key Fixes & Improvements Over Baseline

| Issue | Fix |
|-------|-----|
| Legal suffixes (ltd, pvt, llc, inc) not removed | Added to `_NAME_WORD_MAP` with empty string + `_STOPWORDS_NAME` |
| Common address tokens (rd, st, dr, no, etc.) flooding indices | Added `_STOPWORDS_ADDRESS` |
| Country hard-filtering breaks on "France" | Country never used in blocking; only as `same_country` feature |
| Low blocking recall ceiling | Hybrid TF-IDF + token + postal + prefix signals; increased candidate caps |
| Slow `.map()` on large DataFrames | Replaced with list comprehensions |
| Full-file reads before `.head()` | Added `nrows` parameter to `read_source` / `read_ground_truth` |
| Generic 0.5 threshold | Threshold swept on validation to maximize macro F₀.₅ |
| Missing singleton handling | All S1 entities written; empty list = correctly predicted singleton (scores 1.0 F₀.₅) |

---

## Expected Performance

On the real-sample training data (10K S1, ~1M S2/S3):
- Blocking recall ceiling: ~85–95% (vs <1% with naive blocking)
- Validation macro F₀.₅: >0.70 (vs ~0.04 baseline all-empty)
- Inference time: ~10–20 min on full test set (~1.7M entities)

---

## Files in Submission Package

```
<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       │   ├── pipeline.py          # Main entry point (train + predict)
│       │   ├── normalize.py         # Text normalization
│       │   ├── similarity.py        # String similarities (Levenshtein, Jaccard, etc.)
│       │   ├── io_utils.py          # TSV I/O with nrows support
│       │   └── __init__.py
│       ├── README.md                # This file
│       └── requirements.txt         # Pinned dependencies
└── Documentation_template.md        # Filled methodology document
```

---

## Compliance Checklist

- [ ] No external data/APIs/geocoding used
- [ ] All country labels treated as open set (no hard-coded US/India)
- [ ] Model is HistGradientBoostingClassifier (BSD license, classical ML, <<8B params)
- [ ] Output format exactly matches specification (tab-separated, correct headers)
- [ ] Every test S1 entity has exactly one row in both outputs
- [ ] No S1 self-matches, no invalid prefixes, no duplicates within lists
- [ ] All matched IDs are subset of candidate IDs
- [ ] `validate_submission.py` passes without errors

---

## Methodology Document Pointers

When filling `Documentation_template.md`, emphasize:
1. **Blocking strategy**: Hybrid TF-IDF + multi-signal token index + postal code
2. **Normalization**: Accent folding, legal suffix removal, address abbreviation expansion, landmark stripping
3. **Features**: 16 pairwise similarities covering name/address at token, character, and structural levels
4. **Model**: HistGradientBoostingClassifier with class balancing
5. **Threshold tuning**: Direct macro F₀.₅ optimization on grouped validation split
5. **Scalability**: Batched sparse cosine similarity, list-comprehension preprocessing, memory-efficient indices

---

*Built for Amazon ML Challenge 2026 — Business Entity Resolution*