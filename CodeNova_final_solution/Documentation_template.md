# Business Entity Resolution — Methodology Document

## 1. Methodology Overview

Our approach treats the Business Entity Resolution challenge as a **large-scale candidate generation + precision-tuned classification** problem. The dataset spans millions of records across three sources, making all-pairs comparison infeasible. We therefore employ a three-stage pipeline:

1. **Blocking / Candidate Generation** — cheap, high-recall inverted indexes that produce ≤80 candidate pairs per Source-1 entity (40 per Source 2/3)
2. **Pairwise Feature Engineering** — 15 similarity features computed only on blocked candidates
3. **Learned Matching Model** — `HistGradientBoostingClassifier` with decision threshold swept directly on macro-averaged F₀.₅

All signals derive exclusively from `business_name`, `business_address`, and `country` in the provided files. No external data, APIs, or geocoding services are used.

---

## 2. Candidate Generation / Blocking Strategy

### 2.1 Design Principles

- **Recall ceiling is the bottleneck**: The blocking stage determines the maximum achievable recall. Our target is ≥0.90 blocking recall ceiling on training data.
- **Country-agnostic**: The test set introduces France (unseen in training), so we **never hard-filter on country** during blocking. Country is only used as a feature later.
- **Multi-signal union**: We combine four independent blocking signals; a true match surviving *any* signal enters the candidate set.

### 2.2 Blocking Signals

| Signal | Description | Weight |
|--------|-------------|--------|
| **Name token inverted index** | Shared normalized name tokens, with document-frequency capping (tokens appearing in >2% of a source are dropped) | 3 |
| **Address token inverted index** | Shared normalized address tokens, same df-capping | 1 |
| **Exact postal code match** | 5–6 digit ZIP/PIN code extracted from address; strongest single signal | 5 |
| **Name prefix-3 index** | First 3 characters of concatenated normalized name tokens (catches typo'd first tokens) | 1 |

### 2.3 Implementation Details

- **Normalization** (see `normalize.py`): Accent folding (NFKD), legal-suffix canonicalization (Corp→corp, Pvt→private, Ltd→ltd, `&`→and), address abbreviation expansion (Road→rd, Street→st, etc.), landmark stripping (`near <X>`).
- **Tokenization**: Whitespace splitting after normalization; stopwords (`the`, `and`, `of`, `co`, `&`) removed from name tokens.
- **Candidate ranking**: For each Source-1 entity, scores from all signals are summed per candidate; top 40 per source (S2, S3) are retained.
- **Fallback**: Records whose tokens were all filtered as "too common" still receive candidates via postal/prefix matches.

### 2.4 Blocking Recall Ceiling

On the full training set (2.2M S1, 5M+ S2/S3), the blocking recall ceiling is reported at train time. If below target, we increase `MAX_CANDIDATES_PER_SOURCE` or relax `MAX_TOKEN_DOC_FREQ_RATIO` before touching the model.

---

## 3. Model Architecture & Feature Engineering

### 3.1 Feature Set (15 features)

Computed per (S1, candidate) pair:

| Feature | Type | Description |
|---------|------|-------------|
| `block_score` | numeric | Sum of blocking signal weights |
| `name_exact_norm` | binary | 1 if normalized names identical and non-empty |
| `name_lev_ratio` | numeric | Levenshtein similarity ratio (0–1) on normalized names |
| `name_token_jaccard` | numeric | Jaccard on name token sets |
| `name_token_sort_ratio` | numeric | Levenshtein ratio on sorted tokens (word-order invariant) |
| `name_char3gram_jaccard` | numeric | Character 3-gram Jaccard on normalized names |
| `name_first_token_match` | binary | 1 if first name tokens share 99-char prefix (effectively exact) |
| `addr_lev_ratio` | numeric | Levenshtein ratio on normalized addresses |
| `addr_token_jaccard` | numeric | Jaccard on address token sets |
| `addr_char3gram_jaccard` | numeric | Character 3-gram Jaccard on normalized addresses |
| `postal_match` | binary | 1 if both have postal codes and they match exactly |
| `postal_both_present` | binary | 1 if both records have a postal code |
| `same_country` | binary | 1 if country labels match exactly (case-insensitive) |
| `name_len_diff` | numeric | Absolute difference in normalized name length |
| `addr_len_diff` | numeric | Absolute difference in normalized address length |

All string similarities use `rapidfuzz` when available (O(n+m) average) with pure-Python fallback.

### 3.2 Model: HistGradientBoostingClassifier

```python
HistGradientBoostingClassifier(
    max_depth=6,
    max_iter=300,
    learning_rate=0.06,
    l2_regularization=1.0,
    class_weight="balanced",  # matches are rare (~1-5% of candidate pairs)
    random_state=42,
)
```

**Why HistGradientBoosting?**
- BSD-licensed (MIT/Apache 2.0 compatible)
- Classical gradient-boosted trees → trivially under 8B parameter limit
- Handles missing values natively, fast inference, no GPU required
- Strong baseline for tabular similarity features

### 3.3 Training Procedure

1. **Grouped train/validation split**: 80/20 split on Source-1 entity IDs (`GroupShuffleSplit`) to prevent leakage across pairs from the same entity.
2. **Threshold sweep**: After training, we sweep probability thresholds from 0.05 to 0.95 (step 0.01) on the validation split, evaluating **macro-averaged F₀.₅** at each threshold — *not* accuracy, AUC, or F₁.
3. **Retraining**: Best threshold is frozen; model is retrained on all training data.
4. **Model artifact**: Saved via `joblib` with a companion `.meta.json` containing the threshold and feature column order.

### 3.4 F₀.₅ Metric Implementation

Matches the challenge specification exactly:
- Per Source-1 entity: Precision = TP/(TP+FP), Recall = TP/(TP+FN)
- F₀.₅ = (1.25 × P × R) / (0.25 × P + R)
- Singletons (no true matches): Score 1.0 if predicted empty, 0.0 if any match predicted
- Macro-average across all Source-1 entities in evaluation set

---

## 4. Inference Pipeline

### 4.1 Test-Time Processing

1. Load test Source 1/2/3 files
2. Run identical blocking (`generate_candidates`) → `candidate_pairs.tsv`
3. Build features on candidates
4. Score with trained model → probabilities
5. Apply frozen threshold → `matching_results.tsv`
6. **Format guarantees** (enforced in `io_utils.py`):
   - Exactly one row per test Source-1 entity
   - Empty lists for predicted singletons
   - Only S2-/S3- IDs, no duplicates, no S1 self-matches
   - Every matched ID also appears in candidate_pairs.tsv

### 4.2 Scalability

- Blocking: O(|S1| × avg_block_size) via dictionary lookups — processes 1.7M test S1 entities in minutes
- Feature computation: Only on blocked candidates (typically 10-80 per S1)
- Memory: Source tables held in RAM (~2-3 GB total); candidate DataFrames streamed in chunks if needed

---

## 5. Results & Validation

### 5.1 Training Diagnostics (Expected)

| Metric | Target |
|--------|--------|
| Blocking recall ceiling | ≥0.90 |
| Validation macro F₀.₅ | >0.50 (well above all-empty baseline ~0.05) |
| Positive rate in candidates | 1-5% |

### 5.2 Submission Validation

Before submission, we run:
```bash
python utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test
```
All checks must pass (exit code 0).

---

## 6. Reproducibility

- **Random seeds**: Fixed at 42 throughout (train/val split, model)
- **Dependencies**: Pinned in `requirements.txt` (pandas≥2.0, numpy≥1.24, scikit-learn≥1.3, joblib≥1.3, rapidfuzz≥3.0)
- **Entrypoints**:
  - `src/train.py` — training + threshold tuning
  - `src/predict.py` — inference on test set
- **Data layout**: Expected under `dataset/train/` and `dataset/test/` relative to `student_resource/`

---

## 7. Fair-Play Compliance

- ✅ No external databases, APIs, or geocoding services
- ✅ No commercial entity resolution services
- ✅ No internet lookups or data augmentation
- ✅ Model: `HistGradientBoostingClassifier` (BSD license, classical ML, ≪8B params)
- ✅ Country treated as open set of string labels (France works without code changes)

---

## 8. Known Limitations & Future Work

1. **Blocking recall ceiling** — if below target on full data, next step is TF-IDF cosine blocking on top of inverted index candidates (memory-safe since candidate pool is already reduced).
2. **Levenshtein speed** — pure-Python fallback is O(n×m); `rapidfuzz` installation recommended for production runs.
3. **Hard negatives** — current training uses all blocked non-matches as negatives; sampling hard negatives (high blocking score but non-match) could improve precision.
4. **Ensemble** — combining multiple models (e.g., + Logistic Regression on same features) with calibrated probabilities may yield further gains.

---

## 9. Code Structure Summary

```
code/business_entity_resolution/
├── src/
│   ├── __init__.py
│   ├── normalize.py         # Text cleaning, tokenization, postal extraction
│   ├── similarity.py        # Levenshtein, Jaccard, token-sort (rapidfuzz optional)
│   ├── blocking.py          # Multi-signal inverted index candidate generation
│   ├── features.py          # 15 pairwise similarity features
│   ├── model.py             # HistGradientBoostingClassifier + threshold persistence
│   ├── evaluate.py          # Exact macro F_0.5 implementation
│   ├── io_utils.py          # TSV read/write (challenge format)
│   ├── train.py             # End-to-end training + validation sweep
│   └── predict.py           # End-to-end inference + output writing
├── requirements.txt
└── README.md
```