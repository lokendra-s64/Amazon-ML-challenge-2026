"""I/O helpers for the challenge's tab-separated file formats."""
from __future__ import annotations

import pandas as pd


def read_source(path, nrows=None):
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, nrows=nrows)
    df["business_name"] = df["business_name"].fillna("")
    df["business_address"] = df["business_address"].fillna("")
    df["country"] = df["country"].fillna("")
    return df


def read_ground_truth(path, s1_ids=None):
    """Read ground truth, optionally filtering to only given S1 IDs."""
    import pandas as pd
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    out = {}
    for row in df.itertuples(index=False):
        if s1_ids is not None and row.source1_entity_id not in s1_ids:
            continue
        ids = {x for x in str(row.matched_entity_ids).split(",") if x}
        out[row.source1_entity_id] = ids
    return out


def write_id_list_tsv(path, id_col, list_col, mapping, s1_ids):
    """Write a two-column TSV: one row per s1 id, comma-joined list (empty allowed)."""
    rows = []
    for sid in s1_ids:
        ids = mapping.get(sid, [])
        # de-dup while preserving order, defensive even if callers already do this
        seen = dict.fromkeys(ids)
        rows.append({id_col: sid, list_col: ",".join(seen)})
    pd.DataFrame(rows, columns=[id_col, list_col]).to_csv(path, sep="\t", index=False, encoding="utf-8")


def write_matching_results(path, matches, s1_ids):
    write_id_list_tsv(path, "source1_entity_id", "matched_entity_ids", matches, s1_ids)


def write_candidate_pairs(path, candidates, s1_ids):
    write_id_list_tsv(path, "source1_entity_id", "candidate_entity_ids", candidates, s1_ids)
