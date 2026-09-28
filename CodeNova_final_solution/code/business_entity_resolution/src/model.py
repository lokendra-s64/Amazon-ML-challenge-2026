"""
Matching model.

We use scikit-learn's HistGradientBoostingClassifier: a dependency-free,
BSD-licensed (fully compliant with the "MIT/Apache 2.0, <=8B params"
constraint -- it is a classical gradient-boosted tree ensemble, not a
neural net, so the parameter-count rule is trivially satisfied) gradient
boosting implementation that is fast and handles the tabular similarity
features from features.py well without needing GPUs or extra packages.

The important design choice for this challenge is the DECISION THRESHOLD,
not just the classifier: because F_0.5 weights precision 2x over recall
and rewards correctly-predicted singletons with a full 1.0, we tune the
probability threshold (and a "keep top-K per entity" cap) directly against
macro-averaged F_0.5 on a held-out validation split, instead of using the
default 0.5 cutoff or optimizing accuracy/AUC.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from features import FEATURE_COLUMNS


def train_model(X, y, random_state=42):
    clf = HistGradientBoostingClassifier(
        max_depth=6,
        max_iter=300,
        learning_rate=0.06,
        l2_regularization=1.0,
        class_weight="balanced",
        random_state=random_state,
    )
    clf.fit(X, y)
    return clf


def score_pairs(clf, features_df):
    X = features_df[FEATURE_COLUMNS].to_numpy(dtype=float)
    return clf.predict_proba(X)[:, 1]


def save_model(clf, threshold, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, path)
    with open(str(path) + ".meta.json", "w") as f:
        json.dump({"threshold": threshold, "feature_columns": FEATURE_COLUMNS}, f, indent=2)


def load_model(path):
    clf = joblib.load(path)
    with open(str(path) + ".meta.json") as f:
        meta = json.load(f)
    return clf, meta["threshold"]
