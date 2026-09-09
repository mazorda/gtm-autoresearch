"""Validation tests for the GTM autoresearch engine.

Run: python3 -m pytest tests/test_engine.py -v
Or:  python3 tests/test_engine.py
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from engine.scoring_engine import score_vectorized, evaluate, random_mutate, run_autoresearch


def make_test_data(n=1000):
    """Create minimal test dataset with known signals."""
    np.random.seed(123)
    df = pd.DataFrame({
        "account_id": [f"test_{i}" for i in range(n)],
        "signal_strong": np.random.binomial(1, 0.3, n).astype(float),
        "signal_medium": np.clip(np.random.exponential(0.3, n), 0, 1),
        "noise": np.random.uniform(0, 1, n),
        "mrr": np.clip(np.random.lognormal(5, 0.8, n), 50, 5000),
    })
    # Retention driven by signal_strong and signal_medium, NOT noise
    logit = 2.0 * df["signal_strong"] + 1.5 * df["signal_medium"] - 1.0
    prob = 1 / (1 + np.exp(-logit))
    df["retained_6mo"] = np.random.binomial(1, prob, n).astype(float)
    df["churned"] = 1.0 - df["retained_6mo"]
    df["actual_ltv"] = df["mrr"] * np.where(df["retained_6mo"] == 1, 12, 3)
    return df


def test_score_vectorized():
    df = make_test_data()
    spec = {
        "max_score": 100,
        "features": {
            "signal_strong": {"weight": 10, "min": 0, "max": 25},
            "signal_medium": {"weight": 5, "min": 0, "max": 25},
            "noise": {"weight": 1, "min": -5, "max": 25},
        }
    }
    scores = score_vectorized(df, spec)
    assert len(scores) == len(df)
    assert scores.min() >= 0
    assert scores.max() <= 100
    # Strong signal accounts should score higher
    strong_mean = scores[df["signal_strong"] == 1].mean()
    weak_mean = scores[df["signal_strong"] == 0].mean()
    assert strong_mean > weak_mean, f"Strong signal ({strong_mean:.1f}) should outscore weak ({weak_mean:.1f})"
    print("  test_score_vectorized: PASS")


def test_evaluate_returns_metrics():
    df = make_test_data()
    spec = {
        "max_score": 100,
        "features": {
            "signal_strong": {"weight": 10, "min": 0, "max": 25},
            "signal_medium": {"weight": 5, "min": 0, "max": 25},
        },
        "tiers": {"A": {"min": 8, "max": 101}, "B": {"min": 3, "max": 8}, "D": {"min": 0, "max": 3}}
    }
    scores = score_vectorized(df, spec)
    metrics = evaluate(df, scores, spec)

    assert "revenue_capture_at_20" in metrics
    assert "n_scored" in metrics
    assert metrics["n_scored"] == len(df)

    # Revenue-weighted binary retention should exist
    if "revenue_weighted_retention_by_tier" in metrics:
        assert isinstance(metrics["revenue_weighted_retention_by_tier"], dict)
        # Higher tiers should have higher weighted retention
        nrr = metrics["revenue_weighted_retention_by_tier"]
        if "A" in nrr and "D" in nrr and nrr["A"] is not None and nrr["D"] is not None:
            assert nrr["A"] >= nrr["D"], f"A-tier weighted retention ({nrr['A']}) should >= D-tier ({nrr['D']})"

    print("  test_evaluate_returns_metrics: PASS")


def test_random_mutate():
    spec = {
        "max_score": 100,
        "features": {
            "f1": {"weight": 5, "min": 0, "max": 25},
            "f2": {"weight": 10, "min": 0, "max": 25},
            "f3": {"weight": 3, "min": -5, "max": 25},
        }
    }
    result = random_mutate(spec, n_changes=2)
    assert "spec" in result
    assert "changes" in result
    assert "hypothesis" in result
    # Original should be unchanged
    assert spec["features"]["f1"]["weight"] == 5
    print("  test_random_mutate: PASS")


def test_signal_recovery():
    """The core validation: can the engine find known signals?"""
    df = make_test_data(n=2000)

    # Save to temp parquet
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        df.to_parquet(f.name, index=False)
        data_path = f.name

    with tempfile.TemporaryDirectory() as output_dir:
        best_spec, metrics = run_autoresearch(
            n_experiments=500,
            metric="auc_roc_retain",
            ground_truth_path=data_path,
            seed=123,
            output_dir=output_dir,
        )

    weights = {k: v["weight"] for k, v in best_spec["features"].items()}

    # signal_strong should have highest weight
    assert weights["signal_strong"] > weights["noise"], \
        f"signal_strong ({weights['signal_strong']}) should > noise ({weights['noise']})"

    # signal_medium should have higher weight than noise
    assert weights["signal_medium"] > weights["noise"], \
        f"signal_medium ({weights['signal_medium']}) should > noise ({weights['noise']})"

    # AUC should improve from baseline
    assert metrics.get("auc_roc_retain", 0) > 0.6, \
        f"AUC-ROC ({metrics.get('auc_roc_retain')}) should exceed 0.6"

    print("  test_signal_recovery: PASS")
    Path(data_path).unlink()


if __name__ == "__main__":
    print("Running gtm-autoresearch validation tests...\n")
    test_score_vectorized()
    test_evaluate_returns_metrics()
    test_random_mutate()
    test_signal_recovery()
    print("\nAll tests passed.")
