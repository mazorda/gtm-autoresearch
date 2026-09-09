"""Generic GTM scoring engine with autoresearch loop.

This is the generalized version - no client-specific logic.
Works with any ground truth that has features + outcomes.

Usage:
    python3 scoring_engine.py --n 2000 --metric revenue_capture_at_20
    python3 scoring_engine.py --n 2000 --metric auc_roc_churn
"""

import argparse
import copy
import json
import math
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


JOURNAL_PATH = Path(__file__).parent.parent / "experiment_journal.jsonl"


def load_ground_truth(path: str) -> pd.DataFrame:
    """Load ground truth from parquet."""
    return pq.read_table(path).to_pandas()


def load_spec(path: str) -> dict:
    """Load scoring spec from JSON."""
    with open(path) as f:
        return json.load(f)


def validate_inputs(df: pd.DataFrame, spec: dict, metric: str) -> pd.DataFrame:
    """Validate the data contract once, before any experiment or output write.

    Missing observations must be resolved explicitly upstream, never turned into
    evidence of inactivity or a negative outcome by this loader.
    """
    if df.empty:
        raise ValueError("Ground truth must contain at least one account")
    features = spec.get("features", {})
    if not features:
        raise ValueError("Scoring spec must define at least one feature")
    outcomes = {
        "revenue_capture_at_20": spec.get("mrr_column", "mrr"),
        "auc_roc_retain": spec.get("retain_column", "retained_6mo"),
        "auc_roc_churn": spec.get("churn_column", "churned"),
        "tier_separation": spec.get("ltv_column", "actual_ltv"),
    }
    if metric not in outcomes:
        raise ValueError(f"Unknown primary metric: {metric}")
    required = set(features) | set(spec.get("required_columns", [])) | {outcomes[metric]}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
    if set(features) & set(outcomes.values()):
        raise ValueError("Outcome columns cannot also be scoring features")
    result = df.copy()
    numeric = set(features) | (set(outcomes.values()) & set(df.columns))
    for col in sorted(numeric):
        values = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float, na_value=np.nan)
        if not np.isfinite(values).all():
            raise ValueError(f"Column {col!r} contains missing, nonnumeric or infinite values; resolve upstream")
        result[col] = values
    for key in ("auc_roc_retain", "auc_roc_churn"):
        col = outcomes[key]
        if col in result and not result[col].isin([0, 1]).all():
            raise ValueError(f"Outcome {col!r} must contain only binary 0/1 labels")
    mrr_col = outcomes["revenue_capture_at_20"]
    if mrr_col in result and (result[mrr_col] < 0).any():
        raise ValueError(f"Revenue column {mrr_col!r} must be nonnegative normalized recurring revenue")
    for name, definition in features.items():
        weight = definition["weight"]
        lo, hi = definition.get("min", -5), definition.get("max", 25)
        if not all(type(v) is int
                   for v in (weight, lo, hi)) or not lo <= weight <= hi:
            raise ValueError(f"Feature {name!r} needs integer weight and bounds with min <= weight <= max")
    if not math.isfinite(spec.get("max_score", 100)) or spec.get("max_score", 100) <= 0:
        raise ValueError("max_score must be positive and finite")
    return result


def score_vectorized(df: pd.DataFrame, spec: dict) -> np.ndarray:
    """Vectorized scoring. Returns array of scores."""
    features = spec["features"]
    scores = np.zeros(len(df))

    for feat_name, feat_def in features.items():
        w = feat_def["weight"]
        scores += df[feat_name].values * w

    max_score = spec.get("max_score", 100)
    scores = np.clip(scores, 0, max_score)
    return scores


def evaluate(df: pd.DataFrame, scores: np.ndarray, spec: dict) -> dict:
    """GTM-native evaluation harness. READ-ONLY during experiments.

    The agent can change weights. It cannot change this function.
    This is what makes it GTM-native: every metric here is something
    a RevOps or CS team uses to make resource decisions.
    """
    results = {"n_scored": len(scores)}
    results["score_mean"] = round(float(np.mean(scores)), 2)
    results["score_std"] = round(float(np.std(scores)), 2)

    mrr_col = spec.get("mrr_column", "mrr")
    retain_col = spec.get("retain_column", "retained_6mo")
    churn_col = spec.get("churn_column", "churned")
    ltv_col = spec.get("ltv_column", "actual_ltv")

    mrr = df[mrr_col].values if mrr_col in df.columns else np.zeros(len(df))
    retain = df[retain_col].values if retain_col in df.columns else None
    churn = df[churn_col].values if churn_col in df.columns else None
    ltv = df[ltv_col].values if ltv_col in df.columns else None

    # === PRIMARY: Revenue Capture @20% ===
    valid_mrr = mrr > 0
    if valid_mrr.sum() > 0:
        # Exact capacity: ceil(20% of positive-MRR accounts). Ties retain
        # input order; never use revenue as a tie-breaker.
        eligible = np.flatnonzero(valid_mrr)
        k = max(1, math.ceil(len(eligible) * 0.2))
        selected = eligible[np.argsort(-scores[eligible], kind="stable")[:k]]
        top_mrr = mrr[selected].sum()
        results["revenue_capture_selected_count"] = k
        total_mrr = mrr[valid_mrr].sum()
        results["revenue_capture_at_20"] = round(float(top_mrr / total_mrr), 4)

    # === CHURN PREDICTION ===
    if churn is not None:
        valid_churn = ~np.isnan(churn)
        if valid_churn.sum() > 100 and len(np.unique(churn[valid_churn])) == 2:
            try:
                from sklearn.metrics import roc_auc_score
                results["auc_roc_churn"] = round(float(
                    roc_auc_score(churn[valid_churn], -scores[valid_churn])), 4)
            except (ImportError, ValueError):
                pass

    # === RETENTION PREDICTION ===
    if retain is not None:
        valid_retain = ~np.isnan(retain)
        if valid_retain.sum() > 100:
            try:
                from sklearn.metrics import roc_auc_score
                results["auc_roc_retain"] = round(float(
                    roc_auc_score(retain[valid_retain], scores[valid_retain])), 4)
            except (ImportError, ValueError):
                pass

    # === TIER ASSIGNMENT ===
    tier_config = spec.get("tiers", {
        "A": {"min": 60, "max": 101},
        "B": {"min": 35, "max": 60},
        "C": {"min": 15, "max": 35},
        "D": {"min": 0, "max": 15},
    })
    tiers = np.full(len(scores), "D", dtype=object)
    for label, bounds in tier_config.items():
        mask = (scores >= bounds["min"]) & (scores < bounds["max"])
        tiers[mask] = label

    # === REVENUE-WEIGHTED BINARY RETENTION (NOT NRR/GRR) ===
    if retain is not None and valid_mrr.sum() > 0:
        revenue_weighted_retention_by_tier = {}
        for label in tier_config:
            mask = tiers == label
            if mask.sum() > 10:
                start = mrr[mask].sum()
                retained_mrr = mrr[mask & (retain == 1)].sum() if retain is not None else start
                revenue_weighted_retention_by_tier[label] = round(float(retained_mrr / start), 4) if start > 0 else None
        results["revenue_weighted_retention_by_tier"] = revenue_weighted_retention_by_tier

    # === LOGO RETENTION BY TIER ===
    if retain is not None:
        logo_by_tier = {}
        for label in tier_config:
            mask = tiers == label
            if mask.sum() > 10:
                logo_by_tier[label] = round(float(retain[mask].mean()), 4)
        results["logo_retention_by_tier"] = logo_by_tier

    # === TIER SEPARATION (LTV) ===
    if ltv is not None:
        valid_ltv = ltv > 0
        if valid_ltv.sum() > 50:
            q75 = np.percentile(scores[valid_ltv], 75)
            q25 = np.percentile(scores[valid_ltv], 25)
            top_ltv = ltv[(scores >= q75) & valid_ltv].mean()
            bot_ltv = ltv[(scores <= q25) & valid_ltv].mean()
            if bot_ltv > 0:
                results["tier_separation"] = round(float(top_ltv / bot_ltv), 4)

    # === REVENUE CONCENTRATION (HHI) ===
    if valid_mrr.sum() > 0:
        total = mrr[valid_mrr].sum()
        hhi = 0
        for label in tier_config:
            mask = (tiers == label) & valid_mrr
            share = mrr[mask].sum() / total if total > 0 else 0
            hhi += (share * 100) ** 2
        results["revenue_concentration_hhi"] = round(float(hhi), 0)

    # === ARPA BY TIER ===
    arpa_by_tier = {}
    for label in tier_config:
        mask = (tiers == label) & valid_mrr
        if mask.sum() > 0:
            arpa_by_tier[label] = round(float(mrr[mask].mean()), 2)
    results["arpa_by_tier"] = arpa_by_tier

    # === TIER DISTRIBUTION ===
    for label in tier_config:
        n = int((tiers == label).sum())
        results[f"pct_{label}_tier"] = round(n / len(scores), 4)

    return results


def random_mutate(spec: dict, n_changes: int = 2, magnitude: float = 0.3, rng=None) -> dict:
    """Mutate scoring weights."""
    if magnitude <= 0 or not math.isfinite(magnitude):
        raise ValueError("Mutation magnitude must be positive and finite")
    rng = rng or random
    new = copy.deepcopy(spec)
    features = new["features"]
    names = list(features.keys())
    changes = []

    for feat in rng.sample(names, min(n_changes, len(names))):
        fdef = features[feat]
        old_w = fdef["weight"]
        lo = fdef.get("min", -5)
        hi = fdef.get("max", 25)

        # At least one integer step is reachable even from zero or +/-1.
        radius = max(1, math.ceil(magnitude * max(abs(old_w), 1)))
        candidates = [w for w in range(max(lo, old_w - radius), min(hi, old_w + radius) + 1)
                      if w != old_w]
        new_w = rng.choice(candidates) if candidates else old_w

        if new_w != old_w:
            fdef["weight"] = new_w
            changes.append(f"{feat}: {old_w} -> {new_w}")

    return {
        "spec": new,
        "changes": changes,
        "hypothesis": f"Random mutation: {'; '.join(changes)}" if changes else "no-op"
    }


def run_autoresearch(n_experiments: int = 500, metric: str = "revenue_capture_at_20",
                     ground_truth_path: str = "synthetic_ground_truth.parquet",
                     spec_path: str = None, seed: int = None, output_dir: str = None):
    """Run the autoresearch loop."""
    print(f"\n{'='*60}")
    print(f"GTM AUTORESEARCH ENGINE")
    print(f"Metric: {metric} | Experiments: {n_experiments}")
    print(f"{'='*60}\n")

    if n_experiments < 1:
        raise ValueError("Number of experiments must be positive")
    rng = random.Random(seed)
    journal_path = Path(output_dir) / "experiment_journal.jsonl" if output_dir else JOURNAL_PATH
    best_path = journal_path.parent / "best_spec.json"
    df = load_ground_truth(ground_truth_path)
    print(f"Loaded {len(df):,} accounts, {len(df.columns)} columns")

    # Load or create spec
    if spec_path:
        spec = load_spec(spec_path)
    else:
        # Auto-detect features (exclude outcome columns and ID)
        exclude = {"account_id", "email", "retained_6mo", "churned",
                   "actual_ltv", "tenure_months", "mrr"}
        feature_cols = [c for c in df.columns if c not in exclude]
        spec = {
            "model": "health_score",
            "version": "auto_v1",
            "max_score": 100,
            "features": {col: {"weight": 5, "min": -5, "max": 25} for col in feature_cols},
            "tiers": {
                "A": {"min": 60, "max": 101},
                "B": {"min": 35, "max": 60},
                "C": {"min": 15, "max": 35},
                "D": {"min": 0, "max": 15},
            }
        }
        print(f"Auto-detected {len(feature_cols)} features: {', '.join(feature_cols)}")

    df = validate_inputs(df, spec, metric)

    # Load journal
    journal = []
    if journal_path.exists():
        with open(journal_path) as f:
            for line in f:
                if line.strip():
                    journal.append(json.loads(line))

    # Baseline
    baseline_scores = score_vectorized(df, spec)
    baseline_metrics = evaluate(df, baseline_scores, spec)
    if metric not in baseline_metrics or not np.isfinite(baseline_metrics[metric]):
        raise ValueError(f"Primary metric {metric!r} is unavailable: check sample size, outcome variation and dependencies")
    baseline_value = baseline_metrics[metric]
    journal_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Baseline {metric}: {baseline_value}")
    print(f"Score stats: mean={baseline_metrics['score_mean']}, std={baseline_metrics['score_std']}")
    for m in ["revenue_capture_at_20", "auc_roc_retain", "auc_roc_churn", "tier_separation"]:
        if m in baseline_metrics:
            print(f"  {m}: {baseline_metrics[m]}")

    best_value = baseline_value
    best_spec = spec
    wins = 0
    t_start = time.time()

    for i in range(n_experiments):
        exp_id = f"exp_{len(journal) + 1:05d}"
        t0 = time.time()

        hyp = random_mutate(best_spec, n_changes=rng.choice([1, 2, 3]), rng=rng)
        if not hyp["changes"]:
            continue

        variant_scores = score_vectorized(df, hyp["spec"])
        variant_metrics = evaluate(df, variant_scores, hyp["spec"])
        variant_value = variant_metrics[metric]

        outcome = "win" if variant_value > best_value else "lose"
        dt = time.time() - t0

        entry = {
            "experiment_id": exp_id,
            "seed": seed,
            "evaluation_version": 2,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metric": metric,
            "baseline_value": round(best_value, 6),
            "variant_value": round(variant_value, 6),
            "outcome": outcome,
            "changes": hyp["changes"],
            "hypothesis": hyp["hypothesis"],
            "weights_snapshot": {k: v["weight"] for k, v in hyp["spec"]["features"].items()},
            "metrics": variant_metrics,
            "duration_ms": round(dt * 1000, 1)
        }

        # Append to JSONL
        with open(journal_path, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")
        journal.append(entry)

        if outcome == "win":
            wins += 1
            best_value = variant_value
            best_spec = hyp["spec"]
            sym = "+"
        else:
            sym = "-"

        if (i + 1) % 50 == 0 or outcome == "win":
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed
            print(f"  [{sym}] {exp_id}: {metric}={variant_value:.4f} "
                  f"(best={best_value:.4f}) [{dt*1000:.0f}ms] "
                  f"[{i+1}/{n_experiments}, {rate:.0f}/s, {wins}W]")

    elapsed = time.time() - t_start

    # Final eval
    final_scores = score_vectorized(df, best_spec)
    final_metrics = evaluate(df, final_scores, best_spec)

    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"Experiments: {n_experiments} in {elapsed:.1f}s ({n_experiments/elapsed:.0f}/s)")
    print(f"Wins: {wins} ({100*wins/max(n_experiments,1):.1f}%)")

    print(f"\nBaseline -> Best:")
    for m in ["revenue_capture_at_20", "auc_roc_retain", "auc_roc_churn",
              "tier_separation", "revenue_concentration_hhi"]:
        b = baseline_metrics.get(m, "?")
        f = final_metrics.get(m, "?")
        print(f"  {m}: {b} -> {f}")

    if "revenue_weighted_retention_by_tier" in final_metrics:
        print(f"\nRevenue-weighted binary retention by tier: {final_metrics['revenue_weighted_retention_by_tier']}")
    if "logo_retention_by_tier" in final_metrics:
        print(f"Logo retention by tier: {final_metrics['logo_retention_by_tier']}")
    if "arpa_by_tier" in final_metrics:
        print(f"ARPA by tier: {final_metrics['arpa_by_tier']}")

    print(f"\nFinal weights (sorted by absolute weight):")
    weights = [(k, v["weight"]) for k, v in best_spec["features"].items()]
    weights.sort(key=lambda x: abs(x[1]), reverse=True)
    initial_weights = {k: v["weight"] for k, v in spec["features"].items()}
    for feat, w in weights:
        changed = " *" if w != initial_weights.get(feat, 5) else ""
        print(f"  {feat:25s} {w:>4d}{changed}")

    # Save best spec
    with open(best_path, "w") as f:
        json.dump(best_spec, f, indent=2)

    return best_spec, final_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GTM Autoresearch Scoring Engine")
    parser.add_argument("--n", type=int, default=500, help="Number of experiments")
    parser.add_argument("--metric", default="revenue_capture_at_20",
                        choices=["revenue_capture_at_20", "auc_roc_retain",
                                 "auc_roc_churn", "tier_separation"])
    parser.add_argument("--data", default="synthetic_ground_truth.parquet",
                        help="Path to ground truth parquet")
    parser.add_argument("--spec", default=None, help="Path to scoring spec JSON")
    parser.add_argument("--seed", type=int, help="Replay mutation choices with a fixed seed")
    parser.add_argument("--output-dir", help="Directory for this run journal and best spec")
    args = parser.parse_args()

    run_autoresearch(n_experiments=args.n, metric=args.metric,
                     ground_truth_path=args.data, spec_path=args.spec,
                     seed=args.seed, output_dir=args.output_dir)
