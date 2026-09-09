"""Search on historical outcomes, freeze a candidate, then review on later outcomes.

Run from the repository root: python3 analysis/refresh.py --help
"""
import argparse
import copy
import hashlib
import importlib.metadata
import json
import math
import platform
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine.scoring_engine import evaluate, load_ground_truth, load_spec, run_autoresearch, score_vectorized, validate_inputs

METRICS = ("auc_roc_retain", "auc_roc_churn", "revenue_capture_at_20", "tier_separation")
DEFAULT_TIERS = {"A": {"min": 60, "max": 101}, "B": {"min": 35, "max": 60},
                 "C": {"min": 15, "max": 35}, "D": {"min": 0, "max": 15}}


def fingerprint(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def validate_periods(train, holdout, as_of, id_column):
    """Require fully mature, strictly later observations; never randomly split rows."""
    cutoff = pd.to_datetime(as_of, utc=True, errors="raise")
    if pd.isna(cutoff):
        raise ValueError("as_of must be a valid timestamp")
    windows = []
    for name, frame in (("search", train), ("holdout", holdout)):
        required = {id_column, "features_asof", "outcome_start", "outcome_end"}
        if required - set(frame):
            raise ValueError(f"{name}: missing time/identity columns {sorted(required - set(frame))}")
        if frame.empty or frame[id_column].isna().any() or frame[id_column].astype(str).duplicated().any():
            raise ValueError(f"{name}: require nonempty data and unique, non-null account IDs")
        dates = {col: pd.to_datetime(frame[col], utc=True, errors="coerce", format="ISO8601")
                 for col in ("features_asof", "outcome_start", "outcome_end")}
        if any(series.isna().any() for series in dates.values()):
            raise ValueError(f"{name}: invalid or missing outcome timestamps")
        if not ((dates["features_asof"] < dates["outcome_start"]) &
                (dates["outcome_start"] <= dates["outcome_end"]) &
                (dates["outcome_end"] <= cutoff)).all():
            raise ValueError(f"{name}: features must precede outcomes and all outcomes must be mature at as_of")
        windows.append(dates)
    if windows[0]["outcome_end"].max() >= windows[1]["features_asof"].min():
        raise ValueError("Search outcomes must finish before all holdout feature snapshots")
    return {name: {col: {"min": series.min().isoformat(), "max": series.max().isoformat()}
                   for col, series in dates.items()}
            for name, dates in zip(("search", "holdout"), windows)}


def assign_tiers(scores, spec):
    tiers = spec.get("tiers", DEFAULT_TIERS)
    if not tiers:
        raise ValueError("Tier definitions cannot be empty")
    intervals = sorted((b["min"], b["max"]) for b in tiers.values())
    if any(not (math.isfinite(lo) and math.isfinite(hi) and lo < hi) for lo, hi in intervals):
        raise ValueError("Tier bounds must be finite, increasing intervals")
    if (intervals[0][0] > 0 or intervals[-1][1] <= spec.get("max_score", 100) or
            any(a[1] != b[0] for a, b in zip(intervals, intervals[1:]))):
        raise ValueError("Tiers must cover the full score range without gaps or overlaps")
    labels = np.empty(len(scores), dtype=object)
    for label, bounds in tiers.items():
        labels[(scores >= bounds["min"]) & (scores < bounds["max"])] = label
    return labels


def measured(frame, spec, metric):
    result = evaluate(frame, score_vectorized(frame, spec), spec)
    if metric not in result or not math.isfinite(result[metric]):
        raise ValueError(f"Primary metric {metric} unavailable: check sample size, labels and dependencies")
    return result


def compare(holdout, incumbent, candidate, metric, min_gain, max_tier_movement,
            segment_column=None, max_segment_drop=0.02, id_column="account_id"):
    """Evaluate frozen specs on the same later observations, with explicit gates."""
    before, after = copy.deepcopy(incumbent), copy.deepcopy(candidate)
    for config in (before, after):
        for definition in config["features"].values():
            definition.pop("weight", None)
    if before != after:
        raise ValueError("Refresh comparison permits weight changes only; definitions must remain fixed")
    old, new = measured(holdout, incumbent, metric), measured(holdout, candidate, metric)
    old_scores, new_scores = score_vectorized(holdout, incumbent), score_vectorized(holdout, candidate)
    old_tiers, new_tiers = assign_tiers(old_scores, incumbent), assign_tiers(new_scores, candidate)
    moved = old_tiers != new_tiers
    transitions = pd.crosstab(pd.Series(old_tiers, name="from"), pd.Series(new_tiers, name="to"))
    gain = round(new[metric] - old[metric], 6)
    reasons = []
    if gain <= 0 or gain < min_gain:
        reasons.append(f"Holdout primary gain {gain:+.4f} does not meet a positive improvement of at least {min_gain:.4f}.")
    if moved.mean() > max_tier_movement:
        reasons.append(f"Tier movement {moved.mean():.1%} exceeds the configured {max_tier_movement:.1%} limit.")
    segments = []
    if segment_column:
        if segment_column not in holdout or holdout[segment_column].isna().any():
            raise ValueError(f"Segment column {segment_column!r} is missing or contains unknowns")
        for label, group in holdout.groupby(segment_column, sort=True):
            row = {"segment": str(label), "n": len(group)}
            a = evaluate(group, score_vectorized(group, incumbent), incumbent).get(metric)
            b = evaluate(group, score_vectorized(group, candidate), candidate).get(metric)
            row.update(incumbent=a, candidate=b, delta=None if a is None or b is None else round(b-a, 6))
            segments.append(row)
            if row["delta"] is None:
                reasons.append(f"Segment {label!r} has insufficient evidence for {metric}; gate cannot be verified.")
            elif row["delta"] < -max_segment_drop:
                reasons.append(f"Segment {label!r} deteriorates by {-row['delta']:.4f}, exceeding {max_segment_drop:.4f}.")
    changes = pd.DataFrame({id_column: holdout[id_column].values, "incumbent_score": old_scores,
                            "candidate_score": new_scores, "incumbent_tier": old_tiers,
                            "candidate_tier": new_tiers, "tier_changed": moved})
    contributions = {}
    for feature, definition in incumbent["features"].items():
        contributions[feature] = holdout[feature].values * (candidate["features"][feature]["weight"] - definition["weight"])
    changes["weight_change_contributions_before_clipping"] = [
        json.dumps({f: round(float(values[i]), 4) for f, values in contributions.items() if values[i] != 0})
        for i in range(len(holdout))]
    return {"decision": "KEEP_INCUMBENT" if reasons else "READY_FOR_REVIEW",
            "reasons": reasons, "holdout": {"incumbent": old, "candidate": new, "primary_delta": gain},
            "tier_movement_fraction": float(moved.mean()), "tier_transitions": transitions.to_dict(),
            "segments": segments}, changes


def drift(train, holdout, features):
    rows = []
    for feature in features:
        a, b = train[feature], holdout[feature]
        rows.append({"feature": feature, "search_mean": float(a.mean()), "holdout_mean": float(b.mean()),
                     "search_missing_fraction": float(a.isna().mean()),
                     "holdout_missing_fraction": float(b.isna().mean()),
                     "search_zero_fraction": float((a == 0).mean()),
                     "holdout_zero_fraction": float((b == 0).mean())})
    return rows


def cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def render_report(report):
    h = report["holdout"]
    metric = report["metric"]
    lines = ["# Should we replace our customer score?", "", f"**Decision: {report['decision']}**", "",
             "Keep the incumbent." if report["decision"] == "KEEP_INCUMBENT" else
             "The candidate passes the configured business gates and is ready for human review.", "",
             "This is an observational comparison, not proof of causal revenue lift or statistical significance.", "",
             "## What changed", "", "| Feature | Incumbent weight | Candidate weight |", "|---|---:|---:|"]
    for name, weights in report["weights"].items():
        lines.append(f"| {cell(name)} | {weights['incumbent']} | {weights['candidate']} |")
    lines += ["", "Feature definitions, outcome mappings and tier boundaries were frozen. Only weights changed.",
              "", "## Does it hold up on later outcomes?", "", "| Dataset | Incumbent | Candidate | Change |",
              "|---|---:|---:|---:|"]
    for name, values in (("Search (used for optimization)", report["search"]), ("Later holdout", h)):
        lines.append(f"| {name} | {values['incumbent'][metric]:.4f} | {values['candidate'][metric]:.4f} | {values['candidate'][metric]-values['incumbent'][metric]:+.4f} |")
    lines += ["", f"Primary metric: `{metric}`. Minimum gain: {report['policy']['min_gain']:.4f}.",
              "The holdout was evaluated after the candidate was frozen. Do not use it for another tuning cycle.",
              "", "## Review gates", ""]
    lines += [f"- {cell(reason)}" for reason in report["reasons"]] or ["- All configured business gates passed."]
    if not report["segments"]:
        lines.append("- Segment checks were not configured; segment safety is unverified.")
    lines += ["", "## Which accounts moved?", "",
              f"{report['tier_movement_fraction']:.1%} of holdout accounts changed tier. Limit: {report['policy']['max_tier_movement']:.1%}.",
              "Account-level scores and weight-change contributions are in `account_changes.csv` (local, potentially sensitive).",
              "Contributions explain arithmetic before score clipping; they do not establish causality.", "",
              "| From tier | To tier | Accounts |", "|---|---|---:|"]
    for to, counts in report["tier_transitions"].items():
        for frm, count in counts.items():
            if count:
                lines.append(f"| {cell(frm)} | {cell(to)} | {count} |")
    lines += ["", "## Segment checks", "", "| Segment | Accounts | Incumbent | Candidate | Change |",
              "|---|---:|---:|---:|---:|"]
    for row in report["segments"]:
        vals = ["unavailable" if row[k] is None else f"{row[k]:.4f}" for k in ("incumbent", "candidate", "delta")]
        lines.append(f"| {cell(row['segment'])} | {row['n']} | {' | '.join(vals)} |")
    lines += ["", "## Data comparison", "",
              "Different time periods/cohorts can have different populations. These differences are diagnostics, not model effects.",
              "Strict validation rejects missing numeric features/outcomes; reported zeroes may still be upstream imputations.",
              "", "| Feature | Search mean | Holdout mean | Search zero % | Holdout zero % |", "|---|---:|---:|---:|---:|"]
    for row in report["data_drift"]:
        lines.append(f"| {cell(row['feature'])} | {row['search_mean']:.3f} | {row['holdout_mean']:.3f} | {row['search_zero_fraction']:.1%} | {row['holdout_zero_fraction']:.1%} |")
    lines += ["", "Source freshness and instrumentation coverage cannot be inferred from feature values. Verify these upstream.",
              "Declared feature/outcome windows were checked for temporal ordering and maturity; this cannot prove correct upstream construction.",
              "", "## Other holdout metrics", "", "| Metric | Incumbent | Candidate |", "|---|---|---|"]
    for key in METRICS:
        if key in h["incumbent"] and key in h["candidate"]:
            lines.append(f"| {key} | {h['incumbent'][key]:.4f} | {h['candidate'][key]:.4f} |")
    lines += ["", "## Reproduce and review", "", "`manifest.json` records input, spec and code SHA-256 fingerprints, dependencies, seed, policy and time windows.",
              "`report.json` contains the complete metrics, drift diagnostics and decision. No production configuration is changed.", ""]
    return "\n".join(lines)


def run_refresh(train_path, holdout_path, spec_path, output_dir, as_of, metric="auc_roc_retain",
                n=500, seed=42, min_gain=0.01, max_tier_movement=0.25,
                segment_column=None, max_segment_drop=0.02, id_column="account_id"):
    if metric not in METRICS or n < 1:
        raise ValueError("Choose a supported metric and positive experiment count")
    if any(not math.isfinite(x) or x < 0 for x in (min_gain, max_tier_movement, max_segment_drop)) or max_tier_movement > 1:
        raise ValueError("Gate thresholds must be finite and nonnegative; tier movement must be <= 1")
    output = Path(output_dir)
    if output.exists():
        raise ValueError("Output directory already exists; use a fresh directory to preserve the review record")
    raw_train, raw_holdout = load_ground_truth(train_path), load_ground_truth(holdout_path)
    windows = validate_periods(raw_train, raw_holdout, as_of, id_column)
    incumbent = load_spec(spec_path)
    metadata = {id_column, "features_asof", "outcome_start", "outcome_end"}
    if segment_column:
        metadata.add(segment_column)
    if metadata & set(incumbent.get("features", {})):
        raise ValueError("Identity, time and segment metadata cannot be scoring features")
    train = validate_inputs(raw_train, incumbent, metric)
    holdout = validate_inputs(raw_holdout, incumbent, metric)
    assign_tiers(np.array([0, incumbent.get("max_score", 100)]), incumbent)
    measured(train, incumbent, metric)
    # Check label availability only. Candidate selection below never receives holdout data.
    if metric.startswith("auc_roc"):
        label = incumbent.get("retain_column", "retained_6mo") if metric == "auc_roc_retain" else incumbent.get("churn_column", "churned")
        if len(holdout) <= 100 or holdout[label].nunique() != 2:
            raise ValueError("Holdout requires more than 100 rows and both outcome classes")
    if metric == "revenue_capture_at_20" and not (holdout[incumbent.get("mrr_column", "mrr")] > 0).any():
        raise ValueError("Holdout requires positive revenue")
    if metric == "tier_separation" and (holdout[incumbent.get("ltv_column", "actual_ltv")] > 0).sum() <= 50:
        raise ValueError("Holdout requires more than 50 positive-LTV accounts")
    if segment_column and (segment_column not in holdout or holdout[segment_column].isna().any()):
        raise ValueError("Holdout segment column is missing or contains unknowns")
    inputs = {name: fingerprint(path) for name, path in
              (("search", train_path), ("holdout", holdout_path), ("incumbent_spec", spec_path))}
    output.mkdir(parents=True)
    save_json(output / "incumbent_spec.json", incumbent)
    candidate, _ = run_autoresearch(n, metric, str(train_path), str(spec_path), seed, str(output / "search"))
    # Freeze the candidate on disk before evaluating it against later outcomes.
    save_json(output / "candidate_spec.json", candidate)
    # Detect input changes during optimization instead of certifying mismatched evidence.
    current_inputs = {name: fingerprint(path) for name, path in
                      (("search", train_path), ("holdout", holdout_path), ("incumbent_spec", spec_path))}
    if current_inputs != inputs:
        raise ValueError("Inputs changed during the run; discard this incomplete review and use fixed snapshots")
    result, changes = compare(holdout, incumbent, candidate, metric, min_gain, max_tier_movement,
                              segment_column, max_segment_drop, id_column)
    result.update(metric=metric,
                  search={"incumbent": measured(train, incumbent, metric), "candidate": measured(train, candidate, metric)},
                  weights={f: {"incumbent": v["weight"], "candidate": candidate["features"][f]["weight"]}
                           for f, v in incumbent["features"].items()},
                  data_drift=drift(train, holdout, incumbent["features"]),
                  policy={"min_gain": min_gain, "max_tier_movement": max_tier_movement,
                          "segment_column": segment_column, "max_segment_drop": max_segment_drop})
    manifest = {"run_id": str(uuid.uuid4()), "created_at": datetime.now(timezone.utc).isoformat(), "report_version": 1, "evaluation_version": 2,
                "inputs_sha256": inputs, "candidate_sha256": fingerprint(output / "candidate_spec.json"),
                "code_sha256": {p.name: fingerprint(p) for p in (Path(__file__), Path(__file__).parents[1] / "engine/scoring_engine.py")},
                "python": platform.python_version(),
                "dependencies": {pkg: importlib.metadata.version(pkg) for pkg in ("numpy", "pandas", "pyarrow", "scikit-learn")},
                "seed": seed, "experiments": n, "metric": metric, "as_of": str(as_of), "windows": windows,
                "row_counts": {"search": len(train), "holdout": len(holdout)}, "policy": result["policy"],
                "id_column": id_column, "holdout_reuse": "Not detectable across separate invocations; operator must protect final holdout"}
    save_json(output / "manifest.json", manifest)
    save_json(output / "report.json", result)
    changes.to_csv(output / "account_changes.csv", index=False)
    (output / "report.md").write_text(render_report(result))
    print(f"\n{result['decision']}: {output / 'report.md'}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("train", "holdout", "spec", "output-dir", "as-of"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--metric", choices=METRICS, default="auc_roc_retain")
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-gain", type=float, default=0.01)
    parser.add_argument("--max-tier-movement", type=float, default=0.25)
    parser.add_argument("--segment-column")
    parser.add_argument("--max-segment-drop", type=float, default=0.02)
    parser.add_argument("--id-column", default="account_id")
    args = parser.parse_args()
    try:
        run_refresh(args.train, args.holdout, args.spec, args.output_dir, args.as_of, args.metric,
                    args.n, args.seed, args.min_gain, args.max_tier_movement, args.segment_column,
                    args.max_segment_drop, args.id_column)
    except ValueError as error:
        parser.error(str(error))
