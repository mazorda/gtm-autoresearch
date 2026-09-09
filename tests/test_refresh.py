"""Trust-boundary and decision tests for the refresh workflow."""
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.refresh import assign_tiers, compare, fingerprint, run_refresh, validate_periods
from examples.scoring_refresh.generate_data import SPEC, generate, prepare


def frame(n=400, later=False):
    y = np.arange(n) % 2
    return pd.DataFrame({"account_id": [f"a{i}" for i in range(n)],
                         "core_workflow": y.astype(float), "legacy_clicks": 1-y,
                         "core_workflow_unknown": 0, "mrr": 100.,
                         "retained_6mo": y, "churned": 1-y,
                         "segment": ["one"] * (n//2) + ["two"] * (n - n//2),
                         "features_asof": "2025-08-01" if later else "2025-01-01",
                         "outcome_start": "2025-08-02" if later else "2025-01-02",
                         "outcome_end": "2026-02-01" if later else "2025-07-01"})


def test_known_improvement_can_reach_review():
    baseline = copy.deepcopy(SPEC)
    baseline["features"]["core_workflow"]["weight"] = 0
    report, changes = compare(frame(), baseline, SPEC, "auc_roc_retain", 0.01, 1, "segment")
    assert report["decision"] == "READY_FOR_REVIEW"
    assert report["holdout"]["primary_delta"] == 0.5
    assert len(changes) == 400
    assert changes["tier_changed"].sum() == 200
    assert json.loads(changes.iloc[1]["weight_change_contributions_before_clipping"])["core_workflow"] == 20


def test_segment_deterioration_and_primary_failure_keep_incumbent():
    candidate = copy.deepcopy(SPEC)
    candidate["features"]["core_workflow"]["weight"] = 0
    candidate["features"]["legacy_clicks"]["weight"] = 20
    report, _ = compare(frame(), SPEC, candidate, "auc_roc_retain", 0.01, 1, "segment")
    assert report["decision"] == "KEEP_INCUMBENT"
    assert any("Segment" in reason for reason in report["reasons"])


def test_tier_movement_gate_blocks_a_better_metric():
    baseline = copy.deepcopy(SPEC)
    baseline["features"]["core_workflow"]["weight"] = 0
    report, _ = compare(frame(), baseline, SPEC, "auc_roc_retain", 0.01, 0.1)
    assert report["holdout"]["primary_delta"] > 0
    assert report["decision"] == "KEEP_INCUMBENT"
    assert any("Tier movement" in r for r in report["reasons"])


def test_small_segments_are_unverified_not_silently_passed():
    baseline = copy.deepcopy(SPEC)
    baseline["features"]["core_workflow"]["weight"] = 0
    report, _ = compare(frame(200), baseline, SPEC, "auc_roc_retain", 0.01, 1, "segment")
    assert report["decision"] == "KEEP_INCUMBENT"
    assert all(row["delta"] is None for row in report["segments"])


def test_recut_definitions_are_rejected():
    candidate = copy.deepcopy(SPEC)
    candidate["tiers"]["A"]["min"] = 17
    with pytest.raises(ValueError, match="weight changes only"):
        compare(frame(), SPEC, candidate, "auc_roc_retain", 0.01, 1)


@pytest.mark.parametrize("problem", ["overlap", "immature", "invalid_date", "duplicate_id", "missing_id", "future_features"])
def test_invalid_time_and_identity_contracts(problem):
    train, holdout = frame(), frame(later=True)
    if problem == "overlap":
        holdout["features_asof"] = "2025-06-01"
    elif problem == "immature":
        holdout["outcome_end"] = "2027-01-01"
    elif problem == "invalid_date":
        holdout.loc[0, "outcome_start"] = "broken"
    elif problem == "duplicate_id":
        holdout.loc[0, "account_id"] = "a1"
    elif problem == "missing_id":
        holdout = holdout.drop(columns="account_id")
    else:
        holdout["features_asof"] = "2026-01-01"
    with pytest.raises(ValueError):
        validate_periods(train, holdout, "2026-02-02", "account_id")


def test_same_accounts_can_be_evaluated_at_a_strictly_later_snapshot():
    windows = validate_periods(frame(), frame(later=True), "2026-02-02", "account_id")
    assert set(windows) == {"search", "holdout"}


@pytest.mark.parametrize("tiers", [
    {"A": {"min": 0, "max": 50}, "B": {"min": 40, "max": 101}},
    {"A": {"min": 0, "max": 40}, "B": {"min": 50, "max": 101}},
    {"A": {"min": 0, "max": 100}},
])
def test_tier_gaps_overlaps_and_uncovered_max_are_rejected(tiers):
    with pytest.raises(ValueError, match="full score range"):
        assign_tiers(np.array([0, 100]), dict(SPEC, tiers=tiers))


def test_preparation_preserves_unknown_flag_and_normalizes_annual_billing():
    result = prepare(pd.DataFrame({"core_workflow": [np.nan, 1],
                                   "recurring_amount": [1200, 100], "billing_months": [12, 1]}))
    assert result["mrr"].tolist() == [100, 100]
    assert result["core_workflow_unknown"].tolist() == [1, 0]
    assert result["core_workflow"].tolist() == [0, 1]


def test_synthetic_walkthrough_rejects_historical_winner(tmp_path):
    data, output = tmp_path / "data", tmp_path / "report"
    generate(data)
    result = run_refresh(data / "search.parquet", data / "holdout.parquet", data / "incumbent.json",
                         output, "2026-02-02", n=200, segment_column="segment")
    assert result["search"]["candidate"]["auc_roc_retain"] > result["search"]["incumbent"]["auc_roc_retain"]
    assert result["holdout"]["primary_delta"] < 0
    assert result["decision"] == "KEEP_INCUMBENT"
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["inputs_sha256"]["search"] == fingerprint(data / "search.parquet")
    assert manifest["candidate_sha256"] == fingerprint(output / "candidate_spec.json")
    assert "KEEP_INCUMBENT" in (output / "report.md").read_text()
    assert len(pd.read_csv(output / "account_changes.csv")) == 1000
    with pytest.raises(ValueError, match="already exists"):
        run_refresh(data / "search.parquet", data / "holdout.parquet", data / "incumbent.json",
                    output, "2026-02-02", n=1)


def test_optimizer_never_receives_holdout(tmp_path, monkeypatch):
    import analysis.refresh as module
    train, holdout, spec = tmp_path / "train.parquet", tmp_path / "holdout.parquet", tmp_path / "spec.json"
    frame().to_parquet(train)
    frame(later=True).to_parquet(holdout)
    spec.write_text(json.dumps(SPEC))
    seen = []
    def search(n, metric, path, spec_path, seed, output):
        seen.append(path)
        return copy.deepcopy(SPEC), {}
    monkeypatch.setattr(module, "run_autoresearch", search)
    result = run_refresh(train, holdout, spec, tmp_path / "report", "2026-02-02")
    assert seen == [str(train)]
    assert result["decision"] == "KEEP_INCUMBENT"


def test_segment_gate_blocks_even_when_overall_auc_improves():
    df = frame(800)
    df["segment"] = ["small"] * 200 + ["large"] * 600
    df.loc[200:, "core_workflow"] = 0
    df.loc[200:, "legacy_clicks"] = df.loc[200:, "retained_6mo"]
    candidate = copy.deepcopy(SPEC)
    candidate["features"]["core_workflow"]["weight"] = 0
    candidate["features"]["legacy_clicks"]["weight"] = 20
    report, _ = compare(df, SPEC, candidate, "auc_roc_retain", 0.01, 1, "segment")
    assert report["holdout"]["primary_delta"] > 0.01
    assert report["decision"] == "KEEP_INCUMBENT"
    assert any("small" in reason for reason in report["reasons"])


def test_changed_input_cannot_get_a_completed_report(tmp_path, monkeypatch):
    import analysis.refresh as module
    train, holdout, spec = tmp_path / "train.parquet", tmp_path / "holdout.parquet", tmp_path / "spec.json"
    frame().to_parquet(train)
    frame(later=True).to_parquet(holdout)
    spec.write_text(json.dumps(SPEC))
    def search(*args):
        changed = frame()
        changed.loc[0, "mrr"] = 300
        changed.to_parquet(train)
        return copy.deepcopy(SPEC), {}
    monkeypatch.setattr(module, "run_autoresearch", search)
    output = tmp_path / "report"
    with pytest.raises(ValueError, match="Inputs changed"):
        run_refresh(train, holdout, spec, output, "2026-02-02")
    assert not (output / "report.json").exists()
