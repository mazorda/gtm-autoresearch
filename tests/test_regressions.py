"""Production-derived regressions, using entirely synthetic data."""
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine.scoring_engine import evaluate, load_ground_truth, random_mutate, run_autoresearch, validate_inputs
from analysis.review import review


def spec():
    return {"features": {"activity": {"weight": 0, "min": -5, "max": 25}}}


def data(n=100):
    return pd.DataFrame({"activity": np.linspace(0, 1, n), "mrr": np.full(n, 100.0)})


@pytest.mark.parametrize("n,expected_count", [(100, 20), (101, 21), (1, 1)])
def test_tied_scores_respect_capacity(n, expected_count):
    result = evaluate(data(n), np.zeros(n), spec())
    assert result["revenue_capture_selected_count"] == expected_count
    assert result["revenue_capture_at_20"] == round(expected_count / n, 4)


def test_ties_do_not_use_revenue_to_choose_winners():
    df = data(5)
    df["mrr"] = [1, 100, 100, 100, 100]
    result = evaluate(df, np.ones(5), spec())
    assert result["revenue_capture_at_20"] == round(1 / 401, 4)


def test_non_tied_ranking_excludes_zero_revenue():
    df = data(6)
    df["mrr"] = [0, 10, 20, 30, 40, 50]
    result = evaluate(df, np.array([100, 1, 2, 3, 4, 5]), spec())
    assert result["revenue_capture_selected_count"] == 1
    assert result["revenue_capture_at_20"] == round(50 / 150, 4)


@pytest.mark.parametrize("weight", [-1, 0, 1, -5, 25])
def test_small_and_boundary_weights_can_move(weight):
    original = spec()
    original["features"]["activity"]["weight"] = weight
    mutated = random_mutate(original, rng=random.Random(12))
    new_weight = mutated["spec"]["features"]["activity"]["weight"]
    assert new_weight != weight
    assert -5 <= new_weight <= 25
    assert original["features"]["activity"]["weight"] == weight


def test_fixed_weight_remains_fixed():
    original = {"features": {"locked": {"weight": 0, "min": 0, "max": 0}}}
    assert random_mutate(original)["changes"] == []


@pytest.mark.parametrize("column", ["activity", "mrr"])
def test_missing_required_inputs(column):
    with pytest.raises(ValueError, match=column):
        validate_inputs(data().drop(columns=column), spec(), "revenue_capture_at_20")


@pytest.mark.parametrize("bad", [None, "not a number", float("inf")])
def test_invalid_values_are_not_silently_zeroed(bad, tmp_path):
    df = data()
    df["activity"] = df["activity"].astype(object)
    df.loc[0, "activity"] = bad
    with pytest.raises(ValueError, match="activity"):
        validate_inputs(df, spec(), "revenue_capture_at_20")


def test_loader_preserves_missing_observations(tmp_path):
    df = data()
    df.loc[0, "activity"] = np.nan
    path = tmp_path / "data.parquet"
    df.to_parquet(path)
    assert pd.isna(load_ground_truth(path).loc[0, "activity"])


def test_missing_revenue_does_not_create_fake_revenue_metrics():
    result = evaluate(data().drop(columns="mrr"), np.zeros(100), spec())
    assert "revenue_capture_at_20" not in result
    assert "revenue_concentration_hhi" not in result


def test_retention_proxy_is_not_labeled_nrr_or_grr():
    df = data()
    df["retained_6mo"] = [1] * 50 + [0] * 50
    result = evaluate(df, np.zeros(100), spec())
    assert result["revenue_weighted_retention_by_tier"]["D"] == 0.5
    assert "nrr_by_tier" not in result
    assert "grr_by_tier" not in result


def test_custom_outcome_columns_and_explicit_required_columns():
    df = data().rename(columns={"mrr": "normalized_mrr"})
    config = spec() | {"mrr_column": "normalized_mrr", "required_columns": ["account_id"]}
    with pytest.raises(ValueError, match="account_id"):
        validate_inputs(df, config, "revenue_capture_at_20")
    df["account_id"] = [f"a{i}" for i in range(len(df))]
    assert len(validate_inputs(df, config, "revenue_capture_at_20")) == 100


def test_invalid_binary_outcomes():
    df = data()
    df["churned"] = 2
    with pytest.raises(ValueError, match="binary"):
        validate_inputs(df, spec(), "auc_roc_churn")


def test_seed_replays_weights_and_metrics(tmp_path):
    path = tmp_path / "data.parquet"
    df = data()
    df["mrr"] = np.arange(1, 101)
    df.to_parquet(path)
    results = []
    for name in ("first", "second"):
        results.append(run_autoresearch(20, ground_truth_path=str(path), seed=9,
                                       output_dir=str(tmp_path / name)))
    assert results[0] == results[1]
    entries = [json.loads(line) for line in (tmp_path / "first/experiment_journal.jsonl").read_text().splitlines()]
    assert entries and all(e["seed"] == 9 and e["evaluation_version"] == 2 for e in entries)


@pytest.mark.parametrize("case", ["no_revenue", "single_class", "small_sample", "bad_spec_path", "empty", "no_features"])
def test_unusable_runs_fail_without_writing_outputs(case, tmp_path):
    df = data(101)
    metric, config_path = "revenue_capture_at_20", None
    if case == "no_revenue":
        df["mrr"] = 0
    elif case in ("single_class", "small_sample"):
        metric = "auc_roc_churn"
        df["churned"] = 0 if case == "single_class" else np.arange(len(df)) % 2
        if case == "small_sample":
            df = df.iloc[:100]
    elif case == "bad_spec_path":
        config_path = str(tmp_path / "missing.json")
    elif case == "empty":
        df = df.iloc[:0]
    elif case == "no_features":
        df = df.drop(columns="activity")
    path, output = tmp_path / "input.parquet", tmp_path / "output"
    df.to_parquet(path)
    with pytest.raises((ValueError, FileNotFoundError)):
        run_autoresearch(2, metric=metric, ground_truth_path=str(path), spec_path=config_path,
                         output_dir=str(output))
    assert not output.exists()


def test_review_rejects_mixed_evaluators(tmp_path):
    path = tmp_path / "journal.jsonl"
    path.write_text('\n'.join(json.dumps({"metric": "revenue_capture_at_20", "evaluation_version": v}) for v in (1, 2)))
    with pytest.raises(ValueError, match="mixes"):
        review(str(path))


@pytest.mark.parametrize("weight", [0.5, 1.0, True, 26])
def test_invalid_weight_contract(weight):
    config = spec()
    config["features"]["activity"]["weight"] = weight
    with pytest.raises(ValueError, match="integer weight"):
        validate_inputs(data(), config, "revenue_capture_at_20")


def test_nullable_numeric_missingness_names_column():
    df = data()
    df["activity"] = pd.Series([pd.NA] + [1] * 99, dtype="Int64")
    with pytest.raises(ValueError, match="activity"):
        validate_inputs(df, spec(), "revenue_capture_at_20")


def test_review_keeps_baseline_when_all_candidates_lose(tmp_path, capsys):
    path = tmp_path / "journal.jsonl"
    path.write_text(json.dumps({"metric": "auc_roc_retain", "evaluation_version": 2,
                               "baseline_value": 0.8, "variant_value": 0.7, "outcome": "lose"}))
    review(str(path))
    output = capsys.readouterr().out
    assert "keep the input spec" in output
    assert "Best weights" not in output
