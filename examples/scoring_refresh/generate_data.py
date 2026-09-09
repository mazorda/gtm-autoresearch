"""Synthetic health-score refresh: historical winner, later-outcome failure.

The correlation shift is planted deliberately; this is a teaching example, not a
benchmark or evidence about any real company's customers.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


SPEC = {
    "model": "customer_health", "version": "incumbent", "max_score": 100,
    "required_columns": ["account_id"],
    "features": {"core_workflow": {"weight": 20, "min": 0, "max": 25},
                 "legacy_clicks": {"weight": 0, "min": 0, "max": 25},
                 "core_workflow_unknown": {"weight": 0, "min": 0, "max": 0}},
    "tiers": {"A": {"min": 18, "max": 101}, "B": {"min": 10, "max": 18},
              "C": {"min": 5, "max": 10}, "D": {"min": 0, "max": 5}},
}


def prepare(raw):
    """Explicit demo policies: distinguish unknown workflow usage and normalize MRR."""
    frame = raw.copy()
    if not frame["billing_months"].isin([1, 12]).all():
        raise ValueError("Demo billing_months must be 1 or 12")
    frame["core_workflow_unknown"] = frame["core_workflow"].isna().astype(int)
    # Unknown usage contributes no points under this demo policy; retain its flag.
    # This does not mean no usage was observed. Real operators must choose a policy.
    frame["core_workflow"] = frame["core_workflow"].fillna(0)
    frame["mrr"] = frame["recurring_amount"] / frame["billing_months"]
    return frame.drop(columns=["recurring_amount", "billing_months"])


def generate(output_dir, n=1000, seed=17):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(seed)
    for period in ("search", "holdout"):
        retained = rng.binomial(1, 0.5, n)
        core = np.where(rng.random(n) < 0.75, retained, 1-retained).astype(float)
        legacy = np.where(rng.random(n) < (0.95 if period == "search" else 0.10), retained, 1-retained)
        core[rng.random(n) < 0.06] = np.nan
        months = rng.choice([1, 12], n)
        monthly = rng.choice([50, 100, 200, 500], n)
        frame = pd.DataFrame({
            "account_id": [f"synthetic_{period}_{i:04d}" for i in range(n)],
            "segment": rng.choice(["self_serve", "sales_assisted"], n),
            "core_workflow": core, "legacy_clicks": legacy.astype(float),
            "recurring_amount": monthly * months, "billing_months": months,
            "retained_6mo": retained, "churned": 1-retained,
            "actual_ltv": monthly * np.where(retained, 12, 3),
            "features_asof": "2025-01-01" if period == "search" else "2025-08-01",
            "outcome_start": "2025-01-02" if period == "search" else "2025-08-02",
            "outcome_end": "2025-07-01" if period == "search" else "2026-02-01",
        })
        frame.to_parquet(output / f"{period}_raw.parquet", index=False)
        prepare(frame).to_parquet(output / f"{period}.parquet", index=False)
    (output / "incumbent.json").write_text(json.dumps(SPEC, indent=2) + "\n")
    print(f"Synthetic raw and prepared datasets written to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    generate(args.output_dir)
