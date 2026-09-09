# gtm-autoresearch

Reproducible experiments for interpretable GTM scoring models: change feature
weights, evaluate the result, keep improvements, and record every evaluated candidate.

Inspired by [Karpathy's autoresearch](https://github.com/karpathy/autoresearch).
The mutable artifact here is a JSON scoring spec. The default optimizer is integer
weight hill-climbing; no LLM or production-system access is required.

## Should you replace your current score?

The new [scoring-refresh workflow](docs/scoring-refresh.md) searches historical
outcomes, freezes a candidate, and compares it with your incumbent on later outcomes.
It produces a readable report with tier movement, segment checks, account-level
explanations and fingerprints of the data, specs and code.

Start with the [synthetic walkthrough](examples/scoring_refresh/README.md), or read
its [sample report](examples/scoring_refresh/sample-report.md). The example finds a
historical improvement that fails on later outcomes, so the report says to keep the
incumbent. All example data is synthetic.

The report supports review; it does not automatically promote weights. Source
freshness/coverage and statistical uncertainty still need operator review.

## September 2026 update — evaluation reliability

This update brings lessons from real scoring refreshes into the engine: handle tied
scores correctly, stop on broken inputs, and make experiments easier to reproduce.

- **Correct revenue capture:** tied scores no longer inflate the top-20% population.
- **Better weight exploration:** zero and small weights can move during search.
- **Explicit data checks:** missing features, invalid outcomes and malformed values
  stop a run instead of silently changing its meaning.
- **Reproducible runs:** use `--seed` and a separate `--output-dir` for each experiment session.
- **Accurate metric definitions:** revenue-weighted binary retention is labeled
  correctly; true NRR/GRR are not yet implemented.
- **Regression coverage:** synthetic tests cover these failure modes; GitHub Actions
  is configured to test Python 3.9 and 3.12.

**Upgrading:** recompute historical revenue-capture baselines and update consumers
of the old `nrr_by_tier` / `grr_by_tier` fields. See the [migration notes](#migration-evaluator-version-2)
and [changelog](CHANGELOG.md).

## Why this exists

GTM teams regularly adjust health, engagement, and ICP scores without recording
what changed or checking whether the new version performs better. This engine
makes those changes inspectable and reversible through weight snapshots and an
append-only experiment journal.

You provide the features, outcome definitions, and scoring assumptions. The engine
optimizes a fixed feature set against one objective. Campaign execution and production
promotion remain outside the engine.

## Quick start

Requires Python 3.9+.

```bash
python3 -m pip install -r requirements.txt
python3 examples/health_score/generate_data.py
python3 engine/scoring_engine.py \
  --data synthetic_ground_truth.parquet \
  --metric auc_roc_retain --n 2000 --seed 42 \
  --output-dir runs/health-demo
python3 analysis/review.py runs/health-demo/experiment_journal.jsonl
```

Outputs: `experiment_journal.jsonl` and `best_spec.json`. Use a **new output directory
for each run**, especially after changing data or evaluation objectives. The review
command rejects mixed objectives or evaluator versions; it does not yet fingerprint
datasets, so directories are the boundary between datasets.

`--seed` reproduces mutation choices. Replaying requires the same data **in the same
row order**, spec, engine and dependency versions. Timestamps and durations will differ.
The journal records the seed and evaluator version.

## Bring your own data

Supply a parquet file with one row per account, numeric scoring features, and the
outcome needed by your chosen metric. Prefer an explicit JSON spec:

```json
{
  "model": "health_score",
  "max_score": 100,
  "mrr_column": "mrr_normalized",
  "required_columns": ["account_id"],
  "features": {
    "alert_created": {"weight": 5, "min": 0, "max": 25},
    "session_depth": {"weight": 5, "min": 0, "max": 25}
  },
  "tiers": {
    "A": {"min": 60, "max": 101},
    "B": {"min": 35, "max": 60},
    "C": {"min": 15, "max": 35},
    "D": {"min": 0, "max": 15}
  }
}
```

```bash
python3 engine/scoring_engine.py \
  --data your_ground_truth.parquet --spec your_spec.json \
  --metric revenue_capture_at_20 --n 2000 --seed 42 \
  --output-dir runs/customer-refresh-001
```

- All named features and the selected metric's outcome column are required.
- `required_columns` can additionally require identifiers or other upstream fields.
- Features and present outcome columns must contain finite numeric values. Missing
  or malformed values cause an error; they are never silently filled with zero.
- Retention/churn labels must be 0/1. AUC requires more than 100 rows and both classes.
- MRR must be nonnegative and normalized to a common monthly basis upstream. The
  engine cannot infer whether a number is monthly or annual revenue.
- Revenue capture requires positive revenue. Tier separation requires more than 50
  positive-LTV accounts. An unavailable primary metric stops the run before output.
- Outcome columns cannot also be features. Other leakage, such as future revenue
  under a different name, must be checked upstream.
- Weights and bounds are integers. Features should be scaled deliberately; scoring
  is a weighted sum clipped to `[0, max_score]`.

Optional outcome mappings: `retain_column`, `churn_column`, `ltv_column`.
Without a spec, the engine treats every column except `account_id`, `email`, `mrr`,
`retained_6mo`, `churned`, `actual_ltv`, and `tenure_months` as a feature. Use an explicit
spec for real datasets containing dates, metadata, or custom outcome names.

## Implemented metrics

| Metric | Meaning |
|---|---|
| Revenue Capture @20% | Revenue share in exactly `ceil(0.2 × positive-MRR accounts)` selected accounts |
| AUC-ROC, retention/churn | How well score rankings distinguish binary outcomes |
| Revenue-weighted binary retention by tier | Share of supplied MRR belonging to accounts labeled retained; **not NRR or GRR** |
| Logo retention by tier | Mean retention label in each tier |
| Tier separation | Mean positive LTV above the score's 75th percentile divided by that below its 25th percentile |
| Revenue concentration HHI | Concentration of revenue across scoring tiers |
| ARPA by tier | Mean positive MRR in each tier |
| Score and tier distribution | Mean/std of scores and fraction in each tier |

Optimization choices: `revenue_capture_at_20`, `auc_roc_retain`, `auc_roc_churn`,
`tier_separation`. Other metrics are reports, **not enforced promotion gates**.

Revenue-capture ties are broken by input row order, never by revenue. Sort by a
stable account identifier before creating parquet for reproducible tie handling.
The result depends on that tie order; compare against a random baseline when many
accounts tie. Unlike revenue capture, tier-separation percentile buckets include ties.

True NRR/GRR, expansion, contraction, quick ratio, winback, and activation metrics
are not implemented. See [metric definitions](docs/metrics.md).

## Evaluation limits

The current engine searches and reports on the same dataset. A higher score is an
**in-sample improvement**, not evidence of future performance or incremental revenue.
Use features observed before the outcome window and mature outcome labels. Before
promotion, evaluate the frozen candidate on untouched future outcomes using the
[refresh workflow](docs/scoring-refresh.md) or your own external evaluation. Do not tune repeatedly against that final evaluation set.

Data freshness and instrumentation coverage must be checked upstream. An untracked
action is unknown, not automatically zero activity. Keep fit, health, expansion,
and winback objectives separate and choose the target for the operational decision.

## Migration: evaluator version 2

- Revenue capture now selects an exact-size bucket. Previously, ties could select
  the entire population and report 100% capture. Recompute old baselines before comparison.
- `nrr_by_tier` and `grr_by_tier` are replaced by
  `revenue_weighted_retention_by_tier`. Update downstream consumers.
- Invalid data and unavailable objectives now fail instead of silently degrading.
- Zero and small weights can now move under the default mutation strategy.
- Historical performance claims need rerunning under the corrected evaluator.

See [CHANGELOG.md](CHANGELOG.md) and [ROADMAP.md](ROADMAP.md).

## Development

```bash
python3 -m pip install -r requirements-dev.txt
python3 -m pytest tests -q
```

Tests use synthetic data and temporary output directories. CI runs the suite on
Python 3.9 and 3.12.

## Related

- [Methodology](docs/methodology.md)
- [Autonomous GTM Experimentation playbook](https://mazorda.com/playbooks/autonomous-gtm-experimentation)
- [Karpathy's autoresearch](https://github.com/karpathy/autoresearch)

MIT licensed.
