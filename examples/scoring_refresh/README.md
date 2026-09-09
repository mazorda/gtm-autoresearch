# Should we replace our customer health score?

This walkthrough shows why a better historical score can be a worse update.
All data is synthetic. The behavior shift is deliberately planted for teaching;
the numbers are not a benchmark or client result.

## Run it

From the repository root, after installing `requirements.txt`:

```bash
python3 examples/scoring_refresh/generate_data.py --output-dir runs/refresh-data
python3 analysis/refresh.py \
  --train runs/refresh-data/search.parquet \
  --holdout runs/refresh-data/holdout.parquet \
  --spec runs/refresh-data/incumbent.json \
  --output-dir runs/refresh-review \
  --as-of 2026-02-02 --metric auc_roc_retain \
  --n 200 --seed 42 --segment-column segment
```

Open `runs/refresh-review/report.md`. Or read the committed
[sample report](sample-report.md) without installing anything.
Use new directories when rerunning: existing review records are not overwritten.

## The business question

We have a health score based on core workflow usage. Should we add weight to an
older click signal? Historically those clicks are strongly associated with retention.
In the later period, after a simulated product change, the association reverses.

The optimizer sees only the historical cohort. It finds a candidate that improves
historical retention AUC from **0.7407 to 0.9144**. Once frozen and evaluated against
later outcomes, that candidate falls from the incumbent's **0.7376 to 0.5860**.
It also moves **29.6%** of accounts between tiers, exceeding the demo's 25% limit.
The decision is **KEEP_INCUMBENT**. Exact values refer to the committed sample run;
versions are recorded in each generated manifest.

The small new weight mostly changes the ordering of accounts whose original scores
were tied. A small weight change can therefore have a large ranking effect.

## What the data teaches

- **Revenue units:** raw data mixes annual and monthly recurring amounts. The explicit
  preparation step divides by billing months before supplying `mrr` to the engine.
- **Missing observations:** raw core-workflow values include unknowns. This demo fills
  their score contribution with zero and retains `core_workflow_unknown`, whose weight
  is fixed at zero. This is a declared policy, not evidence of inactivity. In real
  work, excluding insufficiently observed accounts may be a better policy.
- **Ties:** features are discrete; identical scores occur frequently.
- **Time separation:** historical features are from January 2025 and outcomes mature
  in July. Holdout features are from August and outcomes mature in February 2026.
- **Segment tradeoffs:** self-serve and sales-assisted customers are checked separately.

The date columns declare when features/outcomes were observed. They cannot prove
that upstream feature construction avoided future information.

## What to do with a result

`READY_FOR_REVIEW` means the candidate passed your configured business gates, not
that it is statistically significant or automatically approved. Review the metric,
tier movement, segment checks and data context before any production change.

`KEEP_INCUMBENT` means at least one gate failed or could not be verified. Preserve
the existing weights, record the finding and investigate the explanation.
Do not keep tuning against this holdout. Once inspected, it is no longer an untouched
final test. Use a later cohort for a new final evaluation.

## Use your own data

Follow the [input contract and output guide](../../docs/scoring-refresh.md). Keep
real account data and `account_changes.csv` local. Share sanitized aggregate findings
with the community using the [pilot guide](../../docs/community-pilot.md).
