# Scoring-refresh review

`analysis/refresh.py` optimizes an incumbent using historical data, freezes the
candidate, then compares both specs on a separate, strictly later outcome dataset.
It writes a decision report and reproducibility manifest without changing production.

## Input contract

Two parquet files and an explicit JSON scoring spec are required. Both datasets need:

- One row per account, with a unique non-null `account_id` (override via `--id-column`).
- The same numeric scoring features and the chosen metric's outcomes.
- `features_asof`: the timestamp at which the features were available.
- `outcome_start`: strictly after `features_asof`.
- `outcome_end`: at/after `outcome_start`, and no later than explicit `--as-of`.

The latest search outcome must finish **before the earliest holdout feature snapshot**.
Repeated account IDs across periods are permitted: this measures a later snapshot
of existing customers. It does not establish generalization to previously unseen
companies. Within either period duplicates are rejected to avoid accidental fan-out.

Dates describe the operator's data contract. The engine cannot determine whether
a feature secretly includes future information, a source was stale, or an upstream
zero represents missing tracking. Validate these before running. Use aligned outcome
horizons/cohort eligibility for comparable metrics; the tool checks ordering and
maturity, not equivalence of business definitions.

Metadata columns are not scoring features. Include every intended scoring feature
explicitly in the JSON spec. Integer weights, scaling, clipping, outcome mappings and
tier definitions follow the engine's existing contract. The refresh wrapper additionally
requires tier intervals covering the complete score range without gaps or overlaps.
Only weights change; new features or tier definitions require a separate methodology review.

## Command

```bash
python3 analysis/refresh.py \
  --train historical.parquet --holdout later-outcomes.parquet \
  --spec incumbent.json --output-dir runs/review-001 \
  --as-of 2026-09-01 --metric auc_roc_retain \
  --n 500 --seed 42 \
  --min-gain 0.01 --max-tier-movement 0.25 \
  --segment-column motion --max-segment-drop 0.02
```

The optimizer receives only the search file. The candidate is written to disk
before holdout scoring. Input fingerprints are checked again after optimization to
catch changes during the run. Every review needs a new output directory. A failed
run may leave partial search artifacts; do not treat them as a completed review.
The wrapper cannot detect reuse of a holdout across separate invocations.

## Decision rules

- The primary metric must improve strictly and by at least `--min-gain`.
- Holdout tier movement must be no greater than `--max-tier-movement`.
- If `--segment-column` is supplied, every segment's primary metric must be available
  and must not decline by more than `--max-segment-drop`.
- A segment whose metric cannot be computed blocks review readiness. In particular,
  AUC needs more than 100 rows and both classes within each segment.

A violation yields `KEEP_INCUMBENT`; otherwise `READY_FOR_REVIEW`. These are business
rules in metric units, not statistical significance tests. AUC gain 0.01 means one
AUC percentage point. For tier separation, 0.01 means 0.01 ratio units. Choose gates
for your use case before seeing the holdout results.

Without a segment column the report explicitly marks segment safety as unverified.
Other secondary metrics are reported, not automatically gated. READY_FOR_REVIEW
covers only the configured gates, not every aspect of safety or quality.

## Output files

| File | Purpose |
|---|---|
| `report.md` | Human-readable decision, metric comparison, tier movement and segment results |
| `report.json` | Complete evaluations, weight changes, data diagnostics and gate reasons |
| `account_changes.csv` | Per-account scores, tiers and feature contributions to the score change, before clipping |
| `manifest.json` | Run ID, input/spec/code SHA-256 hashes, versions, parameters and declared time windows |
| `incumbent_spec.json` | Starting model |
| `candidate_spec.json` | Frozen candidate, not automatically promoted |
| `search/` | Optimizer journal and best spec from historical data only |

Data drift compares search and holdout feature means, zero rates and missingness.
The strict engine requires missing numeric data to be resolved upstream, so missingness
will normally be zero; missing-observation indicators can preserve that information.
Differences between cohorts are not effects of changing model weights.

The CSV can contain identifying information. Keep real runs local and share only
reviewed, sanitized outputs. Even aggregate reports may contain private segment names.
Fingerprints verify exact file bytes, not semantic equivalence across differently
serialized parquet files. Dependencies are recorded; pin them to replay a run precisely.

## Scope and next improvements

This first report supports temporal validation, tier/segment gates and arithmetic
explanations. Source freshness/coverage contracts, reference-list overlap, uncertainty
intervals, run-history comparison and automatic production promotion are not implemented.

Use the [synthetic walkthrough](../examples/scoring_refresh/README.md) first.
