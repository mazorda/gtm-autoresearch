# Should we replace our customer score?

Synthetic example. Reproduce with the commands in [the walkthrough](README.md).

**Decision: KEEP_INCUMBENT**

Keep the incumbent.

This is an observational comparison, not proof of causal revenue lift or statistical significance.

## What changed

| Feature | Incumbent weight | Candidate weight |
|---|---:|---:|
| core_workflow | 20 | 17 |
| legacy_clicks | 0 | 1 |
| core_workflow_unknown | 0 | 0 |

Feature definitions, outcome mappings and tier boundaries were frozen. Only weights changed.

## Does it hold up on later outcomes?

| Dataset | Incumbent | Candidate | Change |
|---|---:|---:|---:|
| Search (used for optimization) | 0.7407 | 0.9144 | +0.1737 |
| Later holdout | 0.7376 | 0.5860 | -0.1516 |

Primary metric: `auc_roc_retain`. Minimum gain: 0.0100.
The holdout was evaluated after the candidate was frozen. Do not use it for another tuning cycle.

## Review gates

- Holdout primary gain -0.1516 does not meet a positive improvement of at least 0.0100.
- Tier movement 29.6% exceeds the configured 25.0% limit.
- Segment 'sales_assisted' deteriorates by 0.1581, exceeding 0.0200.
- Segment 'self_serve' deteriorates by 0.1457, exceeding 0.0200.

## Which accounts moved?

29.6% of holdout accounts changed tier. Limit: 25.0%.
Account-level scores and weight-change contributions are in `account_changes.csv` (local, potentially sensitive).
Contributions explain arithmetic before score clipping; they do not establish causality.

| From tier | To tier | Accounts |
|---|---|---:|
| A | A | 150 |
| A | B | 296 |
| D | D | 554 |

## Segment checks

| Segment | Accounts | Incumbent | Candidate | Change |
|---|---:|---:|---:|---:|
| sales_assisted | 506 | 0.7252 | 0.5671 | -0.1581 |
| self_serve | 494 | 0.7491 | 0.6034 | -0.1457 |

## Data comparison

Different time periods/cohorts can have different populations. These differences are diagnostics, not model effects.
Strict validation rejects missing numeric features/outcomes; reported zeroes may still be upstream imputations.

| Feature | Search mean | Holdout mean | Search zero % | Holdout zero % |
|---|---:|---:|---:|---:|
| core_workflow | 0.456 | 0.446 | 54.4% | 55.4% |
| legacy_clicks | 0.491 | 0.522 | 50.9% | 47.8% |
| core_workflow_unknown | 0.061 | 0.062 | 93.9% | 93.8% |

Source freshness and instrumentation coverage cannot be inferred from feature values. Verify these upstream.
Declared feature/outcome windows were checked for temporal ordering and maturity; this cannot prove correct upstream construction.

## Other holdout metrics

| Metric | Incumbent | Candidate |
|---|---|---|
| auc_roc_retain | 0.7376 | 0.5860 |
| auc_roc_churn | 0.7376 | 0.5860 |
| revenue_capture_at_20 | 0.2031 | 0.2053 |
| tier_separation | 1.8937 | 1.8937 |

## Reproduce and review

`manifest.json` records input, spec and code SHA-256 fingerprints, dependencies, seed, policy and time windows.
`report.json` contains the complete metrics, drift diagnostics and decision. No production configuration is changed.
