# Evaluation metrics — version 2

The engine optimizes one scalar objective and reports other available metrics.
Reported metrics do not act as automatic promotion constraints.

## Optimization objectives

### Revenue Capture @20%

Among accounts with positive MRR, select exactly `ceil(0.2 * n)` accounts by
descending score. Equal scores retain input row order. Divide selected MRR by all
positive MRR. The selected count is also reported. For small samples the ceiling
means the selected fraction can exceed 20%.

Never break ties using the revenue target itself. Use a stable account-ID order
upstream. Ties can still make results order-dependent. With identical revenue and
100 tied accounts, capture is 20%, not the 100% returned by the old implementation.
For heterogeneous revenue, random ordering captures the selected population fraction
in expectation; any one ordering may differ.

Supply nonnegative monthly-normalized recurring revenue. Missing MRR causes revenue
metrics to be omitted and makes revenue-capture optimization fail. This is revenue
concentration at the supplied snapshot, not a forecast or incremental revenue lift.

### AUC-ROC: retention and churn

Rank discrimination for binary outcomes; this is not classification accuracy.
Retention uses scores, churn uses negative scores (higher score means healthier).
Requires more than 100 observations, both 0/1 classes, and scikit-learn.
Undefined primary objectives cause an error before any experiment is written.

### Tier separation

Among positive-LTV accounts, divide mean LTV for scores >= the 75th percentile by
mean LTV for scores <= the 25th percentile. Requires more than 50 positive-LTV
accounts. These percentile buckets include ties and may overlap for flat scores;
a flat score therefore produces separation 1.0. This is distinct from exact-size
revenue-capture selection.

## Reported metrics

### Revenue-weighted binary retention by tier

`sum(mrr for retained accounts in tier) / sum(mrr in tier)`.
Requires the retention label and positive revenue in the dataset. Tiers require
more than 10 rows; zero-revenue tiers return null. This weights binary retention by
the supplied MRR snapshot. It does not measure expansion or contraction and must
not be labeled NRR or GRR. For starting-revenue weighting, supply starting MRR.

### Logo retention by tier

Mean binary retention label in each tier with more than 10 rows.

### Revenue concentration HHI

`sum((100 * tier_revenue / total_revenue) ** 2)` across configured tiers. With K
exhaustive tiers, the minimum is `10000 / K`, and maximum is 10000. Four equally
weighted tiers yield 2500. This measures concentration across tiers, not account
concentration. It is a diagnostic, not proof of metric gaming or an enforced gate.

### ARPA by tier

Mean MRR among positive-MRR accounts in each tier.

### Distribution diagnostics

Score mean and standard deviation, scored count, and `pct_<label>_tier` fractions.
Configure non-overlapping tier intervals covering `[0, max_score]`. Tier bounds
are minimum-inclusive and maximum-exclusive. The engine initializes unmatched rows
to D, so custom tiers must cover the complete scoring range deliberately.

## Not implemented

True NRR and GRR need aligned starting/ending recurring revenue for the same cohort,
including expansion, contraction and churn. Expansion rate, contraction rate, quick
ratio, winback rate, and activation rate are not implemented either. Earlier docs
listed these as available; that claim was incorrect.

## Interpretation and migration

Evaluator version 2 fixes revenue-capture tie handling and removes the misleading
`nrr_by_tier` / `grr_by_tier` keys. Do not compare its outputs directly to old journals.
Re-evaluate incumbent and candidate on the same data and evaluator. Existing journal
files are preserved, never rewritten.

All optimization results are in-sample. Use an untouched final evaluation period
before promotion, and inspect secondary metrics and cohort changes separately.
