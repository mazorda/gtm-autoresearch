# Changelog

## Unreleased — scoring-refresh report

- Added historical search followed by frozen-candidate comparison on later, mature outcomes.
- Added configurable primary-gain, tier-movement and segment-regression review gates.
- Added Markdown/JSON reports, per-account score-change explanations and run manifests.
- Added a synthetic health-score walkthrough demonstrating a historical winner that fails later.
- Added public sample report, input-contract guide and community pilot invitation draft.
- No production promotion, outreach or client-specific data changes.

## Evaluator version 2 — merged in PR #1

### Fixed
- Revenue Capture @20% selects an exact-size bucket, including when scores tie.
- Zero and small integer weights can move during mutation.
- Missing features, malformed numeric data, invalid labels and unavailable primary
  metrics fail before experiment outputs are written.
- Missing revenue no longer becomes invented unit revenue.
- Explicit missing spec paths fail rather than falling back to auto-detection.
- Journal review refuses mixed objectives/evaluator versions and no longer reports
  a losing candidate as an improvement over the initial baseline.

### Changed
- Replaced misleading NRR/GRR outputs with `revenue_weighted_retention_by_tier`.
- Documentation lists implemented metrics and labels in-sample evaluation limits.
- Removed unrepeatable historical benchmark claims from the README; rerun using
  evaluator version 2 before making new performance claims.

### Added
- `--seed` and `--output-dir`; journal seed and evaluator version.
- Explicit `required_columns` validation in JSON specs.
- Synthetic regression coverage and a Python 3.9/3.12 CI workflow.
- Dependency installation files and an implementation roadmap.

### Migration
Recompute historical baselines. Update consumers of the old NRR/GRR keys. Resolve
missing data upstream and use a new output directory per run. Old journal files
are not modified. These changes do not update any client-specific engine or
production weights.
