# Roadmap

## 1. Correctness foundation — merged in PR #1

Exact-size revenue capture; integer mutation fix; strict data validation; accurate
metric names/docs; seeds; isolated output directories; regression tests and CI.

## 2. Refresh report and public example — implemented, pending publication

Implemented: temporal outcome checks, search-only optimization, frozen-candidate
comparison, primary/tier/segment gates, per-account arithmetic explanations, feature
drift summaries, run/input/spec/code fingerprints, and a synthetic end-to-end example.
See [the refresh guide](docs/scoring-refresh.md).

Still planned: source freshness and instrumentation-coverage contracts, reference-set
overlap, uncertainty intervals, and comparisons across compatible run histories.
The report marks these limits explicitly and does not promote production weights.

Community next step: three to five volunteer testers using the
[pilot invitation and feedback guide](docs/community-pilot.md).

## 3. Bounded agent workflow — later

A `program.md` that guides history review, bounded hypotheses and candidate reports.
Keep evaluator and promotion rules fixed during a search. Add guided proposals or
broader mutation schedules only after measuring a benefit against seeded random search.

## Maintenance cadence

After a meaningful production refresh, contribute a generalized lesson or synthetic
regression fixture and a changelog entry. Publish no client account data or private
commercial details. Keep fit, health, expansion and winback objectives distinct.
