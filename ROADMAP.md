# Roadmap

## 1. Correctness foundation — implemented locally, unreleased

Exact-size revenue capture; integer mutation fix; strict data validation; accurate
metric names/docs; seeds; isolated output directories; regression tests and CI.

## 2. Reproducible refresh and promotion workflow — next

- Run IDs and fingerprints for dataset, complete spec, evaluator and dependencies.
- Review only compatible runs; retain the initial spec and baseline as first-class artifacts.
- Time-based evaluation with mature outcomes and a final untouched evaluation set.
- A before/after report: primary and secondary metrics, tier transitions, missingness,
  source freshness, instrumentation coverage and independent reference-set overlap.
- Explicit candidate/promotion decisions with configurable business constraints.
  Fixed percentage-point thresholds are not statistical significance tests.
- A sanitized production-learnings guide using synthetic reproductions of schema
  drift, mixed revenue units, incomplete tracking and silent score-definition changes.

## 3. Bounded agent workflow — later

A `program.md` that guides history review, bounded hypotheses and candidate reports.
Keep evaluator and promotion rules fixed during a search. Add guided proposals or
broader mutation schedules only after measuring a benefit against seeded random search.

## Maintenance cadence

After a meaningful production refresh, contribute a generalized lesson or synthetic
regression fixture and a changelog entry. Publish no client account data or private
commercial details. Keep fit, health, expansion and winback objectives distinct.
