# Methodology

The engine adapts the constrained experiment loop from
[Karpathy's autoresearch](https://github.com/karpathy/autoresearch) to weighted GTM scores.

1. Validate a fixed dataset and explicit scoring spec.
2. Evaluate the starting weights.
3. Mutate one to three integer feature weights within their bounds.
4. Evaluate the candidate using the unchanged evaluator.
5. Append the evaluated candidate and full weight snapshot to the journal.
6. Keep a candidate only when the primary metric strictly improves; repeat.

Mutations use a local seeded RNG when `--seed` is supplied. The integer mutation
radius is at least one, including at zero, so small weights do not become trapped
by rounding. Fixed bounds remain fixed. Unchanged proposals are skipped.

## What the loop establishes

It can find better in-sample weights within a constrained local search. It does not
exhaustively explore the weight space, prove optimality, establish causal revenue
lift, or evaluate generalization to future cohorts. The journal records history;
the current mutation policy does not learn from earlier journal entries.

Weighted sums make each feature's contribution inspectable. Scaling, clipping,
correlated features, and score ties still affect interpretation. Optimized weights
are not causal effects or a universal feature-importance ranking.

## Operator responsibilities

- Define the operational question: fit, health, expansion or winback prioritization.
- Freeze features observed before the outcome window; exclude immature outcomes.
- Normalize recurring-revenue units and document account identity and cohort rules.
- Check data freshness and instrumentation coverage. Resolve unknown values explicitly.
- Use an explicit JSON spec for production-shaped data.
- Run each experiment session in a separate output directory.
- Compare candidates to the incumbent on the same data and evaluator.
- Evaluate the frozen candidate against untouched future outcomes before promotion.
- Inspect tier movement, reference-set overlap and secondary metrics before rollout.

Those last checks are currently external to the loop. See [the roadmap](../ROADMAP.md)
for planned support. See [metrics.md](metrics.md) for implemented definitions and
[the README](../README.md) for usage and evaluator-version migration.
