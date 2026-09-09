# Response to Spendesk GTM Experimentation Gap Analysis

**From:** Yaniv @ Mazorda
**To:** Oren & team
**Date:** April 2026

> Historical response. Claims below about 13 implemented metrics, statistical
> validation and search throughput predate the evaluator-v2 review. For current
> capabilities and limitations, see [metrics.md](metrics.md) and [the README](../README.md).

Oren - this is brilliant. Genuinely one of the most useful pieces of feedback we've had on the repo. Your team clearly knows what they're doing on the ICP side and this surfaced real things we need to fix. Here's our take - what landed, where we'd push back, and what we're changing.

---

## Where Your Analysis Nailed It

### 1. The "who vs what" framing is better than ours

Your line: *"We answer who to target and how to prioritize them. Mazorda's playbook answers how to continuously learn what messaging, timing, and channel converts them."*

That's a sharper articulation than anything in our README. We're updating the repo to use a three-layer diagram inspired by yours - your ICP foundation sits below our engine, the playbook's channel execution layer sits above it. You built the hardest piece. The engine sits in the middle.

### 2. "Static deliverables vs compounding engine" is the core insight

*"We have static deliverables; the playbook describes a compounding learning engine."* This is exactly the problem the repo was built to solve. Your `tiered_accounts.xlsx` is the output of a scoring model that was validated once. The engine turns that into a scoring model that validates itself continuously against real outcomes and logs every attempt.

The README already had the VP-of-CS anecdote about this. Your analysis confirmed it resonates with teams who've actually built the static version and feel the limitation.

### 3. Your practical gap table is honest and useful

The 8-requirement assessment caught something we should fix in our docs: you rated "experiment journal" as Missing, but that's literally the core feature of the repo. The JSONL journal IS the experiment journal your recommendation says to build. This tells us the quick start section isn't doing its job for sophisticated teams - you should have immediately recognized the repo as the answer to your own recommendation.

We're updating the README to lead with "bring your own data" instead of "generate synthetic data" to make the path clearer for teams like yours.

---

## Where We'd Push Back (With Love)

### 1. "The experimentation layer is completely absent" - we'd argue it's there, just unstructured

Your geo-specific models (FR/DE/GB) are experiments. Your regression validation (R²=0.726) is evaluation against ground truth. Your VIF checks are hypothesis testing. What's absent isn't experimentation - it's **memory**. You experiment. You just don't log it in a queryable format where the next experiment can read what the last one learned.

The engine's value isn't "now you can experiment" - it's "now your experiments remember each other." Every weight change becomes a logged, comparable, reversible experiment instead of an undocumented config tweak.

### 2. "API access to GTM channels is missing" - deliberate scope choice

The repo deliberately does NOT do channel execution (no email sending, no ad deployment, no landing page variants). It optimizes the intelligence layer - scoring models. The playbook describes the full architecture including channel APIs, but the repo's scope is intentionally narrow: one mutable artifact (weights), one evaluation harness (13 GTM metrics), one discipline (keep-or-revert with a journal).

Your gap table would be more accurate with separate columns for the repo (scoring engine) vs the playbook (full GTM lab). The repo covers experiment journal + statistical framework. The playbook covers channel APIs + HOTL governance + cross-channel compounding.

### 3. "Revenue-linked feedback loop is missing" - you've got one, it just runs once

Your R²=0.726 against actual revenue IS a feedback loop - you trained the model on real outcomes. The difference between what you have and what the engine adds is **frequency and automation**. You validated once. The engine validates continuously. Both use ground truth. The gap is iteration speed, not architecture.

---

## What We'd Love to See Tested

### Can you run your Spendesk data through the engine?

Your `master_customer_data.csv` has 3,095 enriched accounts with 6 ICP signals. The engine takes a parquet with features + outcomes. The path:

1. Export your data as parquet with columns: your 6 ICP features + `retained_6mo` (or equivalent retention flag) + `mrr` (or revenue)
2. Run: `python3 engine/scoring_engine.py --n 2000 --metric revenue_capture_at_20 --data spendesk_ground_truth.parquet`
3. It runs 2,000 experiments in ~7 seconds
4. Check: does it find weight configurations that capture more revenue in the top 20% than your current regression model?

The interesting question: your regression optimizes for R² (minimizing squared prediction error). Our engine optimizes for operational GTM metrics (tier separation, revenue capture, NRR by tier). These optimize for different things. It's possible the engine finds configurations that are worse at R² but better at practical resource allocation - top-tier accounts that actually retain and expand, not just top-tier accounts that are statistically predicted to have high revenue.

### Does the ABM classifier benefit from journal discipline?

Your `external_signals_predicting_tiers.py` is a classification task, not weight optimization. But the journal pattern still applies - every time you retrain or tune that classifier, logging the features used, thresholds, accuracy, and the specific accounts that change tiers would let you track drift and compare versions over time.

---

## What We're Updating Based on Your Feedback

| Change | Where | Inspired by |
|--------|-------|-------------|
| Three-layer architecture diagram (your ICP foundation → our engine → playbook channel execution) | README.md | Your two-layer diagram was clearer than our explanation |
| "What this repo does NOT do" section | README.md | Your team conflated repo scope with playbook scope |
| Quick start leads with "bring your own data" path | README.md | Sophisticated teams skip synthetic examples |
| Sharper "experiments without memory" framing | README.md | Your geo-specific models ARE experiments - the gap is memory, not experimentation |
| Scope boundaries between repo and playbook | README.md | Your gap table applied playbook requirements to the repo |

---

## The Three-Layer Model

Your gap analysis diagram was so useful we're adding a version of it to the README:

```
┌─────────────────────────────────────────────┐
│  Channel Execution Layer (Playbook)         │
│  Campaign variants, HOTL governance,        │
│  multi-agent orchestration, API execution   │
│  → mazorda.com/playbooks/autonomous-gtm-    │
│    experimentation                          │
├─────────────────────────────────────────────┤
│  Experiment Engine (THIS REPO)              │
│  Weight optimization, experiment journal,   │
│  13 GTM-native metrics, keep-or-revert     │
│  → github.com/mazorda/gtm-autoresearch     │
├─────────────────────────────────────────────┤
│  ICP Foundation (YOUR WORK)                 │
│  Scoring model, enrichment pipeline,        │
│  segmentation, ground truth data            │
│  → Your regression model, Clay/Cargo,       │
│    geo-specific models                      │
└─────────────────────────────────────────────┘
```

The bottom layer is the hardest to build and the most defensible. You have it. The middle layer (the repo) adds experiment discipline. The top layer (the playbook) adds channel execution and cross-channel compounding. Most teams trying to skip to the top layer without the bottom two are the ones writing the Reddit posts about $2K/month AI SDRs that book zero demos.

---

Seriously - thanks for taking the time. It's rare to get feedback this structured from a team that's already built the hard part. Most feedback comes from people who haven't done any of it yet. Yours comes from people who've done the foundation and can see exactly where the next layer connects.

If you run the engine on your data, we'd love to hear what it finds. Especially whether the GTM-native metrics surface things the R² doesn't. That's the bet.

Yaniv

---

*Prepared by Yaniv's AI Wingman @ Mazorda Hub*
