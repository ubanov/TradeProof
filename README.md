# TradeProof

TradeProof is a heavily modified personal fork of
[TauricResearch/TradingAgents](https://github.com/TauricResearch/TradingAgents)
focused on structured multi-agent investment research, adversarial review,
validation, and reproducible model experimentation.

It is experimental, research-oriented, multi-agent financial-analysis
software — not an automated trading system, not a commercial product, and
not a source of guaranteed or infallible financial recommendations. See the
disclaimer under [TradingAgents Framework](#tradingagents-framework), which
applies in full here.

## Origin

TradeProof began as a fork of
[TauricResearch/TradingAgents](https://github.com/TauricResearch/TradingAgents)
(upstream **TradingAgents v0.5.2**) and still shares a meaningful part of its
original package structure and conventions. Since that starting point it has
diverged significantly from upstream — in model routing, research
verification, canonical research state, risk analysis design, observability,
and reproducibility, among other areas (see
["How TradeProof differs from upstream"](#how-tradeproof-differs-from-upstream)
below, and the fuller list in ["Changes in this fork"](#changes-in-this-fork)).

It's published here as a personal project, in case it's useful to others
working on similar ideas.

- Upstream project: [TauricResearch/TradingAgents](https://github.com/TauricResearch/TradingAgents)
- This fork's repository: [https://github.com/ubanov/TradeProof](https://github.com/ubanov/TradeProof)

The original TradingAgents project and its authors ([TauricResearch](https://github.com/TauricResearch))
remain the source of the underlying framework and research this fork builds
on — see [Citation](#citation). This fork's own Python package, CLI command
(`tradingagents`), and `TRADINGAGENTS_*` environment variable prefix are
unchanged from upstream and are not renamed as part of presenting the
project under the TradeProof name.

## Project objective

TradeProof is optimized first for **analysis quality, auditability, and decision
usefulness**, not for minimum latency or inference cost. A thorough analysis may
take minutes if the additional work produces a better-supported and more
internally consistent recommendation.

The intended workflow is to analyse multiple instruments, produce an auditable
recommendation for each one, and preserve enough evidence and reasoning to
explain **what was recommended and why**. Development therefore has two separate
validation goals:

1. **Reasoning quality** — are claims supported, calculations correct,
   disagreements challenged, uncertainty explicit, and recommendations
   internally consistent?
2. **Decision usefulness** — over time, would acting on frozen historical
   recommendations have produced useful risk-adjusted decisions?

The first goal can be evaluated during development with traces, deterministic
checks, adversarial review, validation, and controlled model replay. The second
requires longitudinal out-of-sample evidence: a convincing report can still be
wrong about future market behaviour.

## How TradeProof differs from upstream

A short summary of the architectural areas where this fork has diverged
most from upstream; the full, detailed list is in
["Changes in this fork"](#changes-in-this-fork) below.

- **Externalized prompts and a shared policy** — agent prompts live in text
  files, with a shared `global_policy.txt` of evidence-integrity rules
  applied across agents.
- **Structured Bull/Bear theses and reviews** — independent initial theses,
  explicit structured cross-review rounds, and classified review outcomes.
- **Canonical research state** — structured thesis/trade-plan state tracked
  alongside free text, with best-effort recovery when structured output
  fails.
- **Frozen evidence** for the Risk phase, so reviewers assess a fixed
  snapshot rather than a moving target.
- **Research Verifier** — an independent, deterministic-adjacent
  quality-control pass over the completed Bull/Bear debate.
- **Bounded research repair** — one corrective pass when the Verifier finds
  a high-severity issue, never an open-ended loop.
- **Deterministic manager/risk integrity checks** — LLM-free passes that
  catch unsupported numeric claims the manager or risk discussion
  reintroduced.
- **Redesigned risk analysis** — independent per-reviewer assessments
  followed by one explicit cross-review round, instead of an open debate.
- **Deterministic trade metrics** — risk/reward and ATR-based figures
  computed in code, not asserted by the model.
- **Bounded decision-stage validation** — Research Manager, Trader, and
  Portfolio Manager candidates can be checked for objective contract, numeric,
  relational, and evidence-support errors, with at most one constrained repair
  by the original producing model.
- **LLM tracing / observability** — an opt-in, complete per-call trace
  (`--trace-llm`) for auditing exactly what was sent and received.
- **Best-effort deterministic replay** — a `--seed` option that derives
  stable per-call seeds for providers that support it.
- **Trace-based model benchmark replay** — selected final synthesis calls can
  be replayed against different models using frozen evidence without rerunning
  the trading pipeline or data-fetching tools.
- **Configurable reasoning effort** and **generation/review model roles** —
  independent reasoning-effort and model configuration for the pipeline's
  generation (produce/synthesize/decide) and review (challenge/verify/audit)
  functions, instead of one model/effort setting for the whole pipeline.

## Changes in this fork

### Prompt architecture

- Agent prompts have been externalized from Python code into text files under
  `tradingagents/prompts/`, making them easier to inspect, modify, compare, and
  version independently from the agent implementation.

- A shared `global_policy.txt` has been added and is automatically applied to
  the relevant LLM agents.

- The shared policy includes common numerical and evidence-integrity rules:
  - do not present unsupported numerical claims as facts;
  - verify arithmetic before relying on a numerical result;
  - independently check numerical claims made by other agents;
  - do not invent probabilities, historical frequencies, expected outcomes,
    or other statistics when they are not supported by supplied data;
  - preserve units and distinguish price distance, ATR, volatility, position
    size, notional exposure, and portfolio risk;
  - prefer evidence, calculations, and explicit reasoning over rhetorical
    argument.

### Research debate architecture

- The Bull and Bear researchers now create their **initial theses
  independently** from the Analyst Team reports.

- The initial Bull thesis is generated without seeing the Bear thesis, and the
  initial Bear thesis is generated without seeing the Bull thesis. This is
  intended to reduce first-mover bias and prevent one side from anchoring the
  other's initial interpretation.

- After the independent phase, Bull and Bear enter explicit **cross-review
  rounds** where they can challenge calculations, evidence, assumptions,
  contradictions, and unsupported conclusions in the opposing thesis.

- Reviews explicitly classify disputed points using states such as
  **ACCEPTED**, **REJECTED**, and **UNRESOLVED**.

- Research depth controls the number of Bull/Bear review rounds:
  - **Shallow:** 1 review round;
  - **Medium:** 2 review rounds;
  - **Deep:** 3 review rounds.

- Research review depth is kept separate from Risk Management discussion depth,
  so increasing research depth does not implicitly increase unrelated Risk
  debate.

### Collaborative research reviews

- Bull and Bear retain distinct bullish and bearish perspectives so the same
  evidence is explored from materially different starting points.

- During cross-review they act as complementary researchers with the shared
  objective of improving forecast accuracy, calibration, evidence quality,
  and robustness, rather than trying to win a debate.

- Accepting a supported correction or lowering conviction after valid
  counterevidence is treated as a successful research outcome, not a loss.
  Genuinely unresolved disagreements remain explicit instead of being forced
  into artificial consensus.

### Research Verifier

- An independent **Research Verifier** has been added between the completed
  Bull/Bear debate and the Research Manager.

- The verifier does not decide whether the bullish or bearish investment thesis
  is preferable. Its role is quality control.

- It checks for issues such as:
  - arithmetic errors;
  - unit or dimensional errors;
  - unsupported claims;
  - source/evidence mismatches;
  - contradictions;
  - conclusions that materially overstate the supplied evidence.

- Verification results distinguish between **PASS**, **WARN**, and **FAIL**.

- Material verification failures can trigger a bounded **repair round** before
  the result is passed to the Research Manager.

- The repair process is deliberately bounded so the system cannot enter an
  indefinite verifier/repair loop.

- Repair reuses the same structured review mechanism as a normal cross-review
  round: when it succeeds, the repaired position updates the canonical
  `bull_thesis`/`bear_thesis` (trade-plan levels, conviction) the same way a
  review would, so the Research Manager, the post-manager integrity check, and
  the Trader all see the repaired levels automatically. A level that a repair
  revises or withdraws is tracked the same way a normal review's is, and can
  never again be treated as active verified provenance. If structured output
  is unavailable, the repair still produces free text for semantic context,
  but the structured trade plan is explicitly left unchanged rather than
  guessed at — this is recorded per side (`APPLIED` / `TEXT_ONLY` / `NOT_RUN`)
  and surfaced to the Research Manager and the Trader so neither treats a
  stale structured value as repair-confirmed.

### Research Manager integrity check

- The Research Verifier checks the Bull/Bear research *before* it reaches the
  Research Manager, but the manager itself is a free-text-capable synthesis
  step: real runs showed it could reintroduce a numeric threshold, sizing
  rule, or price level that was never in the verified evidence (e.g. an
  invented MACD-histogram threshold, volume threshold, allocation percentage,
  or yield trigger).

- The Research Manager's prompt is now explicit that its role is **synthesis
  of the verified research, not creation of a new trading system**: it may
  select between, combine, or summarize existing supported interpretations
  and deterministic metrics, but must not invent new price targets, stops,
  thresholds, probabilities, historical frequencies, sizing/allocation
  percentages, or arbitrary confirmation conditions.

- A lightweight, **deterministic, LLM-free check** runs immediately after the
  Research Manager and before the Trader. It builds a pool of every number
  already traceable to the analyst reports, the verified Bull/Bear trade
  plans, or deterministic calculations, then flags any operational
  threshold/sizing claim in the manager's output that isn't traceable to it,
  a trade-plan level withdrawn during review resurfacing, or conviction
  being described as a forecast probability. No LLM call is added, and no
  automatic correction loop is triggered.

- Result is **PASS** or **WARN** with compact finding categories
  (`NEW_UNSUPPORTED_THRESHOLD`, `NEW_UNSUPPORTED_LEVEL`,
  `UNSUPPORTED_SIZING_RULE`, `CONVICTION_AS_PROBABILITY`,
  `WITHDRAWN_VALUE_REUSED`, among others). On WARN, the offending claims are
  not silently rewritten or dropped (the manager's text could change
  meaning); they are passed to the Trader explicitly marked as unsupported,
  rather than as if they were verified input.

### Constrained review rounds

- Normal Bull/Bear review rounds primarily defend, refute, correct, accept, or
  withdraw claims already present in the initial theses, using only:
  - the supplied analyst reports;
  - either side's initial thesis;
  - the shared objective setup tags;
  - deterministic recalculations from numbers already in that evidence.

- Reviews should not introduce a new market fact, source, price level,
  threshold, or catalyst that cannot be derived from that evidence.

- **NEW_DATA_EXCEPTION**: a narrow, explicit exception for when a new datum is
  genuinely necessary to correct a material factual misunderstanding. A review
  that uses one must mark it explicitly, with what the datum is, why it is
  necessary, and where it came from. The Research Verifier can flag an
  unmarked introduction of new data or an unnecessary/poorly justified
  exception.

### Structured Bull/Bear trade hypotheses

- Each initial Bull and Bear thesis commits to a structured trade hypothesis:
  direction, conviction (see below), an explicit shared analysis horizon,
  entry level or range, take-profit target, and stop-loss/invalidation level,
  alongside supporting evidence, key risks, and explicit data gaps.

- Entry, take-profit, and stop-loss are explicit commitments. Later review
  rounds classify each as **KEEP**, **REVISE**, or **WITHDRAW** rather than
  freely replacing them; a REVISE or WITHDRAW must state a reason grounded in
  already-supplied evidence.

- Bear's direction represents a bearish/reduce-risk thesis, consistent with
  the existing Trader semantics (Buy/Hold/Sell) — it is not forced into a
  literal short trade.

### Conviction semantics

- Conviction is a structured 0–100 score with four 0–25 components (evidence
  quality, internal consistency, robustness to challenge, trade-plan
  coherence). The total is always recomputed deterministically from the
  components in code, so it cannot drift from a model's own arithmetic.

- **Conviction is a self-assessed evidence-strength score, NOT a calibrated
  probability of the market outcome**, and the Research Manager is told not to
  select a thesis, or build a mechanical winner rule, solely because its
  conviction score is higher.

- Conviction is tracked across the debate: an initial value, a value after
  each review round, and a required reason whenever it changes materially.
  Lowering conviction after valid counterevidence is treated as a successful,
  well-calibrated research outcome, not a loss.

### Shared research horizon

- Bull and Bear use the same explicit analysis horizon (a shared config value,
  not independently chosen), so one side cannot silently reason over a
  materially different holding window than the other.

### Objective setup tags

- A small, fixed v1 taxonomy of objective market-state tags (RSI bucket, MACD
  sign, MACD histogram sign, price vs. EMA10/SMA20/SMA50, SMA50-vs-SMA200,
  ATR% bucket, volume-vs-average bucket, trend structure) is computed
  deterministically from market data, once per run, and given identically to
  both Bull and Bear.

- Tags are objective buckets, not interpretations: Bull and Bear may read the
  same tags differently, and that disagreement is intentional. The taxonomy
  deliberately excludes semantic/interpretive tags (e.g. "BULL_FLAG",
  "ACCUMULATION").

### Deterministic trade calculations

- Risk, reward, reward/risk ratio, percentage distances, and ATR-multiple
  distances for each side's trade plan are computed in plain code from the
  committed entry/take-profit/stop-loss/ATR values, never by the LLM.

- An entry given as a range uses its midpoint as the reference price for these
  calculations (a documented default rule); missing entry/target/stop values
  degrade the metrics gracefully instead of erroring.

- The Trader now receives this same deterministic trade plan (entry, take
  profit, stop loss, reward/risk, ATR distance) for both sides directly from
  the research team's verified state, and is instructed to prefer those
  levels over inventing its own from the market report alone.

- The Trader preserves a verified entry **range** as a range (`entry_price_low`
  / `entry_price_high`) instead of being forced to collapse it into an
  invented single execution price; a single verified entry stays a single
  price. The deterministic midpoint above may still be cited explicitly as a
  `reference_entry`, but only as a labelled calculation value, never silently
  substituted for the actual entry/entry range.

### Suggested risk unit

- Each side's thesis carries a `SUGGESTED_RISK_UNIT` (`NO_TRADE` / `LOW` /
  `MEDIUM` / `HIGH`), mapped deterministically from its conviction score
  (`<50` → `NO_TRADE`, `50–64` → `LOW`, `65–79` → `MEDIUM`, `80+` → `HIGH`).

- **This is a research-level signal only. It is NOT a portfolio percentage,
  leverage, or position size** — actual position sizing remains the
  Trader/Portfolio Manager's job and is not implemented by this value.

### Non-interactive and batch CLI execution

- Added non-interactive CLI execution so TradingAgents can be run from scripts
  without answering interactive questions.

- Added batch execution for analysing multiple tickers sequentially using a
  shared configuration and fixed analysis date.

- Batch execution isolates ticker failures so one failed analysis does not
  prevent the remaining tickers from being processed.

- Batch runs can generate:
  - the complete report for each ticker;
  - structured per-ticker result data;
  - an aggregate Markdown summary;
  - an aggregate machine-readable summary.

- Aggregate summaries are generated deterministically from completed run
  results rather than by making an additional LLM call.

- `tradingagents batch` only requires `--tickers` (or `--tickers-file`); every
  other option now has a default, resolved in this order — an explicit flag,
  then an environment variable, then a hardcoded fallback:
  - `--date`: today's date (`YYYY-MM-DD`), fixed once when the command
    starts, so a run that crosses midnight keeps using the day it started on;
  - `--output-dir`: `.\test\<date>`;
  - `--language`: `TRADINGAGENTS_OUTPUT_LANGUAGE`, else `English`;
  - `--depth`: `TRADINGAGENTS_DEPTH`, else `medium`;
  - `--analysts`: `TRADINGAGENTS_ANALYSTS`, else all four
    (`market,social,news,fundamentals`).

  Example — analyze three tickers with every other setting defaulted:
  ```bash
  tradingagents batch --tickers NVDA,SPY,BTC-USD
  ```

### Model roles: Generation / Validation / Review

This fork routes every LLM call by **cognitive function**, not by model
size/speed/cost. Three roles:

- **Generation** — produce, extract, structure, synthesize, propose, decide:
  evidence gathering, first-pass analysis, initial thesis generation,
  synthesis, decision construction.
- **Validation** — check contract/format/arithmetic/obvious consistency on a
  candidate response *before* it moves to the next stage. Validator is **not**
  another analyst or investment opinion: it never judges thesis quality, never
  second-guesses a reasonable inference, and never recommends an
  action/rating/disposition. It only flags objectively checkable problems
  (a number that doesn't add up, a stated relationship that contradicts the
  cited values, a unit mix-up, a missing required field, a structured field
  that contradicts the narrative, a value with no support in the inputs).
  See [Validation](#validation) below.
- **Review** — challenge, verify, audit, correct, repair: adversarial review,
  verification, repair, cross-review, intellectual quality of a thesis, buy/
  sell/strategy reasonableness, complex inference and semantic contradictions.

A role's model can be enormous and slow, or small and fast — the names
describe what a call *does* in the pipeline, not how big or fast the model
behind it is. `generation_llm`, `validator_llm`, and `review_llm` can all
point at the same model or three different ones; nothing in the pipeline
depends on which. Validator is commonly a small/cheap model since its checks
are narrow and objective, but nothing requires that — role defines function,
not size.

This split replaced an earlier `quick_think_llm`/`deep_think_llm` pair whose
names described neither function nor, reliably, size/speed any more as the
fork evolved (Research Manager and Portfolio Manager, for instance, used to
run on the "deep" model purely as a synthesis step, not because synthesis
needs the slower model) — see [Legacy naming](#legacy-naming) below.

| Stage | Role |
|---|---|
| Market / Sentiment / News / Fundamentals Analyst | Generation |
| Bull / Bear initial thesis | Generation |
| Bull / Bear review round(s) | Review |
| Research Verifier (structured + JSON fallback) | Review |
| Research repair (Bull / Bear) | Review |
| Research Manager | Generation |
| Trader | Generation |
| Risk: Aggressive / Conservative / Neutral independent initial assessment | Generation |
| Risk: Aggressive / Conservative / Neutral cross-review | Review |
| Portfolio Manager | Generation |

Research Manager, Trader, and Portfolio Manager additionally run their
candidate output through Validator when validation is enabled — see
[Validation](#validation).

Canonical recovery (reconstructing structured state from a free-text
fallback) always inherits the role of the call it is recovering — a Bull
initial-thesis fallback recovers on generation, a Bull review fallback
recovers on review, a repair fallback recovers on review. It is never fixed
to one role globally.

Research Manager and Portfolio Manager run on generation because their job
is to synthesize a debate/discussion that Review has already audited, not to
audit it a second time; Research Verifier and the Risk cross-review run on
review because auditing is exactly their job. This isn't a claim that one
profile is universally better — our own benchmarks observed complementary
strengths between structured generation/synthesis and adversarial
review/auditing, which is what motivated separating the two roles instead of
using one model knob for the whole pipeline.

Example matching the **current experimentally preferred routing** (any
compatible model may still be used, and roles may share the same model):

```bash
# .env
TRADINGAGENTS_GENERATION_LLM=deepseek-v4.1-flash:cloud
TRADINGAGENTS_GENERATION_REASONING_EFFORT=high

TRADINGAGENTS_REVIEW_LLM=qwen3.8-flash-next:125b-a6b-q4_K_M
TRADINGAGENTS_REVIEW_REASONING_EFFORT=medium

# Validation is opt-in; validator_llm is required only when it is enabled.
TRADINGAGENTS_VALIDATION_ENABLED=true
TRADINGAGENTS_VALIDATOR_LLM=gemma4:26b
```

#### Legacy naming

`quick_think_llm`/`deep_think_llm` and `quick_reasoning_effort`/
`deep_reasoning_effort` (env vars, CLI flags, and config keys) still work,
mapped onto the canonical names:

| Legacy | New |
|---|---|
| `quick_think_llm` | `generation_llm` |
| `deep_think_llm` | `review_llm` |
| `quick_reasoning_effort` | `generation_reasoning_effort` |
| `deep_reasoning_effort` | `review_reasoning_effort` |

An explicit new name always wins over its legacy equivalent if both are set
(e.g. `TRADINGAGENTS_GENERATION_LLM` over `TRADINGAGENTS_QUICK_THINK_LLM`);
otherwise the legacy value is used. There is no need to migrate an existing
`.env` — the legacy names are deprecated, not removed.

### Validation

Validator is a lightweight quality-control layer, not another analyst: it
checks a candidate response's contract/format/arithmetic/obvious consistency
*before* that response moves to the next stage, and — only on an objective
FAIL — sends the **same original agent** (same LLM, same role, same reasoning
effort) back to correct ONLY the reported problems. Validator itself never
rewrites content and never produces a BUY/HOLD/SELL/rating/disposition.

```
Agent produces candidate output
        |
deterministic validation (always, cheap -- JSON/schema, arithmetic, known
        |                 entry/stop/unit relationships, structural checks)
optional Validator LLM (validation_enabled only -- narrow, objective checks)
        |
      PASS/WARN? --yes--> next stage
        | FAIL
validation_enabled? --no--> next stage (FAIL recorded, nothing repaired)
        | yes
SAME original agent repairs ONLY the reported findings (max ONE attempt)
        |
final validation status (observability -- never a second repair)
```

**With validation disabled** (the default): deterministic checks still run
and are still recorded in the saved report/`validation_log` for
observability, but a FAIL there is never repaired, never calls the Validator
LLM, and never mutates the candidate response.

**With validation enabled**: deterministic checks plus the optional Validator
LLM, and a FAIL from either one triggers the single bounded repair below.

**Deterministic checks always run** (cheap, no LLM call) when applicable to
the stage: JSON/schema validity, required fields, known trade-metric
arithmetic, entry/stop-loss direction, unit and relational consistency —
reusing the same primitives the Manager/Risk integrity checks already use
(`tradingagents/agents/integrity_shared.py`, `researchers/arithmetic.py`),
never a second implementation of the same calculation.

**Output recovery and false-positive filtering**: a small/local
`validator_llm` frequently answers the right PASS/WARN/FAIL verdict without
managing the structured tool-call shape. Validator recovers from that SAME
attempt's raw text first (direct JSON parse, then a narrow mechanical reading
of unambiguous text like `PASS` or `status: FAIL` plus a couple of short
bullet reasons) — refusing outright on any contradiction or long/ambiguous
prose. Only when that's impossible does it make one SEPARATE call, and that
call is format-repair only (resend the model's own prior answer + schema,
never the candidate/evidence again), so a real verdict is never silently
overturned by a second, independently-judged attempt. Separately, a finding
is mechanically re-checked against the evidence Validator was actually given
before it's finalized — never against the candidate text itself, which is
what's being checked, not its own support: a number the model calls
unsupported but that literally appears in the evidence, or a relational/unit
claim the stage's own deterministic scanner can't reproduce, is dropped as a
false positive rather than triggering an unnecessary repair. Trader and
Portfolio Manager's `evidence_context` mirrors each stage's own allowed
sources (technical market report, portfolio context, etc.) so Validator never
flags a number the agent was legitimately entitled to cite. See
`docs/model-routing.md` §3.3 for the full detail and exact rules.

**The Validator LLM is opt-in** (`validation_enabled`/`--validation`) and,
when enabled, checks only a narrow, objective taxonomy: arithmetic/relational/
unit errors, a value treated as fact with no support, missing required
elements, a format/contract violation, and internal contradictions. It never
judges thesis quality, investment reasoning, or strategy — that remains
Review's job.

**Bounded repair**: at most one repair attempt per validated output, always
by the same LLM/role that produced the candidate (a Trader finding repairs via
`generation_llm`, never via `validator_llm`). The repair prompt is
restrictive: correct ONLY the reported problems, preserve every other claim
and number, introduce no new evidence/levels/recommendations, keep the exact
same contract/format. The result is rechecked once; if still FAIL, it is
recorded as unresolved and the pipeline continues — never a second repair,
never a loop.

**Where it's wired today**: Research Manager, Trader, and Portfolio Manager —
decision-critical/synthesis stages where an objective contract/arithmetic
error has a clear, bounded fix. Research Manager's deterministic layer
reuses the existing [Research Manager integrity check](#research-manager-integrity-check)
as-is; its evidence context mirrors exactly what the Manager's own prompt
shows it (Bull/Bear initial theses, review outcomes, verification summary,
repair outputs, final Bull/Bear positions, deterministic trade metrics) --
**never** the raw analyst reports, which the Manager's own prompt never
shows it either. Validator detects objective synthesis/contract errors the
Manager itself might introduce (an invented price target, a reward/risk
arithmetic mismatch, a self-assessed conviction score stated as a market
probability); it is explicitly **not** a second Research Verifier and does
not re-litigate which side of the Bull/Bear debate was stronger -- that
stays the Research Verifier's and Research Manager's own job. Analyst
reports, Bull/Bear theses and reviews, Research Verifier, and Risk's
independent/cross-review assessments are deliberately **not** validated yet;
extending Validator's deterministic-only or full coverage to those stages is
future work, not a limitation of the central infrastructure
(`tradingagents/agents/validator.py`, `validator_deterministic.py`,
`validation_schemas.py`), which is stage-agnostic.

Configuration:

| Setting | Env var | CLI flag |
|---|---|---|
| Validator model | `TRADINGAGENTS_VALIDATOR_LLM` | `--validator-llm` |
| Validator reasoning effort | `TRADINGAGENTS_VALIDATOR_REASONING_EFFORT` | `--validator-reasoning-effort` |
| Enable validation | `TRADINGAGENTS_VALIDATION_ENABLED` | `--validation` / `--no-validation` |

There is **no silent fallback** to `generation_llm`: if `validation_enabled`
is true and no `validator_llm` is configured, the run fails fast with a clear
configuration error before any execution starts, rather than silently
changing cost/behavior.

Console output stays quiet on success — no line is printed per PASS. A
`[WARN]` line appears only when a stage's validation fails and a repair is
attempted, and again if it still fails after that repair; full detail lives
in the saved report's "Validation" section and, with `--trace-llm`, in the
trace (`llm_role: "validator"`, plus the original agent's repair call still
recorded under its own real role with a `..._validation_repair` call
identity).

### Model overrides and LLM trace mode

Useful when comparing local Ollama models against each other on the exact
same ticker/date/analysts/depth/language.

- **Per-run model overrides** (`run` and `batch`): `--TRADINGAGENTS_GENERATION_LLM`
  and `--TRADINGAGENTS_REVIEW_LLM` (friendly aliases `--generation-llm`/
  `--review-llm` also work; the deprecated `--deep-think-llm`/`--quick-think-llm`
  still work too, see [Legacy naming](#legacy-naming)) override the configured
  models for that invocation only, without touching `.env`. Generation
  inherits review when no generation override/config value is set — passing
  one model flag runs the entire analysis on that model:

  ```bash
  # One model for everything this run:
  tradingagents batch --tickers NVDA --date 2026-09-30 \
    --analysts market,social,news,fundamentals --depth medium --language Spanish \
    --TRADINGAGENTS_REVIEW_LLM=gpt-oss:20b

  # Two distinct models:
  tradingagents batch --tickers NVDA --date 2026-09-30 \
    --TRADINGAGENTS_REVIEW_LLM=qwen3.8-flash-next:125b-a6b-q4_K_M \
    --TRADINGAGENTS_GENERATION_LLM=gpt-oss:20b
  ```

  An existing `.env`/config with a distinct, non-empty generation model keeps
  working unchanged when no CLI override is given — resolution order is:
  explicit CLI generation override → generation follows a CLI review override
  when no generation override was given → configured generation model →
  inherit review. The resolved generation/review models are printed at
  startup (`Generation model: ...` / `Review model: ...`) and recorded in the
  LLM trace manifest below, so both always agree.

- **`--trace-llm`** (`run` and `batch`, off by default): saves a complete,
  auditable trace of every LLM call under `<output-dir>/<ticker>/debug/` —
  `run_manifest.json` (effective models, provider, prompt-file hashes, git
  commit/branch, package version), a snapshot of the prompt files actually in
  effect for the run (`prompts/`, since a long run's prompt files can change
  on disk mid-run), and one `calls/NNN_<stage>[_<attempt>]/` directory per LLM
  call with the exact rendered request (including its `llm_role` —
  `"generation"`/`"review"` — and `call_identity`), the raw response, the
  parsed structured result (or a recorded parse failure — never a fabricated
  one), and timing/token metadata. Credential-shaped configuration keys
  (`api_key`, `authorization`, `token`, `secret`, `password`, ...) are
  redacted; prompt and response text never is. A `trace_summary.json`
  aggregates call counts for quick before/after comparison. Normal console
  output is unaffected beyond one `[DEBUG] LLM trace enabled: <path>` line; a
  trace-write failure degrades to a `[WARN]` and never aborts the analysis.

  ```bash
  tradingagents batch --tickers NVDA --date 2026-09-30 \
    --TRADINGAGENTS_REVIEW_LLM=gpt-oss:20b --trace-llm
  ```

  **Trace output contains full prompts and model outputs.** Treat it as a
  potentially sensitive analysis artifact — credentials are redacted, but the
  analysis content itself is not.

- **`--seed`** (`run` and `batch`, friendly alias for `--TRADINGAGENTS_SEED`):
  best-effort deterministic replay base seed, for benchmarking, debugging, and
  reproducing a specific run — see "Best-effort deterministic replay" under
  [Reproducibility](#reproducibility).

- **Generation vs Review reasoning effort** — `--generation-reasoning-effort` /
  `--review-reasoning-effort` (`run` and `batch`, aliases for
  `TRADINGAGENTS_GENERATION_REASONING_EFFORT` / `TRADINGAGENTS_REVIEW_REASONING_EFFORT`;
  the deprecated `--quick-reasoning-effort`/`--deep-reasoning-effort` still
  work too) set each role's reasoning level independently, so you can compare
  combinations instead of one effort applied everywhere:

  ```bash
  # DeepSeek V4 Pro: high reasoning on generation, lighter on review
  TRADINGAGENTS_GENERATION_REASONING_EFFORT=high TRADINGAGENTS_REVIEW_REASONING_EFFORT=low \
    tradingagents batch --tickers NVDA --TRADINGAGENTS_GENERATION_LLM=deepseek-v4-pro:cloud

  # Qwen3.8 Flash Next
  TRADINGAGENTS_GENERATION_REASONING_EFFORT=medium TRADINGAGENTS_REVIEW_REASONING_EFFORT=xhigh \
    tradingagents batch --tickers NVDA --TRADINGAGENTS_REVIEW_LLM=qwen3.8-flash-next:125b-a6b-q4_K_M
  ```

  Valid levels are model/backend-specific (e.g. Qwen3.8 Flash Next:
  `false`/`low`/`medium`/`xhigh`; DeepSeek V4 Pro: `false`/`low`/`high`/`max`)
  — TradingAgents does not enforce a fixed list; the backend/model is the
  final authority on whether a level is valid. Leaving a value unset sends no
  override at all, so the backend/model's own default applies. Not every
  provider supports `reasoning_effort` — today that's OpenAI's reasoning-tier
  models and Ollama's OpenAI-compatible endpoint; other providers are
  unaffected and never receive an unrecognized parameter. The effective value
  actually sent (or its absence) is visible per call with `--trace-llm`.

### Benchmark replay

**This is a controlled model comparison using frozen traced inputs. It does
not rerun the trading pipeline or execute data-fetching tools.** It reuses the
final synthesis context captured by `--trace-llm`, preserving the same frozen
evidence, stage semantics, and structured-output contract while changing the
model/reasoning configuration.

Two replay input modes are recorded explicitly:

- `replay_input_mode="exact"` — the frozen message history is replayed directly.
  For tool-using analyst stages, the original tool definitions may be exposed
  with `tool_choice="none"` so the model sees the same protocol context, but no
  tool is ever dispatched or executed by the benchmark.
- `replay_input_mode="tool_history_normalized"` — used only when a provider or
  model ignores the no-tool instruction and attempts another tool call. The
  already-recorded tool-call/tool-result history is deterministically converted
  into a non-executable synthesis context, preserving the textual tool results,
  and the retry is sent with no executable tools available. The normalization
  is recorded in metadata.

In both modes the principle is the same: **same frozen evidence, same final
synthesis task, no tool re-execution, different model/reasoning configuration.**
The benchmark therefore answers a narrow question: how do different models
respond to the same already-collected evidence? It does not compare data
retrieval quality, rerun downstream stages, modify canonical state, or declare a
subjective winner automatically.

**Supported stages** (current scope): Market, Sentiment, News, and Fundamentals
Analyst, plus Research Manager. Bull/Bear, Research Verifier, Trader, Risk,
Portfolio Manager, and Validator are not yet supported.

```bash
tradingagents benchmark-models \
  --from-run ./retest/hybrid-paid-v1/01-LLY-300000003 \
  --stages market,sentiment,news,fundamentals,research_manager \
  --model "qwen3.8-flash-next:125b-a6b-q4_K_M@medium" \
  --model "deepseek-v4.1-flash:cloud@high" \
  --model "gemma4:26b" \
  --output-dir ./retest/model-benchmark-v1
```

`--from-run` is the run's own output folder (containing `<ticker>/debug/` from
that run's `--trace-llm`). `--model` is repeatable; a model name may contain `:`
freely — only `@` introduces an optional reasoning effort. `--seed` derives a
stable seed per (stage, model) pair using the same seed-derivation primitive as
the pipeline; omit it to reuse the source call's original seed when available.
`--provider` applies to every model in the benchmark, defaulting to the source
run's configured provider.

The output directory must be new or empty. Reusing a non-empty benchmark
directory fails before any LLM call so `manifest.json`/`summary.json` can never
describe only part of the artifacts present on disk.

Output tree:

```text
<output-dir>/
  manifest.json
  summary.json
  <Stage_Name>/
    <model-spec-safe-name>/
      request.json
      response.txt   (or response.json for a structured result)
      metadata.json
```

`summary.json` records, per (stage, model), the source call identity, model,
reasoning effort, seed, elapsed time, token counts, structured/fallback status,
parse status, output mode, replay input mode, and any error. It also records
aggregate benchmark timing.

During execution the CLI prints timestamped progress for every comparison
(`N/total`, stage, model, reasoning effort, START and FINISH/ERROR), followed by
per-model and overall elapsed times. Long local-model calls therefore remain
observable instead of leaving the terminal silent until the benchmark ends.

### Risk as independent assessment, not a second debate

Real validation runs (GLD, NVDA, SPY) showed the Risk phase behaving like a
competitive debate rather than a review of the Trader's plan: Risk analysts
defended an assigned role, introduced new thresholds, sizing rules, and
historical analogies not present in the evidence, and sometimes spent later
turns correcting claims they had invented themselves. The Risk phase was
redesigned around one principle: **Risk evaluates the Trader's plan; it does
not design a new one.**

- **Independent assessments, then one cross-review round.** The three Risk
  reviewers (still Aggressive/Conservative/Neutral internally, reframed in
  their prompts as Opportunity/Underexposure, Downside/Tail/Execution, and
  Calibration/Consistency reviewers) each independently assess the frozen
  Trader plan first, without seeing each other's output — mirroring the
  Bull/Bear researchers' independent-initial-thesis phase. Exactly one
  cross-review round follows, where each reviewer sees all three independent
  assessments and may correct or withdraw its own claim, but introduces no
  new evidence, thresholds, or levels. This replaces the previous
  round-robin debate scaled by research depth (3/9/15 turns); the Risk phase
  is now always six turns regardless of depth.

- **Structured disposition, not free debate.** Each turn produces a
  structured `RiskStanceAssessment`: a `risk_level`, a `disposition`
  (`KEEP`/`DEFER`/`REDUCE_RISK`/`REJECT_PLAN`), supported points, unresolved
  risks, corrected/withdrawn claims, unsupported claims noticed in another
  reviewer's assessment, and a `plan_change_required` flag — never a new
  entry/stop/target/threshold/sizing field. A reviewer that believes the plan
  needs a change it cannot express with already-verified levels marks
  `plan_change_required` and explains why, instead of inventing the
  replacement.

- **Frozen evidence set.** At the start of Risk, a fixed evidence bundle
  (analyst reports, verified Bull/Bear structured state, the Research
  Verifier's result, the Research Manager's plan with its integrity caveat,
  deterministic setup tags and trade metrics, the Trader's plan, and known
  data gaps) is assembled once; Risk reviewers may only cite this bundle, not
  fetch or introduce new evidence.

- **Deterministic Risk integrity check.** A new LLM-free pass runs after the
  cross-review round and before Portfolio Manager, sharing its numeric/
  sentence-scanning utilities with the Research Manager's integrity check
  (now also wired up to catch a reward/risk ratio inconsistent with the
  deterministic trade metrics, and a simple relational error such as stating
  one known price is below another when it is not). It flags new unsupported
  thresholds, levels, sizing/portfolio-percentage rules, historical claims,
  probabilities, arithmetic/relational/unit errors, and a retired value or a
  level contradicting the Trader's own plan — PASS/WARN only, no repair loop.

- **Portfolio Manager provenance restrictions.** The Portfolio Manager now
  receives the Risk integrity check's PASS/WARN result and findings
  alongside the risk discussion, with an explicit instruction that Risk
  arguments are assessments, not new evidence: it must not promote a number,
  threshold, or claim introduced only during Risk unless it traces to the
  frozen evidence or the Trader's own plan, and must preserve the Trader's
  existing levels when a plan change is needed but no supported replacement
  exists.

- The three Risk reviewers still follow the same structured-output-with-
  free-text-fallback pattern used by the Trader, Research Manager, and
  Portfolio Manager.

- **Portfolio Manager disposition is real structured state, not inferred from
  the rating.** `rating` (Buy…Sell, the investment/directional view) and
  `disposition` (`KEEP`/`DEFER`/`REDUCE_RISK`/`REJECT_PLAN`, what to do about
  the Trader's plan as written) are independent fields the model sets
  directly — a bullish rating can legitimately pair with `DEFER` or
  `REDUCE_RISK` when the execution plan itself isn't ready. The previous
  mechanical rating→disposition mapping has been removed; an unparsed
  disposition is reported as "not recorded," never guessed from the rating.

- Initial validation runs surfaced several Risk integrity false positives,
  since tightened: a canonical replacement value a cent away from a retired
  one is no longer treated as the same retired value; a percentage describing
  a distance/change/share-of-profit near a price-level word is no longer
  flagged as a unit error; and a comparison word (`above`/`below`/`exceeds`)
  without accompanying conditional language (`if`/`when`/`requires`/...) is
  treated as descriptive prose, not a new operational rule.

### Bug fixes

This fork also carries a set of correctness fixes found during an internal
audit, applied on top of upstream behavior:

- A decision whose holding window opened at a zero price (bad vendor tick, a
  halted/delisted name) no longer silently divides to `inf`/`nan`; the memory
  log now leaves it pending instead of storing a corrupted return that would
  have poisoned any later average (e.g. a backtest's mean alpha for that
  rating bucket).
- The Yahoo OHLCV cache is now written atomically (temp file + rename), so two
  graphs running concurrently in the same process can no longer read a
  partially-written cache file for the same symbol.
- `temperature` is no longer forwarded to OpenAI/Azure reasoning-tier models
  (o-series, GPT-5+) when explicitly configured — those models reject a
  non-default value, the same failure mode already handled for
  `reasoning_effort`.
- FRED rate limits and outages (HTTP 429/5xx) are now classified the same way
  every other vendor's are, so the router's existing graceful-degradation path
  applies instead of surfacing a generic HTTP error.
- A single invalid ticker in a batch run (and the aggregate summary generated
  afterward) can no longer abort the rest of the batch; the failure is now
  isolated and reported per-ticker like any other.
- Minor: an Azure OpenAI client was mislabeled as `azureopenai` instead of
  `azure` in warnings; a non-Typer `exit()` call in the interactive ticker
  prompt was replaced with the same clean-exit path used everywhere else in
  the CLI.
- The Sentiment Analyst's `overall_score = 0` (maximally bearish) was
  incorrectly treated as a missing value and silently overwritten with the
  band's midpoint; a genuine zero now survives unchanged, and a
  deterministically inferred score is explicitly marked `INFERRED_FROM_BAND`
  rather than presented as the model's own judgment (`score_source`).
- A run that fails after the risk discussion completes (observed as an
  uncaught `KeyError` building the Portfolio Manager prompt) no longer loses
  the hours of completed work ahead of it: every agent prompt now renders
  safely against missing placeholders instead of raising, and a failure
  partway through the graph is caught, tagged with the furthest pipeline
  stage reached (`failure_stage`), and saved as `partial_report.md` /
  `result.json` with everything completed so far (analyst reports, Bull/Bear
  research, verification, repair, the Research Manager's plan, the Trader's
  plan, and the risk discussion) — clearly banner-marked as an incomplete run,
  never presented as a final recommendation. A batch run's summary links a
  failed ticker's partial report the same way it links a successful one's.
- Rating/action parsing (Trader, Research Manager, Portfolio Manager) now
  recognizes a bolded standalone heading (`**Hold**`, `**Overweight**`) and
  `Rating:`/`Recommendation:` labels alike, instead of only one exact
  `**Label**: Value` shape; an output that still can't be parsed keeps its raw
  text and is marked `unparsed` rather than guessed at.
- The Trader's action and the Research Manager's recommendation now also
  recognize the Spanish labels `Acción`/`Recomendación` (with or without the
  accent), so a Spanish-language run's `**Acción: Buy**` parses the same way
  `**Action**: Buy` does instead of coming back `null`.
- Expected, recoverable structured-output fallback (a provider falling back to
  free text) now prints a concise `[WARN] ...` line on stdout instead of going
  through `logging`'s default stderr path — under PowerShell, stderr from a
  child process renders as a red `NativeCommandError` block, which trained
  users to ignore red output even though the pipeline was continuing
  normally. A genuine failure still surfaces through the existing
  stderr/failure path unchanged.


## Model evaluation and current routing

TradeProof is being developed with **analysis quality as the primary model-selection
criterion**. Latency and inference cost are measured because they matter
operationally, but they are secondary: a slower model is preferred whenever it
produces materially better research or review.

### Models evaluated

The current routing is based on models that were actually exercised in
TradeProof, not only on vendor descriptions or model size. A model that is not
currently selected may still have been tested and rejected relative to the
alternatives available at that point.

| Model | Observed behaviour in TradeProof | Current status |
|---|---|---|
| **Qwen 3.8 Flash Next 125B** | Strong local reasoning and the best observed adversarial-review behaviour. Medium reasoning has been more useful than xhigh; xhigh was much slower without a consistent quality gain. | **Retained for Review** |
| **DeepSeek V4.1 Flash** | Strong first-pass analysis and synthesis. High reasoning has repeatedly produced high-quality Generation outputs; max added substantially more verbosity without a consistent improvement. | **Retained for Generation** |
| **Gemma 4 26B** | Weaker than Qwen/DeepSeek for broad investment analysis, but effective at narrow objective checks and bounded validation. | **Retained for Validation** |
| **DeepSeek V4 Pro** | Strong results at high reasoning. Max reasoning increased verbosity and did not consistently improve the final analysis. | Evaluated; not currently preferred over V4.1 Flash for Generation |
| **GPT-OSS 120B** | Reasonable general reasoning speed, but weaker structured-output behaviour and provenance discipline in the full pipeline. | Evaluated and not selected for the current routing |
| **GPT-OSS 20B** | Very fast and useful for smoke/guardrail testing, but insufficient for maximum-quality investment analysis. | Retained only for fast development/smoke tests |
| **Qwen 3.8 27B dense** | High analytical quality in the observed run, but extremely slow on the tested hardware. | Not practical on the current local hardware/configuration |

These are **project-specific observations**, not general model rankings. They
depend on TradeProof's prompts, architecture, provider/backend, quantization and
hardware. "Not selected" means the model was actually exercised and did not
offer enough advantage for the role being considered; it does not mean the
model is generally poor.

#### Earlier full-pipeline screening results

Before the current role-specific Generation / Review / Validation routing was
settled, a representative NVDA Medium-depth scenario was used to compare several
local models end-to-end. Keeping this compact table is useful because it records
which models were genuinely tested before being dropped or reassigned to a
narrower role. It is **not** the current routing recommendation.

| Model | Runtime | LLM calls | Tool calls | Input tokens | Output tokens | Fallback warnings | Verification | Final decision | What the run taught us |
|---|---:|---:|---:|---:|---:|---:|---|---|---|
| Qwen 3.8 Flash Next 125B A6B | 03:11:00 | 32 | 24 | 460,610 | 135,193 | 8 | WARN, no repair | Hold / REDUCE_RISK | Strongest local quality in that screening round; later retained primarily for Review rather than all stages |
| Gemma 4 26B A4B | 00:34:45 | 42 | 16 | 299,298 | 102,234 | 14 | WARN, no repair | Hold / KEEP | Useful development model; later found better suited to bounded Validation than broad Generation |
| GPT-OSS 120B | 01:30:44 | 56 | 14 | 407,550 | 64,234 | 19 | WARN, 1 repair | Buy / REDUCE_RISK | Good reasoning speed, but structured-output/provenance discipline was weaker than the alternatives |
| GPT-OSS 20B | 00:15:51 | 52 | 16 | 338,687 | 57,220 | 17 | WARN, no repair | Buy / REDUCE_RISK | Fast enough for smoke tests; quality was not sufficient for final investment analysis |
| Qwen 3.8 27B dense | 14:13:15 | 36 | 24 | 542,922 | 139,287 | 9 | FAIL after repair | Overweight / KEEP | Good analysis, but impractical runtime on the tested hardware |

The scenario used NVDA, analysis date `2026-09-30`, all four analysts, Medium
research depth, Spanish output, and the same TradeProof revision/prompts for the
models in that comparison. The table should therefore be read as evidence from
that specific screening exercise, not as a universal benchmark.

### What the experiments showed

Full-pipeline runs across different seeds/reasoning levels and controlled replay
benchmarks have repeatedly shown that **more reasoning is not automatically
better** and that different pipeline stages benefit from different model
characteristics. Real runs have surfaced and repaired issues including:

- incorrect reward/risk calculations and entry/stop arithmetic;
- reintroduction of trade levels withdrawn earlier in the debate;
- contradictions between a manager recommendation and Trader action;
- semantic mistakes such as confusing a take-profit level with a stop;
- unsupported numbers or operational thresholds;
- failure to preserve `UNKNOWN` when the evidence does not justify a stronger
  conclusion.

The bounded Validator has also demonstrated the intended real repair path:

```text
candidate
   |
   v
objective validation
   |
   +-- PASS/WARN --> continue
   |
   +-- FAIL --> one constrained repair by the original model
                         |
                         v
                       recheck
```

Only one repair attempt is allowed. Validator never replaces the investment
reasoning model or makes the BUY/HOLD/SELL decision itself.

### Controlled replay results

Controlled replay comparisons have so far been run on **LLY** and **MSFT** for
Market, Sentiment, News, Fundamentals, and Research Manager. These are useful
engineering experiments, not a statistically complete model benchmark: two
tickers cannot establish that one model is universally superior, and these
tests evaluate reasoning behaviour rather than investment profitability.

The repeated observations have nevertheless been useful:

- **Market** — DeepSeek V4.1 Flash high and Qwen 125B medium both produced
  detailed, conditional technical analysis. DeepSeek was particularly strong as
  a first-pass generator. Gemma was shallower and made objective indicator
  interpretation errors in controlled replay.
- **Fundamentals** — Qwen and DeepSeek were both strong. Qwen tended to be more
  selective; DeepSeek extracted and derived more from the supplied financial
  data, which can be useful but also creates more surface area for weakly
  supported derived metrics. Gemma lost substantially more detail.
- **Sentiment** — Qwen and DeepSeek generally preserved the important
  distinction between `NO_DATA` and evidence of neutral sentiment. The larger
  limitation is currently data acquisition, not model reasoning.
- **News** — Qwen and DeepSeek generally separated macro information from
  ticker-specific news and avoided inventing company catalysts when evidence
  was unavailable. Company-specific historical news coverage remains weaker
  than macro coverage in some runs.
- **Research Manager** — Qwen and DeepSeek both combined conflicting Bull/Bear
  arguments, deterministic trade metrics, verification results, and uncertainty
  coherently. Qwen's strongest observed contribution has been critical,
  adversarial reasoning; DeepSeek's has been comprehensive first-pass synthesis.

Concrete controlled-replay errors also helped distinguish roles. Examples from
the observed runs include Gemma describing a positive 10Y–2Y spread as an
inverted curve, stating that a MACD value of 7.38 was above a 7.45 signal line,
and sometimes wording source-unavailability as if it proved that no sentiment
posts existed. Those weaknesses as a broad analyst do not prevent Gemma from
being effective as a narrowly-scoped Validator.

### Current preferred routing

Based on the experiments so far, the current preferred experimental routing is:

```text
GENERATION
    DeepSeek V4.1 Flash / high
          |
          v
first-pass analysts, initial theses,
Research Manager, Trader, Portfolio Manager

REVIEW
    Qwen 125B / medium
          |
          v
adversarial debate, Research Verifier,
research repair and Risk cross-review

VALIDATION
    Gemma 4 26B
          |
          v
bounded objective checks and one-shot repair
```

This routing is **not chosen because it is the fastest**. Qwen is kept where its
slower, more critical reasoning has shown the most value; DeepSeek is used for
generation because controlled comparisons have not shown a quality advantage
large enough to justify running Qwen for every first-pass stage; Gemma is kept
in the intentionally narrow Validator role where objective checking matters more
than broad investment synthesis.

One controlled MSFT replay illustrates the operational difference:

| Model | Total measured time |
|---|---:|
| Qwen 125B medium | 53m 07s |
| Gemma 4 26B | 6m 26s |
| DeepSeek V4.1 Flash high | 2m 34s |

These timings are hardware/provider-specific and are not the selection target.
If a slower configuration later proves materially better for decision quality,
TradeProof should prefer the slower configuration.

### Cost observations

Observed cloud spend has so far been small relative to the volume of analysis:

- eight complete hybrid runs using DeepSeek for generation and local models for
  review/validation consumed approximately **$0.78** of cloud credit;
- one controlled replay benchmark using DeepSeek alongside the two local models
  consumed approximately **$0.03** of cloud credit.

These are empirical development observations, not guaranteed provider pricing.
Cost is tracked for operational awareness rather than used as the primary
model-selection criterion.

### What these benchmarks do not prove

These experiments evaluate the **quality and behaviour of the analysis
pipeline**. They do **not** demonstrate that TradeProof recommendations are
profitable. A coherent, internally consistent, well-supported thesis can still
be wrong about future market behaviour.

The next major validation step is longitudinal:

1. analyse multiple symbols using a frozen TradeProof version;
2. persist the recommendation, thesis, levels, evidence, and version identity;
3. never rewrite historical recommendations after the fact;
4. observe subsequent market behaviour;
5. evaluate whether the recommendations, risk controls, and relative rankings
   would have produced useful decisions.

> **First make the reasoning auditable and reliable; then measure whether the
> advice itself has predictive or economic value.**

## News/Sentiment acquisition

An audit of how News and Sentiment data is actually fetched (see
`tradingagents/dataflows/source_coverage.py`) found that free sources like
Yahoo news search, StockTwits, and Reddit have no real historical query
capability — they always serve their current/latest items, which a
historical run then filters to its own date window. Every run now records a
compact `source_coverage` snapshot (no payloads, no secrets) classifying each
source into one of ten states (`SUCCESS_WITH_DATA`, `SUCCESS_EMPTY`,
`UNSUPPORTED_HISTORICAL`, `AUTH_MISSING`, `RATE_LIMITED`, `PROVIDER_ERROR`,
`TIMEOUT`, `FILTERED_TO_ZERO`, `NOT_CONFIGURED`, `SCREENED_TO_ZERO`), written
to `source_coverage.json` alongside the report tree and into `result.json`.

Two states are kept deliberately distinct downstream: **`NO_DATA`** means
every sentiment source legitimately had nothing to show (point-in-time
unsupported, a genuinely empty window, screened out, or not configured) — a
real verdict. **`SOURCE_UNAVAILABLE`** means retrieval itself failed
(timeout/rate-limit/auth/provider error) and says nothing about whether
sentiment existed; it must never be read as "no sentiment observed."

When no sentiment source has usable content, the Sentiment Analyst
**short-circuits**: it builds a deterministic report (`data_status=NO_DATA` or
`SOURCE_UNAVAILABLE`) without calling the LLM at all, rather than spending a
call on three placeholder strings. The rendered report always leads with
`Sentiment data status` in that case — **a Neutral band/5.0 score is never
presented as evidence of neutral sentiment**; it is a schema-required
placeholder, clearly labeled as one.

### SEC EDGAR company events

The News Analyst can also call `get_company_events`, which reads SEC EDGAR's
filing index (`data.sec.gov/submissions/...`) for a ticker and returns its
**Form 8-K** filings in a trailing window (30 days by default) — this first
version covers 8-K only, not Form 4, 10-Q/10-K content, or 6-K. Point-in-time
is gated strictly on `filingDate` (never the period a filing reports on), so
a filing filed after the analysis date can never reach the agent. Each 8-K is
categorized deterministically from SEC's own Item codes (e.g. `2.02` =
earnings/results, `5.02` = management/director change) — never an LLM
judgment of materiality.

SEC company events are **regulatory filings, not press/news coverage**: a
company can go a 30-day window with zero 8-Ks, which is a real,
non-suspicious outcome, not a failure. Most 8-Ks are thin "furnished"
wrappers pointing at an exhibit this version does not fetch — when no usable
excerpt can be extracted from the filing's own primary document,
`content_status=WRAPPER_ONLY` says so explicitly rather than hiding the gap.
This is a complement to News, never a substitute for a real news archive.

## Operational scale and local hardware

A Medium-depth analysis with every analyst enabled is not a single LLM call. It
can include analyst reports, independent Bull/Bear theses, multiple cross-review
rounds, Research Verifier, optional research repair/re-verification, Research
Manager, Trader, three independent Risk assessments, three Risk cross-reviews,
Portfolio Manager, Validator calls, and occasional structured-output fallback or
canonical-recovery calls.

That architecture deliberately spends context and tokens to buy auditability,
adversarial review, provenance control, and objective error detection. Token
volume is therefore often a better measure of pipeline work than wall-clock
time, which depends heavily on hardware, quantization, backend, and CPU/GPU
split.

Local development measurements in this README were produced on:

- CPU: AMD Ryzen 5 7600X
- System RAM: 128 GB @ 3800 MT/s
- GPU: NVIDIA GeForce RTX 5060 Ti
- VRAM: 16 GB
- Inference backend: Ollama
- Context length: 131072

The local Qwen and Gemma timings above should therefore be read as measurements
of this environment, not general model benchmarks. Cloud-provider timings are
similarly provider- and load-dependent.

## Planned experiments

The following ideas are planned or under evaluation and should not be
considered part of the implemented architecture until they are completed.

### Position sizing

- Convert the research-level `SUGGESTED_RISK_UNIT` (implemented) and trade
  quality into an actual position size, using deterministic code over entry,
  stop, volatility (ATR), and portfolio-risk constraints (e.g. max risk per
  trade, portfolio equity).

- Position sizing should not directly equate an LLM conviction score with a
  portfolio percentage — conviction only selects the bounded risk unit; sizing
  math is separate and deterministic.

### Rich immutable recommendation tracking

TradingAgents already provides a decision log and a backtesting facility (see
[Persistence and Recovery](#persistence-and-recovery) and
[Evaluating decisions over time](#evaluating-decisions-over-time)). The planned
TradeProof extension goes further: preserve the **complete structured decision
state** needed to reconstruct what the system recommended and why.

Planned fields include:

- Bull and Bear initial/final theses and conviction trails;
- Research Manager recommendation;
- Trader plan and final Portfolio Manager rating/disposition;
- entry range/reference, stop, target, horizon, deterministic trade metrics;
- key evidence/provenance and known data gaps;
- TradeProof version/commit, prompt hashes, model routing, and a report hash.

The historical record should be immutable after creation. Subsequent market data
may append outcomes, but must never rewrite the original recommendation. Useful
measurements may include entry triggered/not triggered, target/stop reached,
expiry at horizon, realised return, alpha against an appropriate benchmark,
maximum favourable/adverse excursion, realised R, and time to outcome.

### Historical feedback

Once enough immutable outcomes exist, build deterministic historical-performance
summaries for future analyses. Any such feedback must avoid look-ahead leakage
and should be derived primarily from structured market variables and recorded
outcomes rather than free-form retrospective narratives.

Potential summaries include:

- overall results and benchmark-relative results;
- results by rating/disposition and conviction range;
- results for similar technical/fundamental conditions;
- recurring failure patterns and recurring strengths.

The initial Bull/Bear researchers may eventually receive carefully bounded
historical-performance context. Normal review agents should not receive extra
information that was unavailable to the original thesis unless the experiment
explicitly intends to test that change. Past results must remain context, never
a guarantee of future performance.
---


<p align="center">
  <img src="assets/TauricResearch.png" style="width: 60%; height: auto;">
</p>

<div align="center" style="line-height: 1;">
  <a href="https://arxiv.org/abs/2412.20138" target="_blank"><img alt="arXiv" src="https://img.shields.io/badge/arXiv-2412.20138-B31B1B?logo=arxiv"/></a>
  <a href="https://discord.com/invite/hk9PGKShPK" target="_blank"><img alt="Discord" src="https://img.shields.io/badge/Discord-TradingResearch-7289da?logo=discord&logoColor=white&color=7289da"/></a>
  <a href="https://x.com/TauricResearch" target="_blank"><img alt="X Follow" src="https://img.shields.io/badge/X-TauricResearch-white?logo=x&logoColor=white"/></a>
  <a href="https://github.com/TauricResearch/" target="_blank"><img alt="Community" src="https://img.shields.io/badge/GitHub_Community-TauricResearch-14C290?logo=discourse"/></a>
</div>
<br>
<div align="center">
  <a href="https://github.com/TauricResearch" target="_blank"><img alt="TradingAgents #1 Repository of the Day" src="https://trendshift.io/api/badge/repositories/16192" width="250" height="55"/></a>
</div>
<br>
<div align="center">
  <!-- Keep these links. Translations will automatically update with the README. -->
  <a href="https://www.readme-i18n.com/TauricResearch/TradingAgents?lang=de">Deutsch</a> | 
  <a href="https://www.readme-i18n.com/TauricResearch/TradingAgents?lang=es">Español</a> | 
  <a href="https://www.readme-i18n.com/TauricResearch/TradingAgents?lang=fr">français</a> | 
  <a href="https://www.readme-i18n.com/TauricResearch/TradingAgents?lang=ja">日本語</a> | 
  <a href="https://www.readme-i18n.com/TauricResearch/TradingAgents?lang=ko">한국어</a> | 
  <a href="https://www.readme-i18n.com/TauricResearch/TradingAgents?lang=pt">Português</a> | 
  <a href="https://www.readme-i18n.com/TauricResearch/TradingAgents?lang=ru">Русский</a> | 
  <a href="https://www.readme-i18n.com/TauricResearch/TradingAgents?lang=zh">中文</a>
</div>

---

## News

<!-- news:start -->
- [2026-09] **TradingAgents v0.5.1** released with a package layout organised by what each module holds (import paths moved), optional Jev screening of social posts, GPT-6 Sol and Luna as the default models, and fixes to run isolation and SEC EDGAR statements.
- [2026-09] **TradingAgents v0.5.0** released with point-in-time integrity across every dated path, SEC EDGAR fundamentals served as filed, backtesting over a ticker and date grid, portfolio-aware runs, and current model lineups across every provider.
- [2026-08] **TradingAgents v0.4.0** released with look-ahead / point-in-time fixes across FRED macro, social sentiment, and the decision-log memory; clearer decision signals; working CLI checkpoint resume; Trader price grounding; and the GPT-5.6 and GLM-5.3 models.

Full release notes are in [CHANGELOG.md](CHANGELOG.md).

<details>
<summary>Earlier news</summary>

- [2026-07] **TradingAgents v0.3.1** released with correctness and stability fixes: Alpha Vantage look-ahead filtering, graph-router crash-safety, graph-shape-aware checkpoint resume, working crypto sentiment sources, a configurable LLM retry budget, Bedrock API-key auth, and Claude Sonnet 5 / Fable 5 support.
- [2026-06] **TradingAgents v0.3.0** released with a verified data-access contract, an expanded provider registry (NVIDIA, Kimi, Groq, Mistral, Bedrock, and any OpenAI-compatible endpoint), FRED and Polymarket data vendors, a current-generation model catalog, and a CI gate.
- [2026-05] **TradingAgents v0.2.5** released with the grounded Sentiment Analyst, GPT-5.5 etc. model coverage, Qwen/GLM/MiniMax dual-region support, `TRADINGAGENTS_*` env-var configurability with API-key auto-detection, remote Ollama support, non-US alpha benchmarks, and ticker path-traversal hardening.
- [2026-04] **TradingAgents v0.2.4** released with structured-output agents (Research Manager, Trader, Portfolio Manager), LangGraph checkpoint resume, persistent decision log, DeepSeek/Qwen/GLM/Azure provider support, Docker, and a Windows UTF-8 encoding fix.
- [2026-03] **TradingAgents v0.2.3** released with multi-language support, GPT-5.4 family models, unified model catalog, backtesting date fidelity, and proxy support.
- [2026-03] **TradingAgents v0.2.2** released with GPT-5.4/Gemini 3.1/Claude 4.6 model coverage, five-tier rating scale, OpenAI Responses API, Anthropic effort control, and cross-platform stability.
- [2026-02] **TradingAgents v0.2.0** released with multi-provider LLM support (GPT-5.x, Gemini 3.x, Claude 4.x, Grok 4.x) and improved system architecture.
- [2026-01] **Trading-R1** [Technical Report](https://arxiv.org/abs/2509.11420) released, with [Terminal](https://github.com/TauricResearch/Trading-R1) expected to land soon.

</details>
<!-- news:end -->

<div align="center">

🚀 [TradingAgents](#tradingagents-framework) | ⚡ [Installation & CLI](#installation-and-cli) | 🎬 [Demo](https://www.youtube.com/watch?v=90gr5lwjIho) | 📦 [Package Usage](#tradingagents-package) | 🤝 [Contributing](#contributing) | 📄 [Citation](#citation)

</div>

> 🎉 **TradingAgents** officially released! We have received numerous inquiries about the work, and we would like to express our thanks for the enthusiasm in our community.
>
> So we decided to fully open-source the framework. Looking forward to building impactful projects with you!

## TradingAgents Framework

*This section describes the original multi-agent framework concept TradeProof is built on. The Analyst/Researcher/Trader/Risk team structure below is still the conceptual skeleton, but most of these teams' internal behavior has since changed substantially in this fork — see ["How TradeProof differs from upstream"](#how-tradeproof-differs-from-upstream).*

TradingAgents is a multi-agent trading framework that mirrors the dynamics of real-world trading firms. By deploying specialized LLM-powered agents: from fundamental analysts, sentiment experts, and technical analysts, to trader, risk management team, the platform collaboratively evaluates market conditions and informs trading decisions. Moreover, these agents engage in dynamic discussions to pinpoint the optimal strategy.

<p align="center">
  <img src="assets/schema.png" style="width: 100%; height: auto;">
</p>

> TradingAgents framework is designed for research purposes. Trading performance may vary based on many factors, including the chosen backbone language models, model temperature, trading periods, the quality of data, and other non-deterministic factors. [It is not intended as financial, investment, or trading advice.](https://tauric.ai/disclaimer/)

Our framework decomposes complex trading tasks into specialized roles.

### Analyst Team
- Fundamentals Analyst: Evaluates company financials and performance metrics, identifying intrinsic values and potential red flags.
- Sentiment Analyst: Aggregates news headlines, StockTwits, and Reddit chatter into a single sentiment read to gauge short-term market mood.
- News Analyst: Monitors global news and macroeconomic indicators, interpreting the impact of events on market conditions.
- Technical Analyst: Utilizes technical indicators (like MACD and RSI) to detect trading patterns and forecast price movements.

<p align="center">
  <img src="assets/analyst.png" width="100%" style="display: inline-block; margin: 0 2%;">
</p>

### Researcher Team
- Comprises both bullish and bearish researchers who critically assess the insights provided by the Analyst Team. Through structured debates, they balance potential gains against inherent risks.

<p align="center">
  <img src="assets/researcher.png" width="70%" style="display: inline-block; margin: 0 2%;">
</p>

### Trader Agent
- Composes reports from the analysts and researchers to make informed trading decisions, determining the timing and magnitude of trades.

<p align="center">
  <img src="assets/trader.png" width="70%" style="display: inline-block; margin: 0 2%;">
</p>

### Risk Management and Portfolio Manager
- Continuously evaluates portfolio risk by assessing market volatility, liquidity, and other risk factors. The risk management team evaluates and adjusts trading strategies, providing assessment reports to the Portfolio Manager for final decision.
- The Portfolio Manager approves/rejects the transaction proposal. If approved, the order will be sent to the simulated exchange and executed.

<p align="center">
  <img src="assets/risk.png" width="70%" style="display: inline-block; margin: 0 2%;">
</p>

## Installation and CLI

### Installation

Clone this fork (TradeProof):
```bash
git clone https://github.com/ubanov/TradeProof.git
cd TradeProof
```

Create a virtual environment in any of your favorite environment managers:
```bash
conda create -n tradingagents python=3.12
conda activate tradingagents
```

Or with [uv](https://docs.astral.sh/uv/):
```bash
uv venv --python 3.12
source .venv/bin/activate
```

Install the package and its dependencies (`uv pip install .` with uv):
```bash
pip install .
```

### Docker

Alternatively, run with Docker:
```bash
cp .env.example .env  # add your API keys
docker compose run --rm tradingagents
```

After updating the repository, rebuild the image with `docker compose build`.

For local models with Ollama:
```bash
docker compose --profile ollama run --rm tradingagents-ollama
```

### Required APIs

TradingAgents supports multiple LLM providers. Set the API key for your chosen provider:

```bash
export OPENAI_API_KEY=...          # OpenAI (GPT)
export GOOGLE_API_KEY=...          # Google (Gemini)
export ANTHROPIC_API_KEY=...       # Anthropic (Claude)
export XAI_API_KEY=...             # xAI (Grok)
export DEEPSEEK_API_KEY=...        # DeepSeek
export DASHSCOPE_API_KEY=...       # Qwen — International (dashscope-intl.aliyuncs.com)
export DASHSCOPE_CN_API_KEY=...    # Qwen — China (dashscope.aliyuncs.com)
export ZHIPU_API_KEY=...           # GLM via Z.AI (international)
export ZHIPU_CN_API_KEY=...        # GLM via BigModel (China, open.bigmodel.cn)
export MINIMAX_API_KEY=...         # MiniMax — Global (api.minimax.io)
export MINIMAX_CN_API_KEY=...      # MiniMax — China (api.minimaxi.com)
export OPENROUTER_API_KEY=...      # OpenRouter
export MISTRAL_API_KEY=...         # Mistral
export MOONSHOT_API_KEY=...        # Kimi (Moonshot)
export GROQ_API_KEY=...            # Groq
export NVIDIA_API_KEY=...          # NVIDIA NIM
export FRED_API_KEY=...            # FRED macro data (free, optional)
export ALPHA_VANTAGE_API_KEY=...   # Alpha Vantage
export TYPESAFE_API_KEY=...        # Jev social-post screening (optional)
```

For Azure OpenAI, copy `.env.enterprise.example` to `.env.enterprise` and fill in your credentials.

For AWS Bedrock, install the extra with `pip install ".[bedrock]"`, set `llm_provider: "bedrock"`, configure AWS credentials (environment variables, `~/.aws/credentials`, or an IAM role) and `AWS_DEFAULT_REGION`, and use a Bedrock model ID, e.g. `us.anthropic.claude-opus-4-8-v1:0`.

For local models, configure Ollama with `llm_provider: "ollama"`. The default endpoint is `http://localhost:11434/v1`; set `OLLAMA_BASE_URL` to point at a remote `ollama-serve`. Pull models with `ollama pull <name>`, and pick "Custom model ID" in the CLI for any model not listed by default.

Ollama has no provider-side output cap, and a weaker/quantized local model can fail to ever emit a stop token: a real run observed a single call consuming an entire 131K-token context window (30+ minutes, no usable output) instead of a normal few-thousand-token response. As a safety guardrail, this fork defaults Ollama's `max_tokens` to `8192` output tokens; set `TRADINGAGENTS_MAX_TOKENS` to override it (for Ollama or any other provider). This is an output-token cap, not the context length (`num_ctx`), which is unaffected.

For any other OpenAI-compatible server (vLLM, LM Studio, llama.cpp, or a custom relay), use `llm_provider: "openai_compatible"` and set the endpoint via `backend_url` (or `TRADINGAGENTS_LLM_BACKEND_URL`), e.g. `http://localhost:8000/v1` for vLLM or `http://localhost:1234/v1` for LM Studio. The model is whatever your server serves. No key is needed for local servers; set `OPENAI_COMPATIBLE_API_KEY` when the endpoint requires one.

With `TYPESAFE_API_KEY` set, the Sentiment Analyst screens StockTwits and Reddit posts with TypeSafe's Jev before reading them. Posts that are not about the company are dropped, and each source opens with a count of the remaining posts by stance: bullish, bearish, neutral, or unclear. Without the key, posts pass through unscreened. `jev-latest` moves with new releases; set `TYPESAFE_DEFAULT_MODEL` to a versioned ID such as `jev-1.13.0` to hold it fixed across runs.

Alternatively, copy `.env.example` to `.env` and fill in your keys:
```bash
cp .env.example .env
```

### CLI Usage

Launch the interactive CLI:
```bash
tradingagents          # installed command
python -m cli.main     # alternative: run directly from source
```
You will see a screen where you can select your desired tickers, analysis date, LLM provider, research depth, and more. Your previous run's answers come back as the defaults, so pressing Enter accepts them. The `TRADINGAGENTS_*` variables in `.env` still skip their step entirely.

### Markets and tickers

TradingAgents works with any market Yahoo Finance covers, using the exchange-suffixed ticker. Company identity and the alpha benchmark resolve automatically per market.

- US: `AAPL`, `SPY`
- Hong Kong: `0700.HK` · Tokyo: `7203.T` · London: `AZN.L`
- India: `RELIANCE.NS`, `.BO` · Canada: `.TO` · Australia: `.AX`
- China A-shares: Shanghai `.SS`, Shenzhen `.SZ` (e.g. `600519.SS` for Kweichow Moutai)
- Crypto: `BTC-USD`, `ETH-USD`

<p align="center">
  <img src="assets/cli/cli_init.png" width="100%" style="display: inline-block; margin: 0 2%;">
</p>

An interface will appear showing results as they load, letting you track the agent's progress as it runs.

<p align="center">
  <img src="assets/cli/cli_news.png" width="100%" style="display: inline-block; margin: 0 2%;">
</p>

<p align="center">
  <img src="assets/cli/cli_transaction.png" width="100%" style="display: inline-block; margin: 0 2%;">
</p>

## TradingAgents Package

### Implementation Details

We built TradingAgents with LangGraph to ensure flexibility and modularity. The framework supports multiple LLM providers: OpenAI, Google, Anthropic, xAI, DeepSeek, Qwen (Alibaba DashScope, international and China endpoints), GLM (Zhipu), MiniMax (global + China), OpenRouter, Ollama for local models, and Azure OpenAI for enterprise.

Agent prompts are stored under `tradingagents/prompts/`. The shared `global_policy.txt` contains minimal instructions applied to all trading agents. `global_policy.example.hardmode.txt` is a stricter reference policy for local experimentation and is not enabled by default.

### Python Usage

To use TradingAgents inside your code, you can import the `tradingagents` module and initialize a `TradingAgentsGraph()` object. The `.propagate()` function will return a decision. You can run `main.py`, here's also a quick example:

```python
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG

ta = TradingAgentsGraph(debug=True, config=DEFAULT_CONFIG.copy())

# forward propagate
_, decision = ta.propagate("NVDA", "2026-09-01")
print(decision)
```

You can also adjust the default configuration to set your own choice of LLMs, debate rounds, etc.

```python
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG

config = DEFAULT_CONFIG.copy()
config["llm_provider"] = "openai"        # e.g. openai, google, anthropic, deepseek, groq, ollama; openai_compatible covers any OpenAI-compatible endpoint (vLLM, LM Studio, llama.cpp, ...)
config["generation_llm"] = "gpt-6-luna"   # see "Model roles: Generation vs Review"
config["review_llm"] = "gpt-6-sol"
config["max_debate_rounds"] = 2

ta = TradingAgentsGraph(debug=True, config=config)
_, decision = ta.propagate("NVDA", "2026-09-01")
print(decision)
```

See `tradingagents/default_config.py` for all configuration options.

### Fundamentals as filed

US company statements can come from SEC EDGAR, which records the date every figure was filed. A run dated in the past then reads the statements exactly as they stood that day: a fiscal year that has ended but has not been filed yet is not served, and a figure restated later still reads as first reported. Apple's 2008 total assets were filed as $39.6B and restated to $36.2B in 2010, so a run dated in between reads $39.6B.

EDGAR needs no account or API key. Add the vendor to the chain:

```python
config["data_vendors"]["fundamental_data"] = "sec_edgar,yfinance"
```

SEC asks callers to identify themselves and refuses requests that carry no contact address, so a default one is sent. Set your own so SEC can reach you rather than the project:

```bash
SEC_EDGAR_USER_AGENT="Your Name your@email.com"
```

It covers companies that file with the SEC, including foreign companies listed in the US. Anything else, such as Hong Kong or A-share listings, falls through to the next vendor in the chain. EDGAR's machine-readable filings begin in 2009, and a fourth quarter is reported as unavailable rather than derived, because filers publish it only inside the annual figure.

### Current holdings

By default the agents do not know what you hold, so their guidance is written for a reader who applies it to their own position. Pass a portfolio to have the trader, the risk analysts and the portfolio manager work against your actual book.

```python
from tradingagents.portfolio import PortfolioContext

portfolio = PortfolioContext.model_validate({
    "cash": 25000.0,
    "currency": "USD",
    "positions": [{"ticker": "NVDA", "quantity": 120, "average_price": 150.0}],
})
_, decision = ta.propagate("NVDA", "2026-09-01", portfolio=portfolio)
```

The CLI takes the same content as a JSON file: `tradingagents --portfolio my_book.json`.

An empty `positions` list means a flat book, which is different from passing nothing. A run without a portfolio is never treated as flat.

## Persistence and Recovery

TradingAgents persists two kinds of state across runs.

### Decision log

The decision log is always on. Each completed run appends its decision to `~/.tradingagents/memory/trading_memory.md`. On the next run for the same ticker, TradingAgents fetches the realised return (raw, and alpha against the instrument's regional benchmark), generates a one-paragraph reflection, and injects the most recent same-ticker decisions plus recent cross-ticker lessons into the Portfolio Manager prompt, so each analysis carries forward what worked and what didn't.

Override the path with `TRADINGAGENTS_MEMORY_LOG_PATH`.

### Checkpoint resume

Checkpoint resume is opt-in via `--checkpoint`. When enabled, LangGraph saves state after each node so a crashed or interrupted run resumes from the last successful step instead of starting over. The run view says whether it resumed a saved run or started fresh. Checkpoints are cleared automatically on successful completion.

Per-ticker SQLite databases live at `~/.tradingagents/cache/checkpoints/<TICKER>.db` (override the base with `TRADINGAGENTS_CACHE_DIR`). Use `--clear-checkpoints` to reset all of them before a run.

```bash
tradingagents --checkpoint           # enable for this run
tradingagents --clear-checkpoints    # reset before running
```

```python
config = DEFAULT_CONFIG.copy()
config["checkpoint_enabled"] = True
ta = TradingAgentsGraph(config=config)
_, decision = ta.propagate("NVDA", "2026-09-01")
```

## Evaluating decisions over time

One run gives one decision, which cannot tell you whether the system decides well. `run_backtest` runs the same pipeline over a grid of tickers and dates, writes to a decision log of its own, and scores the decisions whose holding window has since traded.

```python
from tradingagents.backtest import iter_grid, run_backtest, summarize

dates = iter_grid("2026-06-01", "2026-08-01", every_n_days=7)
result = run_backtest(["NVDA", "AAPL"], dates, config, selected_analysts=["market", "news"])
print(summarize(result).render())
```

From the CLI:

```bash
tradingagents backtest NVDA,AAPL --start 2026-06-01 --end 2026-08-01 --every 7
```

Each cell is scored on realized alpha against the instrument's regional benchmark, grouped by rating. Your own decision log is never written to, and re-running the same grid with `run_id=result.run_id` skips the cells that already ran, so an interrupted sweep continues where it stopped.

## Reproducibility

TradingAgents is LLM-driven, so two runs of the same ticker and date can differ. This is expected for a research tool built on language models, not a defect. The variation comes from a few distinct sources, and it helps to separate them.

Language model sampling is non-deterministic. Even at a fixed temperature, providers do not guarantee byte-identical output across calls, and reasoning models (the default GPT-6 family, and any thinking-mode model) vary the most because their internal reasoning is itself sampled.

Live data moves. News, StockTwits, and Reddit return different content as time passes, so a run today sees different inputs than a run last week even for the same historical trade date. Pin the analysis date to hold the price and indicator window fixed, but the social and news sources still reflect "now".

To reduce variation you can lower the sampling temperature. Set `temperature` in your config (or `TRADINGAGENTS_TEMPERATURE` in `.env`); lower values make models that honor it more repeatable. The current curated models are reasoning-first and largely ignore temperature, so for tighter reproducibility name a non-reasoning model in your config, or in `TRADINGAGENTS_GENERATION_LLM` and `TRADINGAGENTS_REVIEW_LLM` (see "Model roles: Generation vs Review"). Any model ID your provider serves is accepted, whether or not the picker lists it.

```python
config = DEFAULT_CONFIG.copy()
config["llm_provider"] = "openai"
config["temperature"] = 0.0
# Reasoning models ignore temperature. For tighter reproducibility, name a
# non-reasoning model in generation_llm / review_llm.
```

What does not vary anymore: the analyzed company identity is resolved deterministically from the ticker before any agent runs, and the market analyst grounds exact price and indicator claims in a verified data snapshot. Earlier reports of "different companies" or fabricated price levels across runs are addressed by these two mechanisms.

### Best-effort deterministic replay

For benchmarking, debugging, and reproducing a specific run, the non-interactive `run`/`batch` commands accept a `--seed` (or `TRADINGAGENTS_SEED`):

```bash
# --seed 0 (the default) auto-generates a base seed and prints it:
tradingagents batch --tickers NVDA,AMD --seed 0
# Run seed: 184736291

# Reuse that exact base seed to attempt reproducing the run:
tradingagents batch --tickers NVDA,AMD --seed 184736291
# (equivalently: TRADINGAGENTS_SEED=184736291)
```

Internally, that one base seed is never reused as-is for every LLM call; each call derives its own stable seed from the base seed plus its ticker, pipeline stage, and logical call identity (e.g. "Bull Researcher review 2" vs "Research Verifier JSON fallback"), so results do not depend on the order calls happened to run in. A batch resolves exactly one base seed for the whole run, not one per ticker: `batch --tickers NVDA,AMD --seed 12345` and `batch --tickers AMD --seed 12345` derive the same seeds for AMD's own calls either way. The seed is only forwarded to providers with verified support for it (OpenAI, Azure OpenAI, Ollama); other providers keep running exactly as before, with no seed applied. The effective base seed is also saved in `summary.md`/`summary.json`, `result.json`, and the `--trace-llm` debug manifest, so a past run's seed can always be recovered.

**This is best-effort, not a guarantee of bit-for-bit identical output.** Even with the same seed, results can still vary with the model/backend version, quantization, hardware or kernel differences, non-deterministic GPU operations, a provider without seed support, external tool/API data (news, prices) changing between runs, or any change to the code, prompts, or config.

Backtest results are not guaranteed to match any published figure. Returns depend on the model, the temperature, the date range, data quality, and the sampling above. Treat the framework as a research scaffold for studying multi-agent analysis, not as a strategy with a fixed, replicable return.

## Contributing

Contributions are welcome: bug fixes, documentation, and feature ideas; past contributions are credited per release in [`CHANGELOG.md`](CHANGELOG.md).

## Citation

Please reference our work if you find *TradingAgents* provides you with some help :)

```
@misc{xiao2025tradingagentsmultiagentsllmfinancial,
      title={TradingAgents: Multi-Agents LLM Financial Trading Framework}, 
      author={Yijia Xiao and Edward Sun and Di Luo and Wei Wang},
      year={2025},
      eprint={2412.20138},
      archivePrefix={arXiv},
      primaryClass={q-fin.TR},
      url={https://arxiv.org/abs/2412.20138}, 
}
```
