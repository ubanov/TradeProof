# Model routing: Generation / Validation / Review

## 1. Purpose

This document describes how TradingAgents routes LLM calls to one of three
roles — **generation**, **validation**, and **review** — and why this fork
moved away from the original `quick_think_llm`/`deep_think_llm` naming. It is
the detailed companion to the README sections "Model roles: Generation /
Validation / Review" and "Validation"; read those first for the short
version, examples, and the legacy-naming table.

## 2. Generation role

Generation calls **produce, extract, structure, synthesize, propose, or
decide**. They create something new from the evidence available at that
point in the pipeline:

- The four analysts (Market, Sentiment, News, Fundamentals) extracting a
  report from raw data.
- Bull/Bear's initial thesis — the first, independent position each side
  takes.
- Research Manager and Portfolio Manager — synthesizing an already-audited
  debate/discussion into a plan or decision.
- Trader — constructing a concrete transaction proposal from the plan.
- Risk's three independent initial assessments — each reviewer's first,
  independent read of the Trader's plan.

## 3. Validation role

Validation calls **check contract/format/arithmetic/obvious consistency** on
a candidate response that generation (or review) already produced, before it
moves to the next stage. Validator is a lightweight quality-control layer,
not another analyst or investment opinion:

- It never judges thesis quality, investment reasoning, buy/sell/strategy
  wisdom, or subjective valuation — that stays Review's job.
- It never produces a BUY/HOLD/SELL/rating/disposition itself.
- It never rewrites content itself: on an objective FAIL, the **same
  original agent** (same LLM, same role, same reasoning effort) repairs its
  own output, bounded to one attempt.

```
Agent produces candidate output
        |
deterministic validation (always, cheap)
        |
optional Validator LLM (validation_enabled only)
        |
      PASS/WARN? --yes--> next stage
        | FAIL
SAME original agent repairs findings (max ONE attempt)
        |
final validation status (observability only -- never a second repair)
```

### 3.1 Deterministic checks (always run, no LLM call)

Applied when relevant to the stage, reusing existing primitives rather than
a second implementation of the same calculation:

- JSON syntax and Pydantic/schema validation (`validator_deterministic.
  check_json_syntax` / `check_schema`).
- Known arithmetic (`validator_deterministic.check_arithmetic_claims`, which
  wraps `researchers/arithmetic.py::check_arithmetic` — the same function the
  Research integrity checks already use).
- Relational and unit consistency in the rendered text
  (`validator_deterministic.check_text_relational_and_unit_errors`, wrapping
  `agents/integrity_shared.py::find_relational_errors`/`find_unit_errors`).
- Entry/stop-loss direction for a structured trade proposal
  (`validator_deterministic.check_entry_stop_direction`): a Buy's stop must be
  below entry, a Sell's above.
- Missing required elements (`validator_deterministic.
  check_missing_requirement`).

### 3.2 Validator LLM taxonomy (opt-in, narrow, objective)

Enabled only when `validation_enabled` is set (with `validator_llm`
configured — there is no silent fallback to `generation_llm`). The taxonomy
(`agents/validation_schemas.py::ValidationCategory`) is deliberately narrower
than the Research Verifier's own, and is not a blind reuse of it — categories
that are truly equivalent (`ARITHMETIC_ERROR`, `RELATIONAL_ERROR`,
`UNIT_ERROR`) share a name, but Validator adds its own contract-shaped
categories:

| Category | Example |
|---|---|
| `ARITHMETIC_ERROR` | A stated total doesn't match its own cited components. |
| `RELATIONAL_ERROR` | "Entry below stop-loss" on a stated Buy. |
| `UNIT_ERROR` | A percentage used where an absolute price was required. |
| `UNKNOWN_USED_AS_FACT` | An unverified number asserted as a known value. |
| `UNSUPPORTED_NUMBER` | A number with no support anywhere in the provided evidence. |
| `MISSING_REQUIREMENT` | A required field/section the contract asked for is absent. |
| `FORMAT_CONTRACT_ERROR` | Output doesn't follow the requested structure/labels. |
| `JSON_SYNTAX_ERROR` | Malformed JSON (only reported once, from the deterministic layer). |
| `SCHEMA_ERROR` | Valid JSON that doesn't match the expected schema. |
| `INTERNAL_CONTRADICTION` | Two parts of the same response assert opposite things. |
| `PROMPT_REQUIREMENT_VIOLATION` | A stated instruction (e.g. "no percentages") is violated. |
| `UNCLASSIFIED_VALIDATION_ERROR` | Limited recovery only (see §3.3) — a FAIL/WARN the model stated unambiguously, without a machine-parseable specific category. |

What Validator explicitly does **not** judge: Bull/Bear thesis intellectual
quality, complex inference reasonableness, buy/sell/strategy wisdom,
subjective valuation, deep economic interpretation, complex `SOURCE_MISMATCH`
or `OVERSTATED_INFERENCE`, or any semantic contradiction that requires
financial debate to resolve — all of that remains Review's responsibility
(the Research Verifier and Risk's cross-review).

### 3.3 Output recovery and false-positive filtering

Real runs with a local model (Gemma4:26b as `validator_llm`) surfaced two
problems the design above didn't yet handle:

**1. The model often answers correctly but not in a parseable shape.** It
reliably decides PASS/WARN/FAIL and even gives short reasons, but fails the
structured tool call — collapsing every such response to the same
conservative `WARN, []` as genuinely unparseable noise, which silently
swallowed real FAILs and meant the repair cycle was never exercised. The
fix has two parts:

- `bind_validator` requests `include_raw=True`, so a failed structured parse
  still exposes the model's raw text from that SAME attempt (real LangChain's
  `{"raw", "parsed", "parsing_error"}` shape) -- the only way to recover from
  it without a second, independently-judged call.
- A **Limited Validator Recovery** step (`_recover_same_attempt_response` /
  `_limited_validator_recovery`) runs on that raw text FIRST, never a new LLM
  call: a direct JSON/schema parse (the model wrote JSON in plain content
  instead of calling the tool), then a mechanical reading of unambiguous
  shapes -- bare `PASS`/`WARN`/`FAIL`, `status: FAIL`, `status = PASS`,
  case-insensitive, and a status line followed by a few short bullet/YAML-
  style reasons (e.g. `FAIL\n- entry price missing`). It is deliberately
  mechanical, never semantic: a bare FAIL/WARN with reasons becomes findings
  tagged `UNCLASSIFIED_VALIDATION_ERROR` (never a guessed specific category);
  it refuses outright on any contradiction (two different status tokens),
  excess length, or a first line that isn't a status token at all. Only when
  this refuses does a SEPARATE call happen, and that call
  (`_validator_normalize_prompt`) is FORMAT REPAIR only -- it resends the
  model's own prior raw response and the schema, explicitly instructed not to
  re-evaluate the candidate or change PASS/WARN/FAIL, never the original
  candidate/evidence/instructions again. (An earlier version of this mechanism
  re-sent the full task on every fallback, which could -- and in real runs
  did -- silently overturn a real FAIL into a fresh PASS verdict on pure
  sampling variance; this is a corrected design, not the original one.)

**2. A false-positive provenance pattern.** The model sometimes flagged a
number as `UNSUPPORTED_NUMBER`, or a field as `UNKNOWN_USED_AS_FACT`, when
that exact value was present in the evidence it was given -- and, separately,
Trader/Portfolio Manager's own `evidence_context` passed to Validator was
narrower than what the agent itself was actually given (missing the
technical market report and portfolio context), so Validator flagged
numbers the agent was legitimately citing. Both are fixed: `evidence_context`
now mirrors each stage's own allowed sources (see `trader.py`/
`portfolio_manager.py`), and before the status is finalized,
`_filter_llm_findings` mechanically re-checks a small set of categories
against that evidence -- NEVER the candidate text itself, which is what's
being checked, not a source of its own support -- and drops a finding only
when it can positively confirm the model was wrong (kept whenever it can't
determine that reliably):

- `UNSUPPORTED_NUMBER` — dropped if the claimed number (tolerating `$` and
  thousands separators: `1167.49` / `1,167.49` / `$1,167.49`) appears
  anywhere in `evidence_context`.
- `UNKNOWN_USED_AS_FACT` — dropped only if the claimed value appears in the
  evidence WITHOUT an unknown-style marker ("unknown", "n/a", "not
  available", ...) nearby; kept if it's genuinely marked unknown there, or
  if it doesn't appear in the evidence at all (can't be confirmed either way).
- `RELATIONAL_ERROR` / `UNIT_ERROR` — dropped only if the stage's own
  deterministic pass found nothing of that category AND re-running the same
  deterministic scanner (`check_text_relational_and_unit_errors`) on the
  LLM's own cited claim text also finds nothing — i.e. the deterministic
  result prevails over a contradicting LLM finding, per §3.1's primitives,
  never a new/duplicate check. `ARITHMETIC_ERROR` is deliberately NOT
  cross-checked this way — no safe deterministic re-verifier exists for free
  text without structured operands, and guessing one was explicitly ruled
  out as over-reaching.

If every finding on a FAIL result gets filtered out, the status is
recomputed to PASS (never an empty FAIL); any remaining finding keeps the
model's own original status unchanged — this filter only removes confirmed
false positives, it is not a second severity model.

Every drop is recorded with its reason under `--trace-llm`
(`validator_findings_raw` / `validator_findings_dropped` /
`validator_findings_final`, plus `structured_parse_success` /
`limited_recovery_success` / `json_fallback_called` / `json_fallback_success`
and the raw response recovery worked from — see §10), never silently
discarded.

### 3.4 PASS / WARN / FAIL and bounded repair

- **PASS** — no findings.
- **WARN** — minor finding(s), no repair needed.
- **FAIL** — an objective problem. Triggers exactly one repair attempt, by
  the same LLM/role that produced the candidate, **only when validation is
  enabled** (`validation_enabled`/`validator_llm` configured). With
  validation disabled, a FAIL is still recorded in `validation_log` for
  observability, but never repaired and never mutates the candidate — see
  the "Backward compatibility" note in §11. Any non-empty deterministic
  finding always forces FAIL, regardless of what the Validator LLM says
  (deterministic findings are, by construction, objective defects).

The repair prompt is restrictive by design: correct ONLY the reported
findings, preserve every valid claim/conclusion/number not involved in a
finding, introduce no new evidence/levels/thresholds/recommendations, return
the response in the exact same contract/format. The repaired output is
rechecked once (same deterministic checks, and the Validator LLM again if
enabled); the result is final either way — there is never a second repair,
even if the recheck still fails. This guarantee is enforced structurally by
`validate_and_repair()` (`agents/validator.py`), the single entry point every
integrated stage calls: it runs the check function once, optionally repairs
once, rechecks once, and returns.

### 3.5 Where validation is (and isn't) wired today

| Stage | Deterministic | Validator LLM |
|---|---|---|
| Trader | Yes (relational/unit + entry/stop direction) | Yes, when enabled |
| Portfolio Manager | Yes (relational/unit) | Yes, when enabled |
| Market / Sentiment / News / Fundamentals Analyst | Not yet wired | Not yet wired |
| Bull / Bear initial thesis and review | Not yet wired | Not yet wired |
| Research Verifier | Not yet wired | Not yet wired |
| Research Manager | Not yet wired | Not yet wired |
| Risk independent / cross-review | Not yet wired | Not yet wired |

Trader and Portfolio Manager were chosen for this first pass as the clearest
decision-critical, structured outputs. The central infrastructure
(`agents/validator.py`, `validator_deterministic.py`, `validation_schemas.
py`) is stage-agnostic and reusable; extending coverage to the remaining
stages — likely deterministic-only for the four analyst reports, given their
free-text shape — is future work, not a limitation of the design.

### 3.6 JSON/schema repair (separate, narrow, not yet wired)

`agents/validator.py::repair_json_syntax()` is a standalone function for the
specific case of a structured call returning invalid JSON/schema: it asks the
Validator LLM to fix syntax/shape only ("preserve every number, claim,
conclusion, and recommendation; do not add or remove substantive content"),
then re-parses deterministically, returning `None` (never raising) if it
can't be repaired safely. It exists as infrastructure around the **existing**
structured/free-text fallback mechanisms in `agents/structured.py` — it does
not replace them, and no stage currently calls it; a stage's existing
fallback path still applies when this returns `None`.

## 4. Review role

Review calls **challenge, verify, audit, correct, or repair** something
that generation already produced:

- Bull/Bear's review rounds — challenging the opponent's position and
  defending or revising one's own.
- Research Verifier — the deterministic-adjacent audit of the completed
  debate (structured attempt and JSON fallback both run on review).
- Research repair — correcting a position the Verifier flagged.
- Risk's cross-review round — each reviewer critiquing the other two.

Validator complements Review; it does not replace the Research Verifier, and
the Verifier's own taxonomy/semantics/thresholds are unchanged by Validator's
introduction.

## 5. Full routing table

| Stage | Role |
|---|---|
| Market Analyst | Generation |
| Sentiment / Social Analyst | Generation |
| News Analyst | Generation |
| Fundamentals Analyst | Generation |
| Bull initial thesis | Generation |
| Bear initial thesis | Generation |
| Bull review round(s) | Review |
| Bear review round(s) | Review |
| Research Verifier (structured) | Review |
| Research Verifier (JSON fallback) | Review |
| Research repair — Bull | Review |
| Research repair — Bear | Review |
| Research Manager (structured + fallback) | Generation |
| Trader (structured + fallback) | Generation, then Validation (when enabled) |
| Risk: Aggressive independent initial assessment | Generation |
| Risk: Conservative independent initial assessment | Generation |
| Risk: Neutral independent initial assessment | Generation |
| Risk: Aggressive cross-review | Review |
| Risk: Conservative cross-review | Review |
| Risk: Neutral cross-review | Review |
| Portfolio Manager (structured + fallback) | Generation, then Validation (when enabled) |
| Validator repair (Trader / Portfolio Manager, on FAIL) | Generation (same role as the call being repaired) |
| Canonical recovery (any fallback) | Inherits the role of the call it recovers |

Canonical recovery never has a role of its own — it reconstructs structured
state from a free-text fallback using the exact same LLM (and therefore the
exact same role) as the call that just fell back. A Bull initial-thesis
fallback's recovery call runs on generation; a Bull review fallback's
recovery call, and a repair fallback's recovery call, run on review. A
Validator repair call always runs on the role of the call it is repairing
(generation for Trader/Portfolio Manager today) — Validator's own calls
(the check itself, and the post-repair recheck) are the only calls that run
on the validator role.

## 6. Why this differs from upstream

Upstream TradingAgents, and this fork until this change, distinguished
`quick_think_llm` (cheap/fast) from `deep_think_llm` (expensive/slow),
implying a size/speed/cost axis. As the fork's architecture grew — a
Research Verifier doing real adversarial auditing, a bounded repair phase,
Risk split into independent assessments plus one cross-review round — the
quick/deep split stopped describing anything functionally consistent.
Research Manager and Portfolio Manager, for example, ran on `deep_think_llm`
purely because they came last in the pipeline, not because synthesis
specifically needs the larger model. Research Verifier ran on
`quick_think_llm`, even though verification is exactly the kind of
adversarial, auditing work the "deep" label suggested it should get.

Renaming to generation/review (and later adding validation as a third role)
makes the split describe **function**, not a guess about which stages
deserve the bigger model. It also reassigns a few stages along the way (see
§5): Research Verifier and Risk's cross-review move onto the review role;
Research Manager and Portfolio Manager move onto generation. A deployment is
free to point generation, validation, and review at the same model
(reproducing the old single-model behavior) or at up to three different ones.

## 7. Legacy naming / compatibility

`quick_think_llm`/`deep_think_llm` and `quick_reasoning_effort`/
`deep_reasoning_effort` (config keys, env vars, and CLI flags) are
deprecated but still fully functional. `resolve_role_config()`
(`tradingagents/default_config.py`) is the single compatibility layer:
given any config dict, it resolves `generation_llm`/`review_llm`/
`generation_reasoning_effort`/`review_reasoning_effort` with precedence
**explicit canonical name > legacy equivalent > hardcoded default**, and
mirrors the result back onto the legacy keys so any code still reading them
sees the same, single, correctly-resolved value. It runs once when
`DEFAULT_CONFIG` is built, again at the top of `execute_analysis()`, and
again in `TradingAgentsGraph.__init__` — idempotent, so a config normalized
upstream is a no-op the second and third time.

There is no legacy name for `validator_llm`/`validator_reasoning_effort`/
`validation_enabled` — they are new keys with no prior equivalent, resolved
by the separate `resolve_validation_config()` (same module), which raises a
`ValueError` if `validation_enabled` is true and `validator_llm` is unset,
rather than guessing a fallback model. It is called at the same three points
as `resolve_role_config()`, plus once more, early, in `cli/headless.py`'s
`run_batch_analysis()` — before the per-ticker loop — so a batch run fails
once with one clear message instead of repeating the same error per ticker.

| Legacy | New |
|---|---|
| `quick_think_llm` | `generation_llm` |
| `deep_think_llm` | `review_llm` |
| `quick_reasoning_effort` | `generation_reasoning_effort` |
| `deep_reasoning_effort` | `review_reasoning_effort` |
| *(none)* | `validator_llm` |
| *(none)* | `validator_reasoning_effort` |
| *(none)* | `validation_enabled` |

## 8. Configuration examples

```bash
# .env — DeepSeek V4 Pro on generation (high reasoning), Qwen3.8 Flash Next on review (medium)
TRADINGAGENTS_GENERATION_LLM=deepseek-v4-pro:cloud
TRADINGAGENTS_GENERATION_REASONING_EFFORT=high
TRADINGAGENTS_REVIEW_LLM=qwen3.8-flash-next:125b-a6b-q4_K_M
TRADINGAGENTS_REVIEW_REASONING_EFFORT=medium
```

```bash
# .env — adding validation on a small, cheap model (opt-in)
TRADINGAGENTS_VALIDATION_ENABLED=true
TRADINGAGENTS_VALIDATOR_LLM=gpt-oss:20b
TRADINGAGENTS_VALIDATOR_REASONING_EFFORT=low
```

```bash
# CLI, per-run override, same model for both roles
tradingagents batch --tickers NVDA --generation-llm gpt-oss:20b
```

```bash
# CLI, two distinct models
tradingagents batch --tickers NVDA \
  --generation-llm gpt-oss:20b --review-llm qwen3.8-flash-next:125b-a6b-q4_K_M
```

```bash
# CLI, validation enabled for this run only
tradingagents batch --tickers NVDA --validation --validator-llm gpt-oss:20b
```

Any model ID the configured provider serves is accepted. Any two (or all
three) roles may be the same model — there is no requirement to use distinct
ones.

## 9. Reasoning effort interaction

`generation_reasoning_effort`/`review_reasoning_effort`/
`validator_reasoning_effort` each set `reasoning_effort` independently per
role, forwarded only to providers with verified support (OpenAI's
reasoning-tier models, and Ollama's OpenAI-compatible endpoint — see
`tradingagents/llm_clients/factory.py::build_llm_kwargs`). Leaving a value
unset sends no override at all, so the backend/model's own default applies —
TradingAgents never substitutes a guessed default. Valid levels are
model/backend-specific and are not validated against a fixed list; format
only (non-empty string) is checked.

## 10. Observability / trace

With `--trace-llm`, every call's `request.json` records `stage`, `llm_role`
(`"generation"`/`"validator"`/`"review"`, or `null` for a call that bypassed
the role-aware helpers, e.g. Reflection's direct `.invoke()`), `call_identity`,
`model`, and — inside `generation_params` — the effective `seed` and
`reasoning_effort` actually sent, when either applied. A Validator check call
uses a `..._validation` call identity, a post-repair recheck
`..._validation_recheck`, and the original agent's own repair call
`..._validation_repair` (recorded under its real `generation`/`review` role,
never under `validator`) — each producing a distinct derived seed under
best-effort deterministic replay, the same way every other `attempt`/
`call_identity` pair does.

A Validator call's `metadata.json` additionally records (via
`record_validator_debug`, see `agents/validator.py`/`llm_trace.py`):
`structured_parse_success`, `limited_recovery_success`, `json_fallback_called`
(true only when a SEPARATE normalize-only call was actually made --
`limited_recovery_success: true` always implies `json_fallback_called:
false`), `json_fallback_success`, and `validator_raw_response` (the exact raw
text recovery worked from, truncated to 2000 chars) -- enough to answer, for
any Validator call, what the model said, what actually got parsed, and
whether a second call happened. The post-filter step separately records
`validator_findings_raw` / `validator_findings_dropped` (each with its drop
reason) / `validator_findings_final` on the same call.

`run_manifest.json` records the run's `generation_model`/`review_model` and
`generation_reasoning_effort`/`review_reasoning_effort` (the legacy
`effective_deep_model`/`effective_quick_model`/`quick_reasoning_effort`/
`deep_reasoning_effort` fields are kept alongside them for any existing
manifest consumer). `trace_summary.json` mirrors the same pair under
`generation_model`/`review_model` (and the legacy `deep_model`/`quick_model`
keys).

The saved report's "Validation" section (`agents/validator.py::
render_validation_summary`) gives one compact line — stages checked / pass /
warn / fail / repairs / unresolved — plus detail for unresolved findings
only, never a per-stage dump of every PASS. `AgentState.validation_log` holds
the same compact per-stage entries (status, finding count, categories,
repair attempted/succeeded, LLM call count) that feed it — never the full
findings' evidence payloads or a copy of the candidate text, so it stays
small regardless of how many stages are validated.

## 11. Backward compatibility

- Every legacy env var, CLI flag, and config key keeps working exactly as
  before if the canonical equivalent is never set.
- `resolve_effective_models()` (the original quick/deep CLI-override
  resolver) is unchanged, byte-for-byte, for any external caller still using
  it; `resolve_effective_roles()` is the new generation/review equivalent,
  sharing the same underlying precedence algorithm.
- `GraphSetup`/`TradingAgentsGraph` keep `quick_thinking_llm`/
  `deep_thinking_llm` as aliases to the same LLM objects as
  `generation_llm`/`review_llm`, for any internal/external reader not yet
  migrated.
- No prompts, schemas, verifier taxonomy/thresholds, repair conditions,
  debate/risk round counts, trade metrics, integrity checks, action mapping,
  `max_tokens` behavior, seed derivation, provider semantics, or reasoning
  value validation changed as part of this routing/naming change — only
  which LLM object a node receives, and what that pair (now triple) of
  objects is called.
- Validation is fully opt-in (`validation_enabled` defaults to `false`):
  with it unset, `validator_llm` is never constructed, no Validator LLM call
  is ever made, and -- corrected after real-run testing surfaced this as a
  bug, not a documented trade-off -- **no repair is ever triggered either**,
  regardless of what the (always-on, cheap) deterministic checks find.
  Trader and Portfolio Manager's deterministic checks (relational/unit, and
  entry/stop direction for Trader) still run and are still recorded in
  `validation_log` for observability with validation disabled, but a
  deterministic FAIL in that state is observability-only: it never calls
  `generation_llm` for a repair and never mutates the candidate. Only once
  `validation_enabled`/`validator_llm` is set can a FAIL (deterministic or
  from the Validator LLM) trigger the one bounded repair described in §3.4.
