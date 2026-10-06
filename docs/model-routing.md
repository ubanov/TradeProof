# Model routing: Generation vs Review

## 1. Purpose

This document describes how TradingAgents routes LLM calls to one of two
roles — **generation** and **review** — and why this fork moved away from
the original `quick_think_llm`/`deep_think_llm` naming. It is the detailed
companion to the README section "Model roles: Generation vs Review"; read
that first for the short version, examples, and the legacy-naming table.

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

## 3. Review role

Review calls **challenge, verify, audit, correct, or repair** something
that generation already produced:

- Bull/Bear's review rounds — challenging the opponent's position and
  defending or revising one's own.
- Research Verifier — the deterministic-adjacent audit of the completed
  debate (structured attempt and JSON fallback both run on review).
- Research repair — correcting a position the Verifier flagged.
- Risk's cross-review round — each reviewer critiquing the other two.

## 4. Full routing table

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
| Trader (structured + fallback) | Generation |
| Risk: Aggressive independent initial assessment | Generation |
| Risk: Conservative independent initial assessment | Generation |
| Risk: Neutral independent initial assessment | Generation |
| Risk: Aggressive cross-review | Review |
| Risk: Conservative cross-review | Review |
| Risk: Neutral cross-review | Review |
| Portfolio Manager (structured + fallback) | Generation |
| Canonical recovery (any fallback) | Inherits the role of the call it recovers |

Canonical recovery never has a role of its own — it reconstructs structured
state from a free-text fallback using the exact same LLM (and therefore the
exact same role) as the call that just fell back. A Bull initial-thesis
fallback's recovery call runs on generation; a Bull review fallback's
recovery call, and a repair fallback's recovery call, run on review.

## 5. Why this differs from upstream

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

Renaming to generation/review makes the split describe **function**, not a
guess about which stages deserve the bigger model. It also reassigns a few
stages along the way (see §4): Research Verifier and Risk's cross-review
move onto the review role; Research Manager and Portfolio Manager move onto
generation. A deployment is free to point generation and review at the same
model (reproducing the old single-model behavior) or at two very different
ones.

## 6. Legacy naming / compatibility

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

| Legacy | New |
|---|---|
| `quick_think_llm` | `generation_llm` |
| `deep_think_llm` | `review_llm` |
| `quick_reasoning_effort` | `generation_reasoning_effort` |
| `deep_reasoning_effort` | `review_reasoning_effort` |

## 7. Configuration examples

```bash
# .env — DeepSeek V4 Pro on generation (high reasoning), Qwen3.8 Flash Next on review (medium)
TRADINGAGENTS_GENERATION_LLM=deepseek-v4-pro:cloud
TRADINGAGENTS_GENERATION_REASONING_EFFORT=high
TRADINGAGENTS_REVIEW_LLM=qwen3.8-flash-next:125b-a6b-q4_K_M
TRADINGAGENTS_REVIEW_REASONING_EFFORT=medium
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

Any model ID the configured provider serves is accepted. Generation and
review may be the same model — there is no requirement to use two.

## 8. Reasoning effort interaction

`generation_reasoning_effort`/`review_reasoning_effort` set `reasoning_effort`
independently per role, forwarded only to providers with verified support
(OpenAI's reasoning-tier models, and Ollama's OpenAI-compatible endpoint —
see `tradingagents/llm_clients/factory.py::build_llm_kwargs`). Leaving a
value unset sends no override at all, so the backend/model's own default
applies — TradingAgents never substitutes a guessed default. Valid levels
are model/backend-specific and are not validated against a fixed list;
format only (non-empty string) is checked.

## 9. Observability / trace

With `--trace-llm`, every call's `request.json` records `stage`, `llm_role`
(`"generation"`/`"review"`, or `null` for a call that bypassed the
role-aware helpers, e.g. Reflection's direct `.invoke()`), `call_identity`,
`model`, and — inside `generation_params` — the effective `seed` and
`reasoning_effort` actually sent, when either applied. `run_manifest.json`
records the run's `generation_model`/`review_model` and
`generation_reasoning_effort`/`review_reasoning_effort` (the legacy
`effective_deep_model`/`effective_quick_model`/`quick_reasoning_effort`/
`deep_reasoning_effort` fields are kept alongside them for any existing
manifest consumer). `trace_summary.json` mirrors the same pair under
`generation_model`/`review_model` (and the legacy `deep_model`/`quick_model`
keys).

## 10. Backward compatibility

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
  which LLM object a node receives, and what that pair of objects is called.
