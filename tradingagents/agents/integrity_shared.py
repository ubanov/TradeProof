"""Deterministic text-scanning utilities shared by the Research Manager and
Risk Management integrity checks.

Both checks are LLM-free, conservative, sentence-level scans over a model's
free-text output: a number near operational-trigger language ("exceeds",
"reduce ... by") is a candidate decision rule, checked against a known-good
pool of numbers already traceable to evidence; a two-number comparison
("X is below Y") is checked for arithmetic consistency on its own; a claimed
reward/risk ratio is checked against the deterministic ``TradeMetrics``
already computed in code. Kept in one module so a change to the underlying
number-extraction or sentence-splitting logic cannot drift between the two
checkers (see fork README: "Research Manager integrity check" and "Risk
integrity check").

False positives (a real, sourced number wrongly flagged) are worse than false
negatives (a low-value new number that slips through) -- every function here
is deliberately narrow sentence-level pattern matching, not a general NLP
parser.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Numeric extraction
# ---------------------------------------------------------------------------

_NUMBER_RE = re.compile(
    r"[$€£]?(\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+\.\d+|\d+)\s?(%|[KMBTkmbt])?(?![a-zA-Z0-9])"
)

_SUFFIX_MULTIPLIER = {"k": 1e3, "m": 1e6, "b": 1e9, "t": 1e12}


def extract_numbers(text: str) -> list[float]:
    """Every number-like token in ``text``, normalised (K/M/B/T expanded)."""
    numbers = []
    for match in _NUMBER_RE.finditer(text or ""):
        raw, suffix = match.group(1), match.group(2)
        try:
            value = float(raw.replace(",", ""))
        except ValueError:
            continue
        if suffix and suffix.lower() in _SUFFIX_MULTIPLIER:
            value *= _SUFFIX_MULTIPLIER[suffix.lower()]
        numbers.append(value)
    return numbers


def numbers_match(value: float, pool: list[float]) -> bool:
    """Whether ``value`` is traceable to a number already in ``pool``.

    A small tolerance absorbs rounding/formatting differences (writing
    "189.50" for a sourced "189.5"), not genuine new values.
    """
    return any(abs(value - known) <= max(0.01, abs(known) * 0.01) for known in pool)


# ---------------------------------------------------------------------------
# Sentence splitting
# ---------------------------------------------------------------------------

# A bare newline is always a boundary, not just one preceded by sentence
# punctuation -- this project's own structured renderers (e.g.
# ``render_risk_assessment`` in schemas.py) join section headers and bullets
# with a single "\n", never trailing whitespace after them ("Risk level:
# HIGH\nDisposition: REDUCE_RISK\n..."). Requiring punctuation before the
# newline (the previous behaviour) left an unrelated header/bullet glued to
# the next one into one giant "sentence", so a single coincidental trigger
# word anywhere in it (e.g. "REDUCE_RISK" matching the "reduce" threshold
# trigger) caused every unrelated number elsewhere in that whole block to be
# flagged as if it were near that trigger -- a false-positive cascade, all
# sharing the same truncated excerpt (see fork README: "Risk integrity
# check" false-positive fix, real-run regression).
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?;])\s+|\n+")


def split_sentences(text: str) -> list[str]:
    return [s for s in _SENTENCE_SPLIT_RE.split(text or "") if s.strip()]


def excerpt(sentence: str, limit: int = 160) -> str:
    sentence = sentence.strip()
    return sentence if len(sentence) <= limit else sentence[: limit - 1].rstrip() + "…"


# ---------------------------------------------------------------------------
# Trigger-word scanning (threshold / sizing / historical / probability)
# ---------------------------------------------------------------------------

THRESHOLD_TRIGGERS = (
    "exceed", "exceeds", "exceeding",
    "above", "below", "over ", "under ",
    "at least", "at most", "more than", "less than", "greater than", "fewer than",
    "reduce", "increase", "cut ", "trim ",
)

# Comparison-style triggers describe a relationship that is ambiguous on its
# own: "price remains above 217.23" can be a NEW conditional rule or simply a
# restatement of where price already sits relative to a known level. These
# require an explicit conditional marker nearby before being treated as a
# candidate new rule (see CONDITIONAL_MARKERS). The imperative/action triggers
# ("reduce", "increase", "cut ", "trim ") are not gated this way: an imperative
# sizing instruction is inherently a rule with or without an "if"/"when".
_COMPARISON_TRIGGERS = (
    "exceed", "exceeds", "exceeding",
    "above", "below", "over ", "under ",
    "at least", "at most", "more than", "less than", "greater than", "fewer than",
)

CONDITIONAL_MARKERS = (
    "if ", "only if", "only when", "when ", "unless", "once ", "should ",
    "provided that", "as long as", "proceed if", "requires", "require ",
    "must ", "needs to", "need to", "before entering", "in order to",
    "only after", "until ",
)

SIZING_WORDS = (
    "allocat", "exposure", "position siz", "portfolio weight",
    "of portfolio", "of the portfolio", "of capital",
)

PORTFOLIO_PERCENT_WORDS = (
    "of portfolio", "of the portfolio", "of equity", "of account", "equity loss",
    "portfolio loss", "account balance", "aum",
)

LEVEL_WORDS = (
    "entry", "target", "stop", "take profit", "take-profit",
    "invalidation", "trigger level", "price level",
)

PROBABILITY_WORDS = ("probability", "chance of", "% chance", "likelihood", "odds of")

HISTORICAL_WORDS = (
    "historically", "in the past", "typically", "usually", "on average",
    "tends to", "has historically", "historical pattern", "historical average",
    "times out of", "% of the time",
)


def find_numbers_near_triggers(text: str, triggers: tuple[str, ...], pool: list[float]) -> list[tuple[str, float]]:
    """``(sentence_excerpt, value)`` for every number near a trigger word that
    is not traceable to ``pool``.

    Skips a sentence that states an explicit comparison between two numbers
    (e.g. "216.48 is below 199.81"): that is a factual claim checked on its
    own by ``find_relational_errors``, not an implied future operational
    trigger -- scanning it here too would double-count the same sentence
    under two different categories.

    A comparison-style trigger ("above", "below", "exceeds", ...) additionally
    requires an explicit conditional marker in the same sentence (see
    ``CONDITIONAL_MARKERS``): "price remains above 217.23" is a restatement of
    where price already sits, not a new rule, while "only proceed if price
    exceeds 217.23" is. Imperative triggers ("reduce", "increase", ...) are
    not gated this way -- an imperative instruction is inherently a rule.
    """
    findings = []
    for sentence in split_sentences(text):
        lower = sentence.lower()
        matched = next((trigger for trigger in triggers if trigger in lower), None)
        if matched is None:
            continue
        if _RELATIONAL_RE.search(sentence):
            continue
        if matched in _COMPARISON_TRIGGERS and not any(marker in lower for marker in CONDITIONAL_MARKERS):
            continue
        for value in extract_numbers(sentence):
            if numbers_match(value, pool):
                continue
            findings.append((excerpt(sentence), value))
    return findings


def find_numbers_near_words(text: str, words: tuple[str, ...]) -> list[tuple[str, float]]:
    """``(sentence_excerpt, value)`` for every number in a sentence containing
    one of ``words`` -- no pool check (used where ANY such number is itself
    the problem, e.g. a fabricated historical frequency)."""
    findings = []
    for sentence in split_sentences(text):
        lower = sentence.lower()
        if not any(word in lower for word in words):
            continue
        for value in extract_numbers(sentence):
            findings.append((excerpt(sentence), value))
    return findings


# ---------------------------------------------------------------------------
# Relational-claim consistency ("X is below Y")
# ---------------------------------------------------------------------------

_NUM_TOKEN = r"-?\$?\d[\d,]*(?:\.\d+)?"
_RELATIONAL_RE = re.compile(
    rf"({_NUM_TOKEN})\s*(?:is|are|was|were)?\s*"
    r"(below|above|under|over|exceeds?|greater than|less than|more than|fewer than)\s*"
    rf"[^.!?;\n\d]{{0,40}}({_NUM_TOKEN})",
    re.IGNORECASE,
)


def _parse_number(raw: str) -> float | None:
    try:
        return float(raw.replace(",", "").replace("$", ""))
    except ValueError:
        return None


def _relation_holds(a: float, op: str, b: float) -> bool:
    op = op.lower()
    if op in ("below", "under", "less than", "fewer than"):
        return a < b
    if op in ("above", "over", "greater than", "more than") or op.startswith("exceed"):
        return a > b
    return True


def find_relational_errors(text: str) -> list[str]:
    """Sentences stating a false comparison between two explicit numbers
    (e.g. "216.48 is below 199.81"). Self-contained: no external pool needed,
    the claim is checked against itself."""
    findings = []
    for sentence in split_sentences(text):
        for match in _RELATIONAL_RE.finditer(sentence):
            a = _parse_number(match.group(1))
            b = _parse_number(match.group(3))
            if a is None or b is None:
                continue
            if not _relation_holds(a, match.group(2), b):
                findings.append(excerpt(sentence))
    return findings


# ---------------------------------------------------------------------------
# Reward/risk ratio consistency
# ---------------------------------------------------------------------------

_RATIO_RE = re.compile(
    r"(?:r/?r|reward[\s/-]*risk|risk[\s/-]*reward)(?:\s+ratio)?"
    r"[^0-9]{0,20}(\d+(?:\.\d+)?)\s*(?::|x)\s*1\b",
    re.IGNORECASE,
)


def find_reward_risk_claims(text: str) -> list[tuple[str, float]]:
    """``(sentence_excerpt, claimed_ratio)`` for every "R/R of X:1" style claim."""
    findings = []
    for sentence in split_sentences(text):
        for match in _RATIO_RE.finditer(sentence):
            findings.append((excerpt(sentence), float(match.group(1))))
    return findings


def find_arithmetic_errors(text: str, known_good_ratios: list[float]) -> list[str]:
    """Sentences claiming a reward/risk ratio that matches none of the
    deterministic ``TradeMetrics.reward_risk`` values already computed in code."""
    pool = [r for r in known_good_ratios if r is not None]
    if not pool:
        return []
    findings = []
    for sentence, claimed in find_reward_risk_claims(text):
        if not numbers_match(claimed, pool):
            findings.append(sentence)
    return findings


# ---------------------------------------------------------------------------
# Withdrawn/retired value reuse
# ---------------------------------------------------------------------------

# Scoped to withdrawn/corrected-value reuse detection only -- NOT used for
# general-purpose extraction (``extract_numbers``), because a sign-aware
# scan there would misread an unrelated range like "5-10" as 5 and -10. The
# ``(?<!\d)`` lookbehind keeps that case safe: a "-" immediately after a digit
# (a range separator) is never treated as a sign, only a "-" preceded by
# whitespace/punctuation/start-of-text is (e.g. "declined -8.8% sequentially").
_SIGNED_NUMBER_RE = re.compile(
    r"(?<!\d)(-)?(\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+\.\d+|\d+)\s?(%|[KMBTkmbt])?(?![a-zA-Z0-9])"
)


def extract_signed_numbers(text: str) -> list[float]:
    """Like ``extract_numbers``, but recognises a leading minus sign when it
    is not itself preceded by a digit -- so a retired negative figure (e.g. a
    corrected "-8.8%") can be matched on reuse, without a hyphenated range
    like "5-10" being misread as 5 and -10 (see real-run regression: a
    corrected "-8.8% normalized sequential decline" resurfacing later).
    """
    numbers = []
    for match in _SIGNED_NUMBER_RE.finditer(text or ""):
        sign, raw, suffix = match.group(1), match.group(2), match.group(3)
        try:
            value = float(raw.replace(",", ""))
        except ValueError:
            continue
        if suffix and suffix.lower() in _SUFFIX_MULTIPLIER:
            value *= _SUFFIX_MULTIPLIER[suffix.lower()]
        numbers.append(-value if sign else value)
    return numbers


#  Deliberately much tighter than ``numbers_match``'s 1% pool-matching
# tolerance: two canonical values a cent apart (240.65 retired, 240.64 the
# supported replacement) must be treated as DISTINCT, not collapsed into the
# same retired value. This only absorbs genuine floating-point representation
# noise (e.g. 240.6499999999997 for an exact 240.65), not a legitimately
# different nearby price.
_WITHDRAWN_MATCH_TOLERANCE = 0.005


def find_withdrawn_values_reused(text: str, withdrawn_values: list[float]) -> list[str]:
    """A withdrawn/retired numeric level resurfacing verbatim, as if it were
    still an active, verified value. Sign-aware (see ``extract_signed_numbers``)
    so a retired negative figure is still caught on reuse. Matching is
    intentionally near-exact (see ``_WITHDRAWN_MATCH_TOLERANCE``): a nearby but
    distinct canonical replacement value must not be flagged merely because it
    is close to the retired one.

    ``withdrawn_values`` is expected to have already been filtered through
    ``exclude_values_still_active`` by the caller -- this function has no
    way to tell "217.23 was Bear's own superseded take-profit" apart from
    "217.23 is Bull's current, never-retired stop-loss" on its own; it only
    sees a bare float.
    """
    if not withdrawn_values:
        return []
    findings = []
    for sentence in split_sentences(text):
        numbers = extract_signed_numbers(sentence)
        for withdrawn in withdrawn_values:
            if any(abs(n - withdrawn) <= _WITHDRAWN_MATCH_TOLERANCE for n in numbers):
                findings.append(f"{excerpt(sentence)} (value {withdrawn} was withdrawn/corrected earlier)")
    return findings


def exclude_values_still_active(withdrawn_values: list[float], active_pool: list[float]) -> list[float]:
    """Drop a "withdrawn" value that is ALSO currently, independently
    active in ``active_pool`` (e.g. ``build_known_number_pool``'s bull/bear
    CURRENT stop_loss/take_profit/entry) -- retiring one side's own stale
    CLAIM about a number must never blacklist the NUMBER itself for every
    other, unrelated, still-valid use (real-run regression: Bear's own
    superseded take-profit of 217.23 correctly retired when Bear revised
    its target to 222.84, but 217.23 is ALSO, completely independently,
    Bull's own current, never-retired, verified stop-loss (the real SMA50
    level) -- flagging every mention of 217.23 anywhere in Risk's text as
    WITHDRAWN_VALUE_REUSED, including Bull's own unchanged, still-active
    plan, was not a reused stale value, it was a different side's live
    evidence that happens to share the same real-world price level).

    Distinguishes: retiring a VALUE ("withdraw stop 217.23" with nothing
    else currently vouching for 217.23) still blocks reuse of 217.23, as
    intended. Retiring an INTERPRETATION or a stale CLAIM about a number
    that remains independently active elsewhere does not retire the number
    itself.
    """
    if not withdrawn_values or not active_pool:
        return list(withdrawn_values)
    return [value for value in withdrawn_values if not numbers_match(value, active_pool)]


# ---------------------------------------------------------------------------
# Corrected-away values from free-text review corrections
# ---------------------------------------------------------------------------

_CORRECTION_CONNECTOR_RE = re.compile(
    rf"({_NUM_TOKEN})\s*(?:%|percent)?[^0-9]{{0,40}}"
    r"(?:corrected to|should be|is actually|not|instead of|revised to|updated to)"
    rf"[^0-9]{{0,10}}({_NUM_TOKEN})",
    re.IGNORECASE,
)


def extract_corrected_away_numbers(correction_notes: list[str]) -> list[float]:
    """The superseded ("old"/incorrect) number from each "X corrected to Y"
    style note, so it can be tracked as withdrawn even when it never went
    through the structured trade-plan-field KEEP/REVISE/WITHDRAW mechanism
    (e.g. a narrative metric like "normalized profit -8.8%", not a price
    level). Conservative: only fires on an explicit two-number connector
    phrase, never guesses which of two co-occurring numbers is the old one.
    """
    retired = []
    for note in correction_notes:
        match = _CORRECTION_CONNECTOR_RE.search(note or "")
        if match:
            value = _parse_number(match.group(1))
            if value is not None:
                retired.append(value)
    return retired


# ---------------------------------------------------------------------------
# Unit confusion (a percentage stated as if it WERE an absolute price level)
# ---------------------------------------------------------------------------

_PERCENT_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?\s*%")

# A LEVEL_WORDS match followed (within a short window) by one of these means
# the percentage describes something else -- a distance/change BETWEEN prices,
# a share of revenue/profit/equity, an ATR percentage, a ratio -- not a claim
# that the level itself IS that percentage. Conservative on purpose: a real
# unit error ("stop should be 5%", "entry at 5%") has nothing but a plain
# connector (is/at/of/:/=) between the level word and the percentage.
_UNIT_ERROR_EXCLUDE_IN_WINDOW = (
    "->", "to ", "allow", "additional", "movement", " of ", "change",
    "distance", "move", "versus", "vs ", "compared", "quarterly", "revenue",
    "profit", "equity", "atr", "ratio",
)
_UNIT_ERROR_WINDOW = 40


def find_unit_errors(text: str) -> list[str]:
    """A LEVEL_WORDS sentence (entry/stop/target/...) stating that the level
    itself IS a bare percentage instead of an absolute price -- this system
    always expresses those fields as absolute prices (see global policy:
    "preserve units"), so that specific claim is a unit confusion.

    Scoped to avoid flagging a percentage DISTANCE between two prices, a
    percentage OF revenue/profit/equity, a percent change, an ATR percentage,
    or a ratio expressed as a percentage -- all legitimate prose that happens
    to share a sentence with a level word (see ``_UNIT_ERROR_EXCLUDE_IN_WINDOW``).
    """
    findings = []
    for sentence in split_sentences(text):
        lower = sentence.lower()
        for level_word in LEVEL_WORDS:
            idx = lower.find(level_word)
            if idx == -1:
                continue
            window = lower[idx + len(level_word): idx + len(level_word) + _UNIT_ERROR_WINDOW]
            if any(excluded in window for excluded in _UNIT_ERROR_EXCLUDE_IN_WINDOW):
                continue
            if _PERCENT_NUMBER_RE.search(window):
                findings.append(excerpt(sentence))
                break
    return findings
