"""Validator schema: a narrow, objective quality-control layer distinct from
the Research Verifier (see fork README: "Generation / Validation / Review").

Validation checks CONTRACT/FORMAT/ARITHMETIC/OBVIOUS CONSISTENCY -- never
investment reasoning quality, which remains Review's job. The taxonomy below
is deliberately its own enum, not a reuse of the Research Verifier's
``FindingCategory``: a few names overlap because the underlying defect really
is the same thing (ARITHMETIC_ERROR, UNIT_ERROR, RELATIONAL_ERROR), but most
of Validator's categories are about contract/format compliance, a different
semantic axis than the Verifier's evidence-integrity taxonomy, and mixing
them would blur what each layer is actually for.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, field_validator


class ValidationCategory(str, Enum):
    # Shared with the Research Verifier's taxonomy where the defect really is
    # the same thing -- never mix in Verifier-only semantic categories here.
    ARITHMETIC_ERROR = "ARITHMETIC_ERROR"
    RELATIONAL_ERROR = "RELATIONAL_ERROR"
    UNIT_ERROR = "UNIT_ERROR"
    # Validator-specific: contract/format/structural defects, not evidence
    # reasoning quality.
    UNKNOWN_USED_AS_FACT = "UNKNOWN_USED_AS_FACT"
    UNSUPPORTED_NUMBER = "UNSUPPORTED_NUMBER"
    MISSING_REQUIREMENT = "MISSING_REQUIREMENT"
    FORMAT_CONTRACT_ERROR = "FORMAT_CONTRACT_ERROR"
    JSON_SYNTAX_ERROR = "JSON_SYNTAX_ERROR"
    SCHEMA_ERROR = "SCHEMA_ERROR"
    INTERNAL_CONTRADICTION = "INTERNAL_CONTRADICTION"
    PROMPT_REQUIREMENT_VIOLATION = "PROMPT_REQUIREMENT_VIOLATION"
    # A directional plan's stop-loss/take-profit on the wrong side of entry
    # for its own stated direction (e.g. a SHORT with stop below entry) --
    # its own category, not folded into RELATIONAL_ERROR: the defect is a
    # structurally non-executable plan, not a stated-comparison mismatch in
    # prose (see fork README / NVDA 2026-01-07 regression).
    TRADE_GEOMETRY_ERROR = "TRADE_GEOMETRY_ERROR"
    # Used ONLY by the limited free-text recovery path (see validator.py:
    # "LIMITED VALIDATOR RECOVERY") when a small/local model signals FAIL/WARN
    # without a machine-parseable category -- an honest "something was
    # flagged but the category is unknown" marker, never guessed as one of
    # the specific categories above.
    UNCLASSIFIED_VALIDATION_ERROR = "UNCLASSIFIED_VALIDATION_ERROR"


class ValidationStatus(str, Enum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"


class ValidationFinding(BaseModel):
    """One objective defect found in a candidate response.

    Metadata about output quality, never new market/research evidence --
    ``suggested_correction`` is a hint for the original agent's own repair
    pass, not a replacement value to substitute directly.
    """

    category: ValidationCategory
    claim: str = Field(description="The specific statement or value that is wrong.")
    evidence: list[str] = Field(
        default_factory=list,
        description="Supporting excerpts/values from the candidate or its inputs.",
    )
    reason: str = Field(default="", description="Why this is wrong, in one sentence.")
    suggested_correction: str = Field(
        default="", description="A hint for the correction; never applied directly by Validator."
    )

    @field_validator("category", mode="before")
    @classmethod
    def _normalise_category_casing(cls, value):
        # Case/whitespace only -- mirrors VerificationFinding's own casing
        # normalization (schemas.py): a JSON-fallback reconstruction can be
        # semantically identical but case-mismatched, and must not be
        # discarded wholesale over that alone. Exact-name equivalence only --
        # an unrecognised category still fails validation normally.
        return value.strip().upper() if isinstance(value, str) else value

    @field_validator("evidence", mode="before")
    @classmethod
    def _normalise_evidence(cls, value):
        if value is None:
            return []
        return value if isinstance(value, list) else [str(value)]


class ValidationResult(BaseModel):
    """Validator's complete output for one candidate response."""

    status: ValidationStatus
    findings: list[ValidationFinding] = Field(default_factory=list)

    @field_validator("status", mode="before")
    @classmethod
    def _normalise_status(cls, value):
        return value.strip().upper() if isinstance(value, str) else value

    @field_validator("findings", mode="before")
    @classmethod
    def _normalise_findings(cls, value):
        if value is None:
            return []
        return value if isinstance(value, list) else [value]
