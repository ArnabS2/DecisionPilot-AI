"""
DecisionPilot AI — modules/scorer.py
Multi-dimensional AI scoring engine.

Responsibilities
----------------
  - Score Option A and Option B across six weighted dimensions
  - Derive an aggregate decision score (0-100) per option
  - Compute a confidence percentage that reflects score gap and evidence quality
  - Return a fully-populated, plain Python dict — no external dependencies at
    call time beyond the Anthropic SDK

Six Scoring Dimensions
-----------------------
  1. Goal Alignment        — how directly does the option serve the stated goal?
  2. Cost Impact           — financial cost relative to budget and value gained
  3. Learning Value        — skills, knowledge, and capabilities acquired
  4. Career Growth Potential — long-term trajectory for roles, salary, network
  5. Time Investment       — efficiency of time required vs outcome delivered
  6. Risk Level            — probability × magnitude of negative outcomes (inverted)

Public API
----------
  score_options(...)  -> ScoringResult (dataclass, also exposes .to_dict())
  score_options_dict(...)  -> dict      (plain dict shim for app.py)

Design notes
------------
  - Scoring is performed entirely by the AI; this module orchestrates prompting,
    validation, normalisation, and result packaging.
  - Dimension weights are configurable via DIMENSION_WEIGHTS. They sum to 1.0
    and are multiplied against the raw 0-10 AI scores to produce the 0-100
    aggregate.
  - Confidence is derived algorithmically from the score gap and average
    per-dimension evidence quality reported by the model.
  - All API calls use the shared retry/parse helpers cloned from analyzer.py
    so the module is self-contained and can run independently.
"""

from __future__ import annotations

import json
import logging
import math
import re
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

import anthropic

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────
logger = logging.getLogger("decisionpilot.scorer")
logger.setLevel(logging.INFO)

# ─────────────────────────────────────────────────────────────────────────────
# MODEL CONFIG
# ─────────────────────────────────────────────────────────────────────────────
MODEL           = "claude-sonnet-4-20250514"
MAX_TOKENS      = 1800
MAX_RETRIES     = 2
RETRY_DELAY_SEC = 1.5

# ─────────────────────────────────────────────────────────────────────────────
# DIMENSION DEFINITIONS & WEIGHTS
# ─────────────────────────────────────────────────────────────────────────────
# Each tuple: (dimension_key, display_label, weight, description_for_prompt)
# Weights must sum to exactly 1.0.
DIMENSIONS: list[tuple[str, str, float, str]] = [
    (
        "goal_alignment",
        "Goal Alignment",
        0.28,
        "How directly and completely does this option serve the user's stated personal goal? "
        "Consider both immediate and long-term goal satisfaction.",
    ),
    (
        "cost_impact",
        "Cost Impact",
        0.18,
        "Evaluate the financial cost relative to the stated budget and the value delivered. "
        "Higher scores mean better cost-efficiency or lower financial burden. "
        "If no budget is given, score on absolute cost reasonableness.",
    ),
    (
        "learning_value",
        "Learning Value",
        0.15,
        "How much useful, transferable skill, knowledge, or experience does this option provide? "
        "Consider depth, breadth, and market relevance of what is learned.",
    ),
    (
        "career_growth_potential",
        "Career Growth Potential",
        0.20,
        "Assess the long-term trajectory this option enables: roles, salary progression, "
        "professional network, brand value, and industry positioning.",
    ),
    (
        "time_investment",
        "Time Investment",
        0.10,
        "Rate the efficiency of time required relative to outcomes achieved. "
        "A high score means excellent results-per-hour. "
        "If no time commitment is given, infer from context.",
    ),
    (
        "risk_level",
        "Risk Level",
        0.09,
        "Score on INVERTED risk — 10 means very low risk, 0 means extremely high risk. "
        "Consider financial, career, health, relationship, and opportunity risks together.",
    ),
]

# Validate weights at import time — catches typos immediately
_WEIGHT_SUM = round(sum(w for _, _, w, _ in DIMENSIONS), 10)
assert abs(_WEIGHT_SUM - 1.0) < 1e-9, (
    f"DIMENSIONS weights must sum to 1.0, got {_WEIGHT_SUM}"
)

# Convenient lookups
DIMENSION_KEYS    = [key   for key, _, _, _   in DIMENSIONS]
DIMENSION_LABELS  = [label for _, label, _, _ in DIMENSIONS]
DIMENSION_WEIGHTS = {key: weight for key, _, weight, _ in DIMENSIONS}

# ─────────────────────────────────────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class DimensionScore:
    """Raw AI score + rationale for a single dimension, for one option."""
    key:       str   # e.g. "goal_alignment"
    label:     str   # e.g. "Goal Alignment"
    weight:    float # contribution to aggregate
    raw_score: int   # 0-10 from AI
    rationale: str   # AI explanation for this score
    weighted:  float = field(init=False)  # weight × raw_score × 10 → contribution to 0-100

    def __post_init__(self):
        self.raw_score = _clamp(self.raw_score, 0, 10)
        self.weighted  = round(self.weight * self.raw_score * 10, 2)

    def to_dict(self) -> dict:
        return {
            "key":       self.key,
            "label":     self.label,
            "weight":    self.weight,
            "raw_score": self.raw_score,
            "weighted":  self.weighted,
            "rationale": self.rationale,
        }


@dataclass
class OptionScore:
    """Aggregate score + per-dimension breakdown for one option."""
    label:            str               # option text (truncated for display)
    dimensions:       list[DimensionScore] = field(default_factory=list)
    aggregate_score:  int               = 0   # 0-100 weighted sum
    strongest_dim:    str               = ""  # label of highest-scoring dimension
    weakest_dim:      str               = ""  # label of lowest-scoring dimension
    evidence_quality: int               = 0   # 0-10 AI self-rating of how much it knows

    def __post_init__(self):
        self.aggregate_score = _clamp(self.aggregate_score, 0, 100)
        self.evidence_quality = _clamp(self.evidence_quality, 0, 10)
        if self.dimensions:
            self._derive_extremes()

    def _derive_extremes(self):
        """Identify strongest and weakest dimensions from the breakdown."""
        sorted_dims = sorted(self.dimensions, key=lambda d: d.raw_score)
        self.weakest_dim   = sorted_dims[0].label  if sorted_dims else ""
        self.strongest_dim = sorted_dims[-1].label if sorted_dims else ""

    def to_dict(self) -> dict:
        return {
            "label":            self.label,
            "aggregate_score":  self.aggregate_score,
            "strongest_dim":    self.strongest_dim,
            "weakest_dim":      self.weakest_dim,
            "evidence_quality": self.evidence_quality,
            "dimensions":       [d.to_dict() for d in self.dimensions],
        }


@dataclass
class ScoringResult:
    """
    Top-level result returned by score_options().
    Contains scores for both options plus derived confidence and metadata.
    """
    # Option scores
    option_a: OptionScore = field(default_factory=OptionScore)
    option_b: OptionScore = field(default_factory=OptionScore)

    # Aggregate outputs (the three required outputs per spec)
    score_a:               int   = 0    # 0-100
    score_b:               int   = 0    # 0-100
    confidence_percentage: int   = 50   # 0-100

    # Derived signals
    score_gap:    int  = 0      # abs(score_a - score_b)
    leading:      str  = "A"   # "A" | "B" — which option scored higher
    is_decisive:  bool = False  # True when gap ≥ 10 points

    # Meta
    elapsed_ms:   int  = 0
    raw_response: str  = field(default="", repr=False)

    def __post_init__(self):
        self.score_a               = _clamp(self.score_a, 0, 100)
        self.score_b               = _clamp(self.score_b, 0, 100)
        self.confidence_percentage = _clamp(self.confidence_percentage, 0, 100)
        self.score_gap             = abs(self.score_a - self.score_b)
        self.leading               = "A" if self.score_a >= self.score_b else "B"
        self.is_decisive           = self.score_gap >= 10

    def to_dict(self) -> dict:
        """
        Return the canonical plain-dict result consumed by app.py and other modules.

        Structure
        ---------
        {
          "score_a": int,
          "score_b": int,
          "confidence_percentage": int,
          "score_gap": int,
          "leading": "A" | "B",
          "is_decisive": bool,
          "elapsed_ms": int,
          "option_a": { ... OptionScore dict ... },
          "option_b": { ... OptionScore dict ... },
        }
        """
        return {
            "score_a":               self.score_a,
            "score_b":               self.score_b,
            "confidence_percentage": self.confidence_percentage,
            "score_gap":             self.score_gap,
            "leading":               self.leading,
            "is_decisive":           self.is_decisive,
            "elapsed_ms":            self.elapsed_ms,
            "option_a":              self.option_a.to_dict(),
            "option_b":              self.option_b.to_dict(),
        }


# ─────────────────────────────────────────────────────────────────────────────
# ANTHROPIC CLIENT  (module-level singleton, thread-safe for Streamlit)
# ─────────────────────────────────────────────────────────────────────────────
_client: Optional[anthropic.Anthropic] = None

def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic()   # reads ANTHROPIC_API_KEY from env
    return _client

# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL UTILITIES
# ─────────────────────────────────────────────────────────────────────────────
def _clamp(value: Any, lo: int, hi: int) -> int:
    """Coerce value to int and clamp to [lo, hi]. Never raises."""
    try:
        return max(lo, min(hi, int(value)))
    except (TypeError, ValueError):
        return lo


def _strip_fences(text: str) -> str:
    """
    Remove markdown code fences the model sometimes emits despite instructions.
    Handles:  ```json ... ```  |  ``` ... ```  |  orphan opening fence
    """
    text = text.strip()
    # Full fence block
    m = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    # Orphan opening fence
    if text.startswith("```"):
        lines = text.splitlines()[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return text


def _parse_json(text: str) -> dict:
    """
    Three-stage JSON extraction:
      1. Direct parse after fence stripping
      2. Regex extraction of outermost { ... }
      3. Return {} and log a warning
    """
    cleaned = _strip_fences(text)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    m = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass

    logger.warning("scorer: could not parse JSON. First 400 chars: %s", text[:400])
    return {}


def _call_api(prompt: str, system: str) -> tuple[str, int]:
    """
    Call the Anthropic Messages API with retry on transient errors.

    Returns
    -------
    (raw_text, elapsed_ms)

    Raises
    ------
    Last caught exception if all retries are exhausted.
    """
    client   = _get_client()
    last_exc: Exception = RuntimeError("No attempt made")

    for attempt in range(1, MAX_RETRIES + 2):
        try:
            t0 = time.monotonic()
            response = client.messages.create(
                model     = MODEL,
                max_tokens= MAX_TOKENS,
                system    = system,
                messages  = [{"role": "user", "content": prompt}],
            )
            elapsed = int((time.monotonic() - t0) * 1000)
            logger.info("scorer: API call ok in %d ms (attempt %d)", elapsed, attempt)
            return response.content[0].text, elapsed

        except anthropic.RateLimitError as exc:
            logger.warning("scorer: rate limit (attempt %d): %s", attempt, exc)
            last_exc = exc
            time.sleep(RETRY_DELAY_SEC * attempt)

        except anthropic.APIStatusError as exc:
            logger.error("scorer: API status error (attempt %d): %s", attempt, exc)
            last_exc = exc
            if attempt <= MAX_RETRIES:
                time.sleep(RETRY_DELAY_SEC)

        except Exception as exc:
            logger.error("scorer: unexpected error: %s", exc)
            raise

    raise last_exc

# ─────────────────────────────────────────────────────────────────────────────
# CONFIDENCE ALGORITHM
# ─────────────────────────────────────────────────────────────────────────────
def _compute_confidence(
    score_a:           int,
    score_b:           int,
    evidence_quality_a: int,
    evidence_quality_b: int,
) -> int:
    """
    Derive a confidence percentage from three signals:

    1. Score gap magnitude — a larger gap = higher confidence the leading
       option is genuinely better (sigmoid-scaled so even 1-point gaps
       give some confidence, and 30+ point gaps asymptote near 100%).

    2. Average evidence quality — the model self-rates how much contextual
       information was available to score each option (0-10).
       Low evidence quality dampens confidence.

    3. A floor of 40% — we never claim zero confidence, because some
       information is always present.

    Formula
    -------
        gap_factor      = sigmoid-like mapping of gap onto [0, 1]
        quality_factor  = avg_evidence_quality / 10
        raw_confidence  = 0.65 * gap_factor + 0.35 * quality_factor
        confidence      = floor + (1 - floor) * raw_confidence
        floor           = 0.40
    """
    gap = abs(score_a - score_b)

    # Sigmoid-like gap mapping:  f(x) = 1 - e^(-x/15)
    # f(5)  ≈ 0.28   f(10) ≈ 0.49   f(20) ≈ 0.74   f(30) ≈ 0.86
    gap_factor     = 1.0 - math.exp(-gap / 15.0)

    avg_quality    = (evidence_quality_a + evidence_quality_b) / 2.0
    quality_factor = avg_quality / 10.0

    raw            = 0.65 * gap_factor + 0.35 * quality_factor
    floor          = 0.40
    confidence     = floor + (1.0 - floor) * raw

    return _clamp(round(confidence * 100), 0, 100)

# ─────────────────────────────────────────────────────────────────────────────
# PROMPT BUILDER
# ─────────────────────────────────────────────────────────────────────────────
_SYSTEM_SCORER = """\
You are DecisionPilot AI's Scoring Engine — a precise, analytical evaluator.

Your ONLY task is to score two decision options across predefined dimensions.

Rules (strictly enforced):
1. Every raw_score must be an integer from 0 to 10. No decimals.
2. Every rationale must be 1-2 sentences, specific to the inputs, ≥ 10 words.
3. evidence_quality must be an integer 0-10 reflecting how much context you have.
   10 = full context, 5 = moderate, 1 = very little information.
4. Aggregate scores must equal the weighted sum of dimension scores × 10,
   rounded to the nearest integer. Do NOT invent a different number.
5. Respond ONLY with valid JSON. No markdown. No text outside the JSON object.
"""

def _build_scoring_prompt(
    option_a:        str,
    option_b:        str,
    goal:            str,
    budget:          str,
    context:         str,
) -> str:
    """
    Build the full scoring prompt, injecting dimension definitions
    and the required JSON schema.
    """
    # Build dimension specification block
    dim_spec_lines: list[str] = []
    for key, label, weight, description in DIMENSIONS:
        dim_spec_lines.append(
            f'  • {label} (key: "{key}", weight: {weight:.2f})\n'
            f'    {description}'
        )
    dim_spec = "\n".join(dim_spec_lines)

    # Build the per-option schema fragment (same structure for A and B)
    dim_schema_lines: list[str] = []
    for key, label, weight, _ in DIMENSIONS:
        dim_schema_lines.append(
            f'      "{key}": {{'
            f'"raw_score": <int 0-10>, '
            f'"rationale": "<specific 1-2 sentence reason>"'
            f'}}'
        )
    dim_schema = ",\n".join(dim_schema_lines)

    return f"""\
Score the following two decision options.

━━━ INPUTS ━━━
Option A        : {option_a}
Option B        : {option_b}
Personal Goal   : {goal}
Budget          : {budget or "Not specified"}
Additional Context: {context or "None provided"}

━━━ SCORING DIMENSIONS ━━━
{dim_spec}

━━━ CALCULATION RULE ━━━
aggregate_score = round( sum( raw_score[dim] × weight[dim] × 10 ) )

━━━ REQUIRED JSON SCHEMA ━━━
{{
  "option_a": {{
    "dimensions": {{
{dim_schema}
    }},
    "aggregate_score" : <int 0-100, calculated per rule above>,
    "evidence_quality": <int 0-10, how much context you had to score A>
  }},
  "option_b": {{
    "dimensions": {{
{dim_schema}
    }},
    "aggregate_score" : <int 0-100, calculated per rule above>,
    "evidence_quality": <int 0-10, how much context you had to score B>
  }}
}}

Return ONLY the JSON object. No markdown. No explanation outside the JSON.
"""

# ─────────────────────────────────────────────────────────────────────────────
# RESULT BUILDER
# ─────────────────────────────────────────────────────────────────────────────
def _build_option_score(raw_option: dict, label: str) -> OptionScore:
    """
    Convert one option's raw dict (from AI) into a typed OptionScore.

    Also recomputes the aggregate score locally from dimension scores × weights
    so the result is always mathematically correct, regardless of what the
    model returned.
    """
    raw_dims: dict = raw_option.get("dimensions", {})
    dim_scores:  list[DimensionScore] = []
    weighted_sum: float = 0.0

    for key, display_label, weight, _ in DIMENSIONS:
        dim_data  = raw_dims.get(key, {})
        raw_score = _clamp(dim_data.get("raw_score", 5), 0, 10)
        rationale = str(dim_data.get("rationale", "")).strip() or "No rationale provided."

        ds = DimensionScore(
            key       = key,
            label     = display_label,
            weight    = weight,
            raw_score = raw_score,
            rationale = rationale,
        )
        dim_scores.append(ds)
        weighted_sum += ds.weighted    # already = weight × raw_score × 10

    # Local recompute — authoritative aggregate score
    local_aggregate = _clamp(round(weighted_sum), 0, 100)

    # Cross-check against what the model returned; log a warning if they differ
    model_aggregate = _clamp(raw_option.get("aggregate_score", local_aggregate), 0, 100)
    if abs(local_aggregate - model_aggregate) > 3:
        logger.warning(
            "scorer: model aggregate (%d) differs from local recompute (%d) "
            "for '%s'. Using local value.",
            model_aggregate, local_aggregate, label[:40],
        )

    return OptionScore(
        label             = label[:60],
        dimensions        = dim_scores,
        aggregate_score   = local_aggregate,          # always use locally computed value
        evidence_quality  = _clamp(raw_option.get("evidence_quality", 5), 0, 10),
    )


def _fallback_result(option_a: str, option_b: str) -> ScoringResult:
    """
    Return a safe fallback ScoringResult when the API or parse fails.
    Scores default to 50/50 with 40% confidence.
    """
    logger.error("scorer: returning fallback ScoringResult due to failure.")
    blank_a = OptionScore(label=option_a[:60], aggregate_score=50, evidence_quality=0)
    blank_b = OptionScore(label=option_b[:60], aggregate_score=50, evidence_quality=0)
    return ScoringResult(
        option_a               = blank_a,
        option_b               = blank_b,
        score_a                = 50,
        score_b                = 50,
        confidence_percentage  = 40,
    )

# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────
def score_options(
    option_a: str,
    option_b: str,
    goal:     str,
    budget:   str = "",
    context:  str = "",
) -> ScoringResult:
    """
    Score Option A and Option B across six weighted dimensions using AI.

    Parameters
    ----------
    option_a : str
        Description of the first choice (e.g. "Join the startup as SDE-2").
    option_b : str
        Description of the second choice (e.g. "Stay at current company").
    goal : str
        The user's personal goal that the decision should serve.
    budget : str, optional
        Financial context (free-form, currency-agnostic). Default: "".
    context : str, optional
        Any additional background or constraints. Default: "".

    Returns
    -------
    ScoringResult
        Fully-populated dataclass. Never raises — returns a safe fallback
        with equal 50/50 scores and 40% confidence on API/parse failure.

    Notes
    -----
    - Aggregate scores are always recomputed locally from dimension scores
      and weights, so they are mathematically guaranteed to be correct.
    - Confidence is derived from score gap + evidence quality, not from the AI.
    """
    logger.info("scorer: scoring '%s' vs '%s'", option_a[:40], option_b[:40])

    prompt = _build_scoring_prompt(option_a, option_b, goal, budget, context)

    try:
        raw_text, elapsed_ms = _call_api(prompt, _SYSTEM_SCORER)
    except Exception as exc:
        logger.error("scorer: API call failed: %s", exc)
        return _fallback_result(option_a, option_b)

    parsed = _parse_json(raw_text)
    if not parsed or "option_a" not in parsed or "option_b" not in parsed:
        logger.error("scorer: missing expected keys in parsed response.")
        return _fallback_result(option_a, option_b)

    # Build typed option scores (aggregate recomputed locally inside)
    opt_a = _build_option_score(parsed["option_a"], option_a)
    opt_b = _build_option_score(parsed["option_b"], option_b)

    # Derive confidence algorithmically
    confidence = _compute_confidence(
        score_a            = opt_a.aggregate_score,
        score_b            = opt_b.aggregate_score,
        evidence_quality_a = opt_a.evidence_quality,
        evidence_quality_b = opt_b.evidence_quality,
    )

    result = ScoringResult(
        option_a               = opt_a,
        option_b               = opt_b,
        score_a                = opt_a.aggregate_score,
        score_b                = opt_b.aggregate_score,
        confidence_percentage  = confidence,
        elapsed_ms             = elapsed_ms,
        raw_response           = raw_text,
    )

    logger.info(
        "scorer: done — A=%d | B=%d | gap=%d | confidence=%d%% | %.0f ms",
        result.score_a, result.score_b,
        result.score_gap, result.confidence_percentage, elapsed_ms,
    )
    return result


def score_options_dict(
    option_a: str,
    option_b: str,
    goal:     str,
    budget:   str = "",
    context:  str = "",
) -> dict:
    """
    Plain-dict wrapper around score_options().
    Returns the three required outputs at the top level, plus full breakdown.

    Guaranteed keys in returned dict
    ---------------------------------
    score_a               : int   (0-100)
    score_b               : int   (0-100)
    confidence_percentage : int   (0-100)
    score_gap             : int
    leading               : str   ("A" | "B")
    is_decisive           : bool
    elapsed_ms            : int
    option_a              : dict  (full OptionScore breakdown)
    option_b              : dict  (full OptionScore breakdown)
    """
    return score_options(option_a, option_b, goal, budget, context).to_dict()


# ─────────────────────────────────────────────────────────────────────────────
# DIMENSION METADATA HELPERS  (for UI rendering in app.py)
# ─────────────────────────────────────────────────────────────────────────────
def get_dimension_labels() -> list[str]:
    """Return ordered list of dimension display labels."""
    return DIMENSION_LABELS.copy()


def get_dimension_weights() -> dict[str, float]:
    """Return {dimension_key: weight} mapping."""
    return DIMENSION_WEIGHTS.copy()


def scores_to_radar_series(result: ScoringResult) -> dict:
    """
    Convert a ScoringResult into a structure ready for Plotly radar charts.

    Returns
    -------
    {
      "dimensions":      [str, ...],     # display labels
      "option_a_scores": [int, ...],     # raw_score 0-10 per dimension
      "option_b_scores": [int, ...],
      "option_a_label":  str,
      "option_b_label":  str,
    }
    """
    def _raw_scores(opt: OptionScore) -> list[int]:
        score_map = {d.key: d.raw_score for d in opt.dimensions}
        return [score_map.get(key, 0) for key in DIMENSION_KEYS]

    return {
        "dimensions":      DIMENSION_LABELS,
        "option_a_scores": _raw_scores(result.option_a),
        "option_b_scores": _raw_scores(result.option_b),
        "option_a_label":  result.option_a.label,
        "option_b_label":  result.option_b.label,
    }


# ─────────────────────────────────────────────────────────────────────────────
# MANUAL SMOKE-TEST  (run: python -m modules.scorer)
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    logging.basicConfig(
        level  = logging.INFO,
        stream = sys.stdout,
        format = "%(levelname)s | %(name)s | %(message)s",
    )

    print("\n" + "═" * 62)
    print("  DecisionPilot AI — scorer.py smoke test")
    print("═" * 62 + "\n")

    # ── Weight sanity check ──
    weight_sum = sum(w for _, _, w, _ in DIMENSIONS)
    assert abs(weight_sum - 1.0) < 1e-9, "Weight sum assertion failed"
    print(f"✅ Dimension weights sum to {weight_sum:.4f} (expected 1.0)\n")
    for key, label, w, _ in DIMENSIONS:
        print(f"   {label:<26} weight={w:.2f}  key={key}")

    # ── Confidence algorithm spot checks ──
    print()
    for gap, eq, el in [(0, 5, 5), (10, 7, 6), (25, 8, 8), (40, 9, 9)]:
        a, b = 60, 60 - gap
        c = _compute_confidence(a, b, eq, el)
        print(f"   gap={gap:>2} | eq_a={eq} eq_b={el} → confidence={c}%")

    # ── Live API call ──
    print("\n" + "─" * 62)
    print("  Live API call …")
    print("─" * 62)

    result = score_options(
        option_a = "Accept SDE-2 role at a well-funded Series-A startup — ₹18 LPA + equity",
        option_b = "Stay as Senior Engineer at TCS — ₹14 LPA, stable, low risk",
        goal     = "Reach ₹30 LPA compensation and a Staff Engineer role within 3 years",
        budget   = "6 months of living expenses saved; no dependants",
        context  = "3 years of experience, strong Python/AWS skills, risk-tolerant mindset",
    )

    print(f"\n✅ score_options() completed in {result.elapsed_ms} ms")
    print(f"   score_a               = {result.score_a}")
    print(f"   score_b               = {result.score_b}")
    print(f"   confidence_percentage = {result.confidence_percentage}%")
    print(f"   score_gap             = {result.score_gap}")
    print(f"   leading               = Option {result.leading}")
    print(f"   is_decisive           = {result.is_decisive}")

    print("\n   Option A dimension breakdown:")
    for d in result.option_a.dimensions:
        bar = "█" * d.raw_score + "░" * (10 - d.raw_score)
        print(f"   {d.label:<26} [{bar}] {d.raw_score}/10  (weighted: {d.weighted:.1f})")

    print("\n   Option B dimension breakdown:")
    for d in result.option_b.dimensions:
        bar = "█" * d.raw_score + "░" * (10 - d.raw_score)
        print(f"   {d.label:<26} [{bar}] {d.raw_score}/10  (weighted: {d.weighted:.1f})")

    # ── Dict output check ──
    d = result.to_dict()
    assert "score_a"               in d, "Missing score_a"
    assert "score_b"               in d, "Missing score_b"
    assert "confidence_percentage" in d, "Missing confidence_percentage"
    assert "option_a"              in d, "Missing option_a breakdown"
    assert "option_b"              in d, "Missing option_b breakdown"
    print(f"\n✅ to_dict() keys verified: {sorted(d.keys())}")

    # ── Radar helper ──
    radar = scores_to_radar_series(result)
    assert len(radar["option_a_scores"]) == len(DIMENSIONS)
    print(f"✅ scores_to_radar_series() returned {len(radar['option_a_scores'])} dimensions")

    print("\n" + "═" * 62)
    print("  All scorer smoke tests passed.")
    print("═" * 62 + "\n")
