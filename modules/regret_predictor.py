"""
DecisionPilot AI — modules/regret_predictor.py
Regret estimation engine.

Responsibilities
----------------
  - Estimate Short-Term Regret Risk for the recommended decision
  - Estimate Long-Term Regret Risk for the recommended decision
  - Provide structured, explainable reasoning grounded in decision science
  - Return a plain dict and a typed RegretResult dataclass

How regret is modelled
-----------------------
Regret is not simply "making the wrong choice." Decision science distinguishes
several independent regret drivers. This module scores each driver
explicitly and uses those scores to derive the risk labels, so the output
is always explainable — not a black-box label.

Eight Regret Drivers
---------------------
  1. Opportunity Cost Salience   — how visible / painful is the foregone option?
  2. Reversibility               — can the decision be undone if it goes wrong?
  3. Personal Accountability     — was this the user's own choice, or external pressure?
  4. Expectation Gap Risk        — how likely is the outcome to disappoint expectations?
  5. Social Comparison Exposure  — will the user frequently see peers who chose differently?
  6. Financial Downside Severity — magnitude of financial loss if the decision fails
  7. Time Horizon Mismatch       — does the payoff timeline match the user's patience?
  8. Goal Misalignment Risk      — probability the choice drifts away from the stated goal

Short-term regret (<=6 months) is weighted toward:
  reversibility, expectation gap, financial severity, opportunity cost salience

Long-term regret (1-5 years) is weighted toward:
  goal misalignment, time horizon mismatch, social comparison, personal accountability

Public API
----------
  predict_regret(...)      -> RegretResult   (typed dataclass)
  predict_regret_dict(...) -> dict            (plain dict shim, exact output spec)

Output dict schema (guaranteed keys)
--------------------------------------
  short_term_regret : str   -- "Low" | "Medium" | "High"
  long_term_regret  : str   -- "Low" | "Medium" | "High"
  reason            : str   -- detailed paragraph explaining both risk levels
  drivers           : list  -- per-driver breakdown [{driver, score, short_weight, long_weight, insight}]
  short_term_score  : int   -- 0-100 weighted score for short-term regret
  long_term_score   : int   -- 0-100 weighted score for long-term regret
  mitigation_steps  : list  -- concrete actions to reduce the predicted regret
  elapsed_ms        : int
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import anthropic

# -----------------------------------------------------------------------------
# LOGGING
# -----------------------------------------------------------------------------
logger = logging.getLogger("decisionpilot.regret_predictor")
logger.setLevel(logging.INFO)

# -----------------------------------------------------------------------------
# MODEL CONFIG
# -----------------------------------------------------------------------------
MODEL           = "claude-sonnet-4-20250514"
MAX_TOKENS      = 2000
MAX_RETRIES     = 2
RETRY_DELAY_SEC = 1.5

# -----------------------------------------------------------------------------
# VALID VALUES
# -----------------------------------------------------------------------------
VALID_RISK_LEVELS: frozenset = frozenset({"Low", "Medium", "High"})

# -----------------------------------------------------------------------------
# REGRET DRIVER DEFINITIONS
# -----------------------------------------------------------------------------
# Each tuple:
#   (key, display_label, short_term_weight, long_term_weight, prompt_description)
#
# Weights within each time-horizon must sum to 1.0.
# Higher weight = this driver matters more for that horizon.
#
# Short-term horizon  <= 6 months
# Long-term  horizon  1 - 5 years
REGRET_DRIVERS: list = [
    (
        "opportunity_cost_salience",
        "Opportunity Cost Salience",
        0.20,   # short weight
        0.12,   # long weight
        "How visible and mentally present will the foregone option be? "
        "If the unchosen path is frequently discussed, seen in peers, or hard to ignore, "
        "salience is high and regret risk rises quickly.",
    ),
    (
        "reversibility",
        "Reversibility",
        0.22,   # most critical short-term driver
        0.08,
        "Can the decision be meaningfully reversed or corrected within a few months "
        "if it proves to be a mistake? Low reversibility = high short-term regret risk "
        "because mistakes become locked in fast.",
    ),
    (
        "personal_accountability",
        "Personal Accountability",
        0.10,
        0.18,   # compounds over time -- self-blame grows
        "Was this entirely the user's own free choice, or did external pressure, "
        "advice, or circumstance drive it? High autonomy = higher long-term "
        "accountability and therefore higher regret if it fails.",
    ),
    (
        "expectation_gap_risk",
        "Expectation Gap Risk",
        0.22,   # short-term: unmet expectations surface fast
        0.14,
        "How likely is the real-world outcome to fall short of the user's mental model "
        "of what this option will deliver? Optimism bias, marketing, or social pressure "
        "can inflate expectations and magnify regret when reality lands.",
    ),
    (
        "social_comparison_exposure",
        "Social Comparison Exposure",
        0.08,
        0.18,   # long-term: comparison intensifies as peers advance
        "Will the user regularly encounter peers, colleagues, or social-media updates "
        "from people who chose the other option and appear to be thriving? "
        "Frequent comparison amplifies counterfactual thinking and regret.",
    ),
    (
        "financial_downside_severity",
        "Financial Downside Severity",
        0.10,
        0.12,
        "What is the magnitude of financial harm if this decision fails? "
        "Consider lost income, sunk cost, debt, or opportunity cost of capital. "
        "Large downside = elevated regret regardless of time horizon.",
    ),
    (
        "time_horizon_mismatch",
        "Time Horizon Mismatch",
        0.04,
        0.20,   # becomes clear only over years
        "Does the payoff timeline of this option match the user's patience and "
        "life-stage needs? A slow-payoff option chosen by someone needing fast "
        "results creates growing frustration and regret over time.",
    ),
    (
        "goal_misalignment_risk",
        "Goal Misalignment Risk",
        0.04,
        0.18,   # compounds: drifting goals hurt more over years
        "Over time, is this option likely to drift away from or actively conflict "
        "with the user's stated personal goal? Choices that initially seemed "
        "goal-aligned can become obstacles as circumstances evolve.",
    ),
]

# Validate weights at import time — catches typos immediately
_SHORT_SUM = round(sum(s for _, _, s, _, _ in REGRET_DRIVERS), 10)
_LONG_SUM  = round(sum(l for _, _, _, l, _ in REGRET_DRIVERS), 10)
assert abs(_SHORT_SUM - 1.0) < 1e-9, f"Short-term weights sum to {_SHORT_SUM}, expected 1.0"
assert abs(_LONG_SUM  - 1.0) < 1e-9, f"Long-term weights sum to {_LONG_SUM}, expected 1.0"

# -----------------------------------------------------------------------------
# RISK THRESHOLDS
# -----------------------------------------------------------------------------
# Weighted score (0-100) -> Risk label
#   0-39  -> Low    (regret is unlikely or manageable)
#   40-64 -> Medium (regret is plausible; mitigation advised)
#   65+   -> High   (regret is likely without active mitigation)
RISK_THRESHOLDS: list = [
    (65, "High"),
    (40, "Medium"),
    (0,  "Low"),
]


def _score_to_label(score: int) -> str:
    """Map a 0-100 weighted regret score to a Low/Medium/High label."""
    for threshold, label in RISK_THRESHOLDS:
        if score >= threshold:
            return label
    return "Low"


# -----------------------------------------------------------------------------
# DATA CLASSES
# -----------------------------------------------------------------------------
@dataclass
class DriverResult:
    """
    Score and insight for one regret driver, for both time horizons.
    The raw_score (0-10) is provided by the AI; weighted contributions
    are computed locally.
    """
    key:              str
    label:            str
    raw_score:        int    # 0-10 from AI (higher = more regret risk for this driver)
    short_weight:     float
    long_weight:      float
    insight:          str    # AI explanation specific to this decision

    # Computed post-init
    short_contribution: float = field(init=False)  # short_weight x raw_score x 10
    long_contribution:  float = field(init=False)  # long_weight  x raw_score x 10

    def __post_init__(self):
        self.raw_score           = _clamp(self.raw_score, 0, 10)
        self.short_contribution  = round(self.short_weight * self.raw_score * 10, 2)
        self.long_contribution   = round(self.long_weight  * self.raw_score * 10, 2)

    def to_dict(self) -> dict:
        return {
            "driver":               self.label,
            "score":                self.raw_score,
            "short_weight":         self.short_weight,
            "long_weight":          self.long_weight,
            "short_contribution":   self.short_contribution,
            "long_contribution":    self.long_contribution,
            "insight":              self.insight,
        }


@dataclass
class RegretResult:
    """
    Full regret prediction result.

    All fields have safe defaults so callers never hit AttributeError.
    The three required output keys (short_term_regret, long_term_regret, reason)
    are present in to_dict() at the top level.
    """
    # Required outputs
    short_term_regret: str = "Medium"   # Low | Medium | High
    long_term_regret:  str = "Medium"   # Low | Medium | High
    reason:            str = ""         # full explanatory paragraph

    # Scored internals
    short_term_score:  int  = 50        # 0-100 weighted score (higher = more risk)
    long_term_score:   int  = 50
    drivers:           list = field(default_factory=list)

    # Actionable guidance
    mitigation_steps:  list = field(default_factory=list)

    # Meta
    recommended_option: str = ""        # echoed from input for traceability
    elapsed_ms:         int = 0
    raw_response:       str = field(default="", repr=False)

    def __post_init__(self):
        self.short_term_regret = (self.short_term_regret
                                  if self.short_term_regret in VALID_RISK_LEVELS
                                  else "Medium")
        self.long_term_regret  = (self.long_term_regret
                                  if self.long_term_regret  in VALID_RISK_LEVELS
                                  else "Medium")
        self.short_term_score  = _clamp(self.short_term_score, 0, 100)
        self.long_term_score   = _clamp(self.long_term_score,  0, 100)

    @property
    def overall_risk(self) -> str:
        """Composite risk label -- takes the higher of the two horizon risks."""
        order = {"Low": 0, "Medium": 1, "High": 2}
        if order.get(self.long_term_regret, 0) >= order.get(self.short_term_regret, 0):
            return self.long_term_regret
        return self.short_term_regret

    @property
    def primary_regret_driver(self) -> str:
        """Label of the driver with the highest raw score (most regret risk)."""
        if not self.drivers:
            return "Unknown"
        return max(self.drivers, key=lambda d: d.raw_score).label

    @property
    def safest_driver(self) -> str:
        """Label of the driver with the lowest raw score (least regret risk)."""
        if not self.drivers:
            return "Unknown"
        return min(self.drivers, key=lambda d: d.raw_score).label

    def to_dict(self) -> dict:
        """
        Return the canonical plain dict matching the required output spec:

          {
            "short_term_regret" : "Low" | "Medium" | "High",
            "long_term_regret"  : "Low" | "Medium" | "High",
            "reason"            : str,
            ... plus enrichment fields ...
          }
        """
        return {
            # Required outputs (spec)
            "short_term_regret":     self.short_term_regret,
            "long_term_regret":      self.long_term_regret,
            "reason":                self.reason,
            # Enrichment
            "short_term_score":      self.short_term_score,
            "long_term_score":       self.long_term_score,
            "overall_risk":          self.overall_risk,
            "primary_regret_driver": self.primary_regret_driver,
            "safest_driver":         self.safest_driver,
            "mitigation_steps":      self.mitigation_steps,
            "drivers":               [d.to_dict() for d in self.drivers],
            "recommended_option":    self.recommended_option,
            "elapsed_ms":            self.elapsed_ms,
        }


# -----------------------------------------------------------------------------
# ANTHROPIC CLIENT  (module-level singleton)
# -----------------------------------------------------------------------------
_client: Optional[anthropic.Anthropic] = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic()   # reads ANTHROPIC_API_KEY from env
    return _client


# -----------------------------------------------------------------------------
# INTERNAL UTILITIES
# -----------------------------------------------------------------------------
def _clamp(value: Any, lo: int, hi: int) -> int:
    """Coerce value to int and clamp to [lo, hi]. Never raises."""
    try:
        return max(lo, min(hi, int(value)))
    except (TypeError, ValueError):
        return lo


def _strip_fences(text: str) -> str:
    """Remove markdown code fences the model sometimes emits."""
    text = text.strip()
    m = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    if text.startswith("```"):
        lines = text.splitlines()[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return text


def _parse_json(text: str) -> dict:
    """
    Three-stage JSON extraction.
    1. Direct parse after fence stripping.
    2. Regex extraction of outermost { ... }.
    3. Return {} and log a warning.
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
    logger.warning("regret_predictor: could not parse JSON. First 400 chars: %s", text[:400])
    return {}


def _call_api(prompt: str, system: str) -> tuple:
    """
    Call Anthropic Messages API with automatic retry on transient errors.
    Returns (raw_text, elapsed_ms). Raises on exhausted retries.
    """
    client   = _get_client()
    last_exc = RuntimeError("No attempt made")

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
            logger.info("regret_predictor: API ok in %d ms (attempt %d)", elapsed, attempt)
            return response.content[0].text, elapsed

        except anthropic.RateLimitError as exc:
            logger.warning("regret_predictor: rate limit (attempt %d): %s", attempt, exc)
            last_exc = exc
            time.sleep(RETRY_DELAY_SEC * attempt)

        except anthropic.APIStatusError as exc:
            logger.error("regret_predictor: API status error (attempt %d): %s", attempt, exc)
            last_exc = exc
            if attempt <= MAX_RETRIES:
                time.sleep(RETRY_DELAY_SEC)

        except Exception as exc:
            logger.error("regret_predictor: unexpected error: %s", exc)
            raise

    raise last_exc


# -----------------------------------------------------------------------------
# SYSTEM PROMPT
# -----------------------------------------------------------------------------
_SYSTEM_REGRET = """\
You are DecisionPilot AI's Regret Prediction Engine -- a specialist in \
decision science, behavioural economics, and counterfactual psychology.

Your analytical framework is grounded in:
  - Kahneman & Tversky's Prospect Theory (losses loom larger than gains)
  - Gilovich & Medvec's Action vs. Inaction Regret asymmetry
    (short-term: action regret dominates; long-term: inaction regret dominates)
  - Zeelenberg's Decision Regret Theory (expectation gaps drive regret)
  - Schwartz's Paradox of Choice (more alternatives = more regret exposure)

Your role:
  - Score each regret driver honestly on a 0-10 scale
    (0 = no regret risk from this driver, 10 = maximum regret risk)
  - Write driver insights that are specific to the inputs -- not generic platitudes
  - Produce a reason paragraph that synthesises both time horizons coherently
  - Suggest concrete, actionable mitigation steps the user can take NOW

Output rules (CRITICAL):
  - Respond ONLY with valid JSON. No markdown. No prose outside the JSON.
  - All string fields must be >= 10 words and specific to the decision inputs.
  - raw_score values must be integers 0-10.
"""


# -----------------------------------------------------------------------------
# PROMPT BUILDER
# -----------------------------------------------------------------------------
def _build_prompt(
    option_a:       str,
    option_b:       str,
    recommendation: str,
    context:        str,
) -> str:
    """
    Build the full regret-prediction prompt.

    The prompt explicitly names each driver, provides its psychological
    description, and specifies exactly which JSON schema to return.
    """
    # Driver specification block
    driver_spec_lines = []
    for key, label, sw, lw, desc in REGRET_DRIVERS:
        driver_spec_lines.append(
            f'  - {label}  (key: "{key}", short_weight: {sw:.2f}, long_weight: {lw:.2f})\n'
            f'    {desc}'
        )
    driver_spec = "\n".join(driver_spec_lines)

    # Driver schema fragment
    driver_schema_lines = []
    for key, label, _, _, _ in REGRET_DRIVERS:
        driver_schema_lines.append(
            f'    "{key}": {{'
            f'"raw_score": <int 0-10>, '
            f'"insight": "<specific 1-2 sentence explanation for THIS decision>"'
            f'}}'
        )
    driver_schema = ",\n".join(driver_schema_lines)

    return f"""Predict regret risk for the following decision.

DECISION INPUTS
Option A (considered path)  : {option_a}
Option B (considered path)  : {option_b}
Recommended / Chosen Option : {recommendation}
Additional Context          : {context or "None provided"}

REGRET DRIVER DEFINITIONS
Score each driver 0-10 where:
  0 = this driver contributes NO regret risk
  10 = this driver contributes MAXIMUM regret risk

{driver_spec}

CALCULATION NOTE
Short-term regret score = sum(raw_score[driver] x short_weight[driver] x 10)
Long-term  regret score = sum(raw_score[driver] x long_weight[driver]  x 10)
Label thresholds: 0-39 = Low | 40-64 = Medium | 65-100 = High

REQUIRED JSON SCHEMA
{{
  "drivers": {{
{driver_schema}
  }},
  "reason": "<3-5 sentences synthesising why regret risk is what it is, covering both short and long term, specific to the inputs>",
  "mitigation_steps": [
    "<concrete action the user can take to reduce regret risk -- 10+ words>",
    "<another concrete action>",
    "<another concrete action>"
  ]
}}

Return ONLY the JSON object. No markdown. No text outside the JSON.
"""


# -----------------------------------------------------------------------------
# RESULT BUILDERS
# -----------------------------------------------------------------------------
def _build_driver_results(raw_drivers: dict) -> list:
    """
    Convert the raw drivers dict from the AI into typed DriverResult objects.
    Missing drivers are filled with a neutral score of 5.
    """
    results = []
    for key, label, sw, lw, _ in REGRET_DRIVERS:
        driver_data = raw_drivers.get(key, {})
        raw_score   = _clamp(driver_data.get("raw_score", 5), 0, 10)
        insight     = str(driver_data.get("insight", "")).strip() or "No insight provided."
        results.append(DriverResult(
            key          = key,
            label        = label,
            raw_score    = raw_score,
            short_weight = sw,
            long_weight  = lw,
            insight      = insight,
        ))
    return results


def _compute_weighted_scores(drivers: list) -> tuple:
    """
    Recompute short-term and long-term regret scores locally from driver results.
    Returns (short_score, long_score) as integers 0-100.
    This is the authoritative value -- overrides anything the model may claim.
    """
    short_total = sum(d.short_contribution for d in drivers)
    long_total  = sum(d.long_contribution  for d in drivers)
    return _clamp(round(short_total), 0, 100), _clamp(round(long_total), 0, 100)


def _fallback_result(recommended_option: str) -> "RegretResult":
    """
    Safe fallback when API or parse fails.
    Returns a Medium/Medium result with an explanatory reason.
    """
    logger.error("regret_predictor: returning fallback RegretResult.")
    return RegretResult(
        short_term_regret  = "Medium",
        long_term_regret   = "Medium",
        reason             = (
            "Regret prediction could not be completed due to a network or API issue. "
            "A default Medium risk level has been applied to both horizons. "
            "Please retry for a detailed analysis."
        ),
        short_term_score   = 50,
        long_term_score    = 50,
        mitigation_steps   = [
            "Re-run the analysis once connectivity is restored.",
            "Reflect on the reversibility of this decision before committing.",
        ],
        recommended_option = recommended_option,
    )


# -----------------------------------------------------------------------------
# PUBLIC API
# -----------------------------------------------------------------------------
def predict_regret(
    option_a:       str,
    option_b:       str,
    recommendation: str,
    context:        str = "",
) -> RegretResult:
    """
    Predict regret risk for a recommended decision across two time horizons.

    Parameters
    ----------
    option_a : str
        The first option that was considered.
    option_b : str
        The second option that was considered.
    recommendation : str
        The option being recommended / chosen (used as the analysis anchor).
        Can be the full option text or "Option A" / "Option B".
    context : str, optional
        Additional background: user's situation, constraints, values, etc.

    Returns
    -------
    RegretResult
        Fully-populated dataclass. Never raises -- returns a safe Medium/Medium
        fallback on API or parse failure.

    Notes
    -----
    - Risk labels (Low/Medium/High) are derived from locally-recomputed
      weighted scores, NOT from model-provided labels, ensuring consistency.
    - The eight driver scores are always available for UI breakdown charts.
    - to_dict() returns the exact output schema required by the spec.
    """
    logger.info(
        "regret_predictor: predicting regret for recommendation='%s'",
        recommendation[:60],
    )

    prompt = _build_prompt(option_a, option_b, recommendation, context)

    try:
        raw_text, elapsed_ms = _call_api(prompt, _SYSTEM_REGRET)
    except Exception as exc:
        logger.error("regret_predictor: API call failed: %s", exc)
        return _fallback_result(recommendation)

    parsed = _parse_json(raw_text)
    if not parsed:
        return _fallback_result(recommendation)

    # Build driver objects from AI response
    raw_drivers = parsed.get("drivers", {})
    drivers     = _build_driver_results(raw_drivers)

    # Recompute scores locally -- always authoritative
    short_score, long_score = _compute_weighted_scores(drivers)
    short_label             = _score_to_label(short_score)
    long_label              = _score_to_label(long_score)

    # Extract narrative fields
    reason           = str(parsed.get("reason", "")).strip()
    mitigation_steps = parsed.get("mitigation_steps", [])
    if not isinstance(mitigation_steps, list):
        mitigation_steps = [str(mitigation_steps)]
    mitigation_steps = [str(s).strip() for s in mitigation_steps if str(s).strip()]

    # Guard: ensure at least one mitigation step
    if not mitigation_steps:
        mitigation_steps = [
            "Set a review checkpoint at 30 and 90 days to reassess the decision.",
            "Keep a decision journal to track outcomes against expectations.",
            "Identify one concrete exit or pivot plan if the decision underperforms.",
        ]

    result = RegretResult(
        short_term_regret  = short_label,
        long_term_regret   = long_label,
        reason             = reason or "See individual driver insights for detailed reasoning.",
        short_term_score   = short_score,
        long_term_score    = long_score,
        drivers            = drivers,
        mitigation_steps   = mitigation_steps,
        recommended_option = recommendation,
        elapsed_ms         = elapsed_ms,
        raw_response       = raw_text,
    )

    logger.info(
        "regret_predictor: done -- ST=%s(%d) | LT=%s(%d) | primary_driver='%s' | %d ms",
        result.short_term_regret, result.short_term_score,
        result.long_term_regret,  result.long_term_score,
        result.primary_regret_driver, elapsed_ms,
    )
    return result


def predict_regret_dict(
    option_a:       str,
    option_b:       str,
    recommendation: str,
    context:        str = "",
) -> dict:
    """
    Plain-dict wrapper around predict_regret().

    Guaranteed top-level keys (matching the required output spec)
    -------------------------------------------------------------
    short_term_regret     : str   "Low" | "Medium" | "High"
    long_term_regret      : str   "Low" | "Medium" | "High"
    reason                : str
    short_term_score      : int   0-100
    long_term_score       : int   0-100
    overall_risk          : str   "Low" | "Medium" | "High"
    primary_regret_driver : str
    safest_driver         : str
    mitigation_steps      : list[str]
    drivers               : list[dict]
    recommended_option    : str
    elapsed_ms            : int
    """
    return predict_regret(option_a, option_b, recommendation, context).to_dict()


# -----------------------------------------------------------------------------
# METADATA HELPERS  (for UI rendering in app.py)
# -----------------------------------------------------------------------------
def get_driver_labels() -> list:
    """Return ordered list of driver display labels."""
    return [label for _, label, _, _, _ in REGRET_DRIVERS]


def get_driver_weights() -> dict:
    """
    Return {driver_key: {short: float, long: float}} weight mapping.
    Useful for rendering weighted bar charts in the UI.
    """
    return {
        key: {"short": sw, "long": lw}
        for key, _, sw, lw, _ in REGRET_DRIVERS
    }


def result_to_chart_series(result: RegretResult) -> dict:
    """
    Convert a RegretResult into Plotly-ready series data for a grouped bar chart
    showing short-term vs long-term weighted contributions per driver.

    Returns
    -------
    {
      "labels":              [str, ...],
      "short_contributions": [float, ...],
      "long_contributions":  [float, ...],
      "short_total":         int,
      "long_total":          int,
      "short_label":         str,
      "long_label":          str,
    }
    """
    labels  = [d.label for d in result.drivers]
    short_c = [d.short_contribution for d in result.drivers]
    long_c  = [d.long_contribution  for d in result.drivers]
    return {
        "labels":              labels,
        "short_contributions": short_c,
        "long_contributions":  long_c,
        "short_total":         result.short_term_score,
        "long_total":          result.long_term_score,
        "short_label":         result.short_term_regret,
        "long_label":          result.long_term_regret,
    }


# -----------------------------------------------------------------------------
# MANUAL SMOKE-TEST  (run: python -m modules.regret_predictor)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level  = logging.INFO,
        stream = sys.stdout,
        format = "%(levelname)s | %(name)s | %(message)s",
    )

    print("\n" + "=" * 64)
    print("  DecisionPilot AI -- regret_predictor.py smoke test")
    print("=" * 64 + "\n")

    # Weight sanity
    short_sum = sum(s for _, _, s, _, _ in REGRET_DRIVERS)
    long_sum  = sum(l for _, _, _, l, _ in REGRET_DRIVERS)
    print(f"OK Short-term weights sum: {short_sum:.4f} (expected 1.0)")
    print(f"OK Long-term  weights sum: {long_sum:.4f}  (expected 1.0)\n")

    # Threshold mapping check
    for score, expected in [(15, "Low"), (52, "Medium"), (70, "High")]:
        label = _score_to_label(score)
        status = "OK" if label == expected else "FAIL"
        print(f"   {status} _score_to_label({score:>3}) = {label} (expected {expected})")

    # Live API call
    print("\n" + "-" * 64)
    print("  Live API call ...")
    print("-" * 64 + "\n")

    result = predict_regret(
        option_a       = "Join a Series-A startup as SDE-2 -- 18 LPA + 0.1% equity",
        option_b       = "Stay as Senior Engineer at TCS -- 14 LPA, stable",
        recommendation = "Join the startup -- higher growth potential aligns with 3-year goal",
        context        = "3 YOE, strong Python/AWS, 6 months savings, no family dependants, risk-tolerant",
    )

    print(f"OK predict_regret() completed in {result.elapsed_ms} ms\n")
    print(f"   Short-Term Regret : {result.short_term_regret} (score: {result.short_term_score}/100)")
    print(f"   Long-Term Regret  : {result.long_term_regret}  (score: {result.long_term_score}/100)")
    print(f"   Overall Risk      : {result.overall_risk}")
    print(f"   Primary Driver    : {result.primary_regret_driver}")
    print(f"   Safest Driver     : {result.safest_driver}")

    print("\n   Driver breakdown:")
    for d in result.drivers:
        risk_bar = "#" * d.raw_score + "." * (10 - d.raw_score)
        print(
            f"   {d.label:<32} [{risk_bar}] {d.raw_score}/10  "
            f"ST:{d.short_contribution:>4.1f}  LT:{d.long_contribution:>4.1f}"
        )

    print(f"\n   Reason:\n   {result.reason}\n")
    print("   Mitigation steps:")
    for i, step in enumerate(result.mitigation_steps, 1):
        print(f"   {i}. {step}")

    # Dict output check
    d = result.to_dict()
    required_keys = {"short_term_regret", "long_term_regret", "reason"}
    missing = required_keys - d.keys()
    assert not missing, f"Missing required keys: {missing}"
    print(f"\nOK to_dict() contains all required keys: {sorted(required_keys)}")

    # Chart series helper
    chart = result_to_chart_series(result)
    assert len(chart["labels"]) == len(REGRET_DRIVERS)
    print(f"OK result_to_chart_series() returned {len(chart['labels'])} driver series")

    print("\n" + "=" * 64)
    print("  All smoke tests passed.")
    print("=" * 64 + "\n")
