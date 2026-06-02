"""
DecisionPilot AI — modules/analyzer.py
Core AI analysis engine.

Responsibilities:
  - Build and validate all decision-analysis prompts
  - Call the Anthropic API via a shared client
  - Parse, validate, and normalise AI responses
  - Expose clean typed interfaces consumed by app.py and other modules

Public API
----------
  analyse_decision(...)   -> AnalysisResult   (full structured breakdown)
  quick_score(...)        -> QuickScore        (lightweight score + recommendation only)
  validate_inputs(...)    -> list[str]         (list of human-readable errors, empty = OK)

All monetary / financial comparisons are currency-agnostic (the user's own
strings are forwarded to the model verbatim).
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

import anthropic

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────
logger = logging.getLogger("decisionpilot.analyzer")
logger.setLevel(logging.INFO)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
MODEL              = "claude-sonnet-4-20250514"
MAX_TOKENS_FULL    = 2500
MAX_TOKENS_QUICK   = 800
MAX_RETRIES        = 2
RETRY_DELAY_SEC    = 1.5

VALID_RISK_LEVELS  = {"Low", "Medium", "High"}
VALID_CHOICES      = {"A", "B"}
VALID_STRENGTHS    = {"Weak", "Moderate", "Strong", "Very Strong"}

# Category → domain-specific scoring heuristic hints injected into the prompt
CATEGORY_HINTS: dict[str, str] = {
    "Career":          "Weigh role growth, compensation trajectory, skill development, and job security.",
    "Education":       "Prioritise learning quality, credential value, ROI on time and tuition, and network access.",
    "Finance":         "Focus on risk-adjusted returns, liquidity, compounding effects, and downside protection.",
    "Purchase":        "Evaluate utility, total cost of ownership, opportunity cost of capital, and long-term value retention.",
    "Project":         "Consider scope feasibility, resource constraints, learning potential, and portfolio impact.",
    "Internship":      "Balance stipend, mentorship quality, brand value, and conversion probability.",
    "Personal Growth": "Emphasise mindset expansion, habit formation, social capital, and long-term fulfilment.",
    "Other":           "Apply balanced multi-criteria analysis across all relevant life dimensions.",
}

# ─────────────────────────────────────────────────────────────────────────────
# DATA CLASSES  (typed result containers)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class OptionAnalysis:
    """Per-option breakdown returned by the AI."""
    score:            int   = 0
    pros:             list[str] = field(default_factory=list)
    cons:             list[str] = field(default_factory=list)
    risk_level:       str   = "Medium"
    risk_details:     str   = ""
    long_term_impact: str   = ""
    opportunity_cost: str   = ""

    # ── normalise values coming from the model ──
    def __post_init__(self):
        self.score      = _clamp(self.score, 0, 100)
        self.risk_level = self.risk_level if self.risk_level in VALID_RISK_LEVELS else "Medium"
        self.pros       = _ensure_list(self.pros)
        self.cons       = _ensure_list(self.cons)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AnalysisResult:
    """
    Full analysis result returned by analyse_decision().
    All fields have safe defaults so callers never hit AttributeError.
    """
    # Per-option detail
    option_a: OptionAnalysis = field(default_factory=OptionAnalysis)
    option_b: OptionAnalysis = field(default_factory=OptionAnalysis)

    # Aggregate signals
    confidence:               int  = 50
    recommended:              str  = "A"
    recommendation_strength:  str  = "Moderate"
    reasoning:                str  = ""
    summary:                  str  = ""

    # Regret predictor
    short_term_regret_risk:   str  = "Medium"
    long_term_regret_risk:    str  = "Medium"
    regret_reason:            str  = ""

    # Meta
    category:                 str  = ""
    title:                    str  = ""
    elapsed_ms:               int  = 0
    raw_response:             str  = field(default="", repr=False)

    def __post_init__(self):
        self.confidence              = _clamp(self.confidence, 0, 100)
        self.recommended             = self.recommended if self.recommended in VALID_CHOICES else "A"
        self.recommendation_strength = (self.recommendation_strength
                                        if self.recommendation_strength in VALID_STRENGTHS
                                        else "Moderate")
        self.short_term_regret_risk  = (self.short_term_regret_risk
                                        if self.short_term_regret_risk in VALID_RISK_LEVELS
                                        else "Medium")
        self.long_term_regret_risk   = (self.long_term_regret_risk
                                        if self.long_term_regret_risk in VALID_RISK_LEVELS
                                        else "Medium")

    # ── Convenience accessors ──
    @property
    def recommended_option(self) -> OptionAnalysis:
        return self.option_a if self.recommended == "A" else self.option_b

    @property
    def other_option(self) -> OptionAnalysis:
        return self.option_b if self.recommended == "A" else self.option_a

    @property
    def score_delta(self) -> int:
        """How much better the recommended option scored."""
        return abs(self.option_a.score - self.option_b.score)

    @property
    def is_decisive(self) -> bool:
        """True when the score gap is meaningful (≥ 10 points)."""
        return self.score_delta >= 10

    def to_dict(self) -> dict:
        """Serialise to a plain dict (suitable for JSON / CSV)."""
        d = asdict(self)
        d["option_a"] = self.option_a.to_dict()
        d["option_b"] = self.option_b.to_dict()
        return d


@dataclass
class QuickScore:
    """Lightweight result for fast recommendations."""
    option_a_score: int  = 0
    option_b_score: int  = 0
    recommended:    str  = "A"
    confidence:     int  = 50
    one_liner:      str  = ""
    elapsed_ms:     int  = 0

    def __post_init__(self):
        self.option_a_score = _clamp(self.option_a_score, 0, 100)
        self.option_b_score = _clamp(self.option_b_score, 0, 100)
        self.confidence     = _clamp(self.confidence, 0, 100)
        self.recommended    = self.recommended if self.recommended in VALID_CHOICES else "A"

# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL UTILITIES
# ─────────────────────────────────────────────────────────────────────────────
def _clamp(value: Any, lo: int, hi: int) -> int:
    """Coerce value to int and clamp to [lo, hi]."""
    try:
        return max(lo, min(hi, int(value)))
    except (TypeError, ValueError):
        return lo


def _ensure_list(value: Any) -> list[str]:
    """Guarantee we always get a list of strings."""
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _strip_fences(text: str) -> str:
    """Remove markdown code fences that the model sometimes emits."""
    text = text.strip()
    # Match ```json ... ``` or ``` ... ```
    fence_pattern = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)
    m = fence_pattern.match(text)
    if m:
        return m.group(1).strip()
    # Partial — starts with fence but no closing
    if text.startswith("```"):
        lines = text.splitlines()
        # Drop first line (fence opener) and any trailing fence
        inner = lines[1:]
        if inner and inner[-1].strip() == "```":
            inner = inner[:-1]
        return "\n".join(inner).strip()
    return text


def _parse_json(text: str) -> dict:
    """
    Robustly extract a JSON object from model output.
    Tries three strategies in order:
      1. Direct parse after stripping fences
      2. Regex extraction of the outermost { ... }
      3. Return empty dict and log a warning
    """
    cleaned = _strip_fences(text)

    # Strategy 1 — direct
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Strategy 2 — extract first top-level object
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    logger.warning("Could not parse JSON from model response. Raw (first 500 chars): %s", text[:500])
    return {}


def _build_client() -> anthropic.Anthropic:
    """Return a cached Anthropic client (module-level singleton)."""
    return anthropic.Anthropic()          # reads ANTHROPIC_API_KEY from env


_client: Optional[anthropic.Anthropic] = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = _build_client()
    return _client


def _call_api(prompt: str, system: str, max_tokens: int) -> str:
    """
    Call the Anthropic Messages API with automatic retry on transient errors.
    Returns the raw text content of the first content block.
    """
    client = _get_client()
    last_exc: Exception = RuntimeError("No attempt made")

    for attempt in range(1, MAX_RETRIES + 2):           # +2 so range gives MAX_RETRIES retries
        try:
            t0 = time.monotonic()
            response = client.messages.create(
                model=MODEL,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": prompt}],
            )
            elapsed = int((time.monotonic() - t0) * 1000)
            logger.info("API call succeeded in %d ms (attempt %d)", elapsed, attempt)
            return response.content[0].text, elapsed

        except anthropic.RateLimitError as exc:
            logger.warning("Rate limit hit (attempt %d/%d): %s", attempt, MAX_RETRIES + 1, exc)
            last_exc = exc
            time.sleep(RETRY_DELAY_SEC * attempt)

        except anthropic.APIStatusError as exc:
            logger.error("API status error (attempt %d): %s", attempt, exc)
            last_exc = exc
            if attempt <= MAX_RETRIES:
                time.sleep(RETRY_DELAY_SEC)

        except Exception as exc:
            logger.error("Unexpected API error: %s", exc)
            raise

    raise last_exc

# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────────────────────────────────────────────
_SYSTEM_STRATEGIST = """\
You are DecisionPilot AI — a world-class decision strategist, rational advisor, and life coach.

Your analytical framework:
1. CRITICAL THINKING  — Do not take inputs at face value. Challenge assumptions.
2. TRADE-OFF EXPOSURE — Surface hidden costs, opportunity costs, and second-order effects.
3. RISK QUANTIFICATION— Assess probability and magnitude of negative outcomes honestly.
4. GOAL ALIGNMENT     — Every score and recommendation must trace back to the user's stated goal.
5. COGNITIVE BIAS CHECK—Identify and correct for common biases (sunk cost, status quo, optimism).
6. EVIDENCE-BASED     — Ground reasoning in observable patterns, not platitudes.
7. ACTIONABLE CLARITY — End with an unambiguous recommendation the user can act on today.

Output rules (CRITICAL):
- Respond ONLY with valid JSON. Zero markdown. Zero prose outside the JSON object.
- All string values must be grammatically correct, specific, and non-generic.
- Scores must reflect genuine comparative analysis, NOT arbitrary numbers.
- Pros/cons lists must have 3–5 items each, each item ≥ 8 words, specific to the inputs.
"""

# ─────────────────────────────────────────────────────────────────────────────
# PROMPT BUILDERS
# ─────────────────────────────────────────────────────────────────────────────
def _build_full_analysis_prompt(
    title: str,
    option_a: str,
    option_b: str,
    goal: str,
    category: str,
    budget: str,
    time_commitment: str,
    context: str,
) -> str:
    hint = CATEGORY_HINTS.get(category, CATEGORY_HINTS["Other"])

    return f"""\
Perform a complete decision analysis for the following inputs and return the result
as a single JSON object exactly matching the schema below.

━━━ DECISION INPUTS ━━━
Title            : {title}
Option A         : {option_a}
Option B         : {option_b}
Personal Goal    : {goal}
Category         : {category}
Budget / Finance : {budget or "Not specified"}
Time Commitment  : {time_commitment or "Not specified"}
Additional Context: {context or "None provided"}

━━━ CATEGORY GUIDANCE ━━━
{hint}

━━━ REQUIRED JSON SCHEMA ━━━
{{
  "option_a": {{
    "score"           : <integer 0-100, honest comparative score>,
    "pros"            : ["<specific strength ≥8 words>", ...],   // 3-5 items
    "cons"            : ["<specific weakness ≥8 words>", ...],   // 3-5 items
    "risk_level"      : "<Low | Medium | High>",
    "risk_details"    : "<2-3 sentences on specific risks of Option A>",
    "long_term_impact": "<2-3 sentences on where this leads in 3-5 years>",
    "opportunity_cost": "<1-2 sentences on what is sacrificed by choosing A>"
  }},
  "option_b": {{
    "score"           : <integer 0-100>,
    "pros"            : ["<specific strength ≥8 words>", ...],
    "cons"            : ["<specific weakness ≥8 words>", ...],
    "risk_level"      : "<Low | Medium | High>",
    "risk_details"    : "<2-3 sentences on specific risks of Option B>",
    "long_term_impact": "<2-3 sentences on where this leads in 3-5 years>",
    "opportunity_cost": "<1-2 sentences on what is sacrificed by choosing B>"
  }},
  "confidence"              : <integer 0-100, how confident the AI is in its recommendation>,
  "recommended"             : "<A | B>",
  "recommendation_strength" : "<Weak | Moderate | Strong | Very Strong>",
  "reasoning"               : "<4-6 sentences of specific, critical reasoning explaining the recommendation>",
  "summary"                 : "<single punchy sentence summarising the final recommendation>",
  "short_term_regret_risk"  : "<Low | Medium | High>",
  "long_term_regret_risk"   : "<Low | Medium | High>",
  "regret_reason"           : "<2-3 sentences explaining what specific outcome would cause regret and why>",
  "cognitive_biases_detected": ["<bias name: explanation>", ...],  // 1-3 biases the user may be exhibiting
  "key_assumptions"         : ["<assumption the analysis relies on>", ...]  // 2-4 items
}}

Return ONLY the JSON object. No markdown. No explanation outside the JSON.
"""


def _build_quick_score_prompt(
    title: str,
    option_a: str,
    option_b: str,
    goal: str,
    category: str,
) -> str:
    return f"""\
Give a quick decision score for the following. Be direct and decisive.

Decision : {title}
Option A  : {option_a}
Option B  : {option_b}
Goal      : {goal}
Category  : {category}

Return ONLY this JSON:
{{
  "option_a_score": <integer 0-100>,
  "option_b_score": <integer 0-100>,
  "recommended"   : "<A | B>",
  "confidence"    : <integer 0-100>,
  "one_liner"     : "<one punchy sentence explaining the recommendation>"
}}
"""

# ─────────────────────────────────────────────────────────────────────────────
# INPUT VALIDATION
# ─────────────────────────────────────────────────────────────────────────────
def validate_inputs(
    title: str,
    option_a: str,
    option_b: str,
    goal: str,
    budget: str = "",
    time_commitment: str = "",
) -> list[str]:
    """
    Validate user inputs before hitting the API.
    Returns a list of human-readable error strings.
    An empty list means all inputs are valid.
    """
    errors: list[str] = []

    def _check(value: str, field_name: str, min_len: int = 3, max_len: int = 500):
        stripped = value.strip() if value else ""
        if not stripped:
            errors.append(f"'{field_name}' is required.")
        elif len(stripped) < min_len:
            errors.append(f"'{field_name}' is too short (minimum {min_len} characters).")
        elif len(stripped) > max_len:
            errors.append(f"'{field_name}' is too long (maximum {max_len} characters).")

    _check(title,    "Decision Title",   min_len=5,  max_len=200)
    _check(option_a, "Option A",         min_len=3,  max_len=400)
    _check(option_b, "Option B",         min_len=3,  max_len=400)
    _check(goal,     "Personal Goal",    min_len=5,  max_len=400)

    # Options must be meaningfully different
    if option_a.strip().lower() == option_b.strip().lower():
        errors.append("Option A and Option B appear to be identical. Please enter two distinct choices.")

    # Optional field length guards
    if budget and len(budget.strip()) > 200:
        errors.append("'Budget' is too long (maximum 200 characters).")
    if time_commitment and len(time_commitment.strip()) > 200:
        errors.append("'Time Commitment' is too long (maximum 200 characters).")

    return errors

# ─────────────────────────────────────────────────────────────────────────────
# RESULT BUILDERS  (raw dict → typed dataclass)
# ─────────────────────────────────────────────────────────────────────────────
def _build_option_analysis(raw: dict) -> OptionAnalysis:
    return OptionAnalysis(
        score            = raw.get("score", 0),
        pros             = raw.get("pros", []),
        cons             = raw.get("cons", []),
        risk_level       = raw.get("risk_level", "Medium"),
        risk_details     = raw.get("risk_details", ""),
        long_term_impact = raw.get("long_term_impact", ""),
        opportunity_cost = raw.get("opportunity_cost", ""),
    )


def _build_analysis_result(
    raw: dict,
    *,
    title: str,
    category: str,
    elapsed_ms: int,
    raw_response: str,
) -> AnalysisResult:
    return AnalysisResult(
        option_a                  = _build_option_analysis(raw.get("option_a", {})),
        option_b                  = _build_option_analysis(raw.get("option_b", {})),
        confidence                = raw.get("confidence", 50),
        recommended               = raw.get("recommended", "A"),
        recommendation_strength   = raw.get("recommendation_strength", "Moderate"),
        reasoning                 = raw.get("reasoning", ""),
        summary                   = raw.get("summary", ""),
        short_term_regret_risk    = raw.get("short_term_regret_risk", "Medium"),
        long_term_regret_risk     = raw.get("long_term_regret_risk", "Medium"),
        regret_reason             = raw.get("regret_reason", ""),
        category                  = category,
        title                     = title,
        elapsed_ms                = elapsed_ms,
        raw_response              = raw_response,
    )


def _fallback_result(title: str, category: str) -> AnalysisResult:
    """
    Return a safe fallback result when the API call or parse fails.
    Callers should always check result.reasoning for empty string as a signal.
    """
    logger.error("Returning fallback AnalysisResult due to API/parse failure.")
    return AnalysisResult(
        title    = title,
        category = category,
        reasoning= (
            "The analysis could not be completed. This may be due to a network issue, "
            "an API error, or an unexpected model response. Please try again."
        ),
        summary  = "Analysis unavailable — please retry.",
    )

# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────
def analyse_decision(
    title:           str,
    option_a:        str,
    option_b:        str,
    goal:            str,
    category:        str  = "Other",
    budget:          str  = "",
    time_commitment: str  = "",
    context:         str  = "",
) -> AnalysisResult:
    """
    Perform a full AI-powered decision analysis.

    Parameters
    ----------
    title           : Short description of the decision (e.g. "Accept the job offer?")
    option_a        : First choice (e.g. "Join the startup in Bengaluru")
    option_b        : Second choice (e.g. "Stay at current company")
    goal            : User's personal goal that the decision should serve
    category        : One of the DECISION_CATEGORIES strings
    budget          : Optional financial context (free-form string)
    time_commitment : Optional time context (free-form string)
    context         : Optional additional background

    Returns
    -------
    AnalysisResult  : Fully populated dataclass. Never raises — returns a
                      fallback result on API/parse failure.
    """
    logger.info("Starting full analysis for: '%s'", title)

    prompt = _build_full_analysis_prompt(
        title, option_a, option_b, goal,
        category, budget, time_commitment, context,
    )

    try:
        raw_text, elapsed_ms = _call_api(prompt, _SYSTEM_STRATEGIST, MAX_TOKENS_FULL)
    except Exception as exc:
        logger.error("API call failed: %s", exc)
        return _fallback_result(title, category)

    parsed = _parse_json(raw_text)
    if not parsed:
        return _fallback_result(title, category)

    result = _build_analysis_result(
        parsed,
        title       = title,
        category    = category,
        elapsed_ms  = elapsed_ms,
        raw_response= raw_text,
    )

    logger.info(
        "Analysis complete — Recommended: Option %s | Confidence: %d%% | Δscore: %d | %.0f ms",
        result.recommended, result.confidence, result.score_delta, elapsed_ms,
    )
    return result


def quick_score(
    title:    str,
    option_a: str,
    option_b: str,
    goal:     str,
    category: str = "Other",
) -> QuickScore:
    """
    Fast scoring endpoint — lower latency, lower token cost.
    Use for previews, loading states, or lightweight recommendations.

    Returns
    -------
    QuickScore : Scored result with a one-liner recommendation.
    """
    logger.info("Starting quick score for: '%s'", title)
    prompt = _build_quick_score_prompt(title, option_a, option_b, goal, category)

    try:
        raw_text, elapsed_ms = _call_api(prompt, _SYSTEM_STRATEGIST, MAX_TOKENS_QUICK)
    except Exception as exc:
        logger.error("Quick score API call failed: %s", exc)
        return QuickScore(one_liner="Quick score unavailable. Please try the full analysis.")

    parsed = _parse_json(raw_text)
    if not parsed:
        return QuickScore(one_liner="Could not parse quick score. Please try the full analysis.")

    return QuickScore(
        option_a_score = parsed.get("option_a_score", 0),
        option_b_score = parsed.get("option_b_score", 0),
        recommended    = parsed.get("recommended", "A"),
        confidence     = parsed.get("confidence", 50),
        one_liner      = parsed.get("one_liner", ""),
        elapsed_ms     = elapsed_ms,
    )


# ─────────────────────────────────────────────────────────────────────────────
# BACKWARD-COMPAT SHIM  (keeps app.py's analyse_decision() call working)
# ─────────────────────────────────────────────────────────────────────────────
def analyze_decision(
    title:           str,
    option_a:        str,
    option_b:        str,
    goal:            str,
    category:        str  = "Other",
    budget:          str  = "",
    time_commitment: str  = "",
    context:         str  = "",
) -> dict:
    """
    Thin shim that calls analyse_decision() and returns a plain dict.
    Used by app.py for compatibility with the existing render logic.
    """
    result = analyse_decision(
        title, option_a, option_b, goal,
        category, budget, time_commitment, context,
    )
    return result.to_dict()


# ─────────────────────────────────────────────────────────────────────────────
# MANUAL SMOKE-TEST  (run: python -m modules.analyzer)
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(levelname)s | %(name)s | %(message)s")

    print("\n" + "═"*60)
    print("  DecisionPilot AI — analyzer.py smoke test")
    print("═"*60 + "\n")

    # Validation test
    errs = validate_inputs("", "A", "A", "")
    assert len(errs) >= 3, f"Expected ≥3 validation errors, got {len(errs)}"
    print(f"✅ validate_inputs() returned {len(errs)} errors as expected.\n")

    # Quick score
    qs = quick_score(
        title    = "Should I learn Rust or Go this quarter?",
        option_a = "Learn Rust — systems programming, steep learning curve",
        option_b = "Learn Go — backend services, gentler learning curve",
        goal     = "Become a backend engineer at a top product company",
        category = "Education",
    )
    print(f"✅ quick_score() → Recommended: Option {qs.recommended} | "
          f"Confidence: {qs.confidence}% | {qs.one_liner}\n")

    # Full analysis
    result = analyse_decision(
        title           = "Should I accept the startup offer or stay at my MNC?",
        option_a        = "Join Series-A startup as SDE-2, ₹18 LPA + 0.1% equity",
        option_b        = "Stay at TCS as Senior Engineer, ₹14 LPA, stable role",
        goal            = "Reach ₹30 LPA TC and senior IC role within 3 years",
        category        = "Career",
        budget          = "₹6 months of living expenses saved",
        time_commitment = "Open to 2 years minimum commitment",
        context         = "I have 3 YOE, strong Python/AWS skills, no family dependants",
    )
    print(f"✅ analyse_decision() complete in {result.elapsed_ms} ms")
    print(f"   Option A score : {result.option_a.score}")
    print(f"   Option B score : {result.option_b.score}")
    print(f"   Recommended    : Option {result.recommended} ({result.recommendation_strength})")
    print(f"   Confidence     : {result.confidence}%")
    print(f"   Summary        : {result.summary}")
    print(f"   Is decisive    : {result.is_decisive}")
    print(f"   Score delta    : {result.score_delta} pts\n")
    print("═"*60)
    print("  All smoke tests passed.")
    print("═"*60 + "\n")
