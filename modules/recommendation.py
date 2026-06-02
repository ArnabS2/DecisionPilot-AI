"""
DecisionPilot AI — modules/recommendation.py
Final recommendation synthesis engine.

Responsibilities
----------------
  - Consume outputs from analyzer.py, scorer.py, and future_simulator.py
  - Synthesise all signals into a single, authoritative recommendation
  - Compute a final confidence score from a weighted multi-signal blend
  - Derive recommendation_strength from the confidence score algorithmically
  - Generate final_reasoning and summary via a focused AI call
  - Return the exact output spec as a plain Python dict and a typed dataclass

Why a separate recommendation module?
--------------------------------------
Each upstream module answers a narrow question:
  - analyzer.py   → "Which option is objectively better, and why?"
  - scorer.py     → "How do the options compare across six measurable dimensions?"
  - future_simulator.py → "What does life actually look like after the choice?"

recommendation.py asks the *meta* question:
  "Given everything we know, what is the single best action for this specific
  person, and how confident should they be in that action?"

This requires cross-signal reasoning that no single upstream module has full
visibility into. It is intentionally the last step in the pipeline.

Confidence blending
--------------------
The final confidence_score is a weighted blend of four upstream signals:

  Signal                        Weight   Source
  ─────────────────────────────────────────────────────────
  Analyzer confidence            0.35    analyzer output["confidence"]
  Scorer confidence              0.25    scorer output["confidence_percentage"]
  Score gap magnitude (A vs B)   0.20    scorer output["score_gap"], sigmoid-mapped
  Future simulation completeness 0.20    simulation output coverage quality

Total weight = 1.00. Each signal is normalised to 0-100 before blending.
The blend is then adjusted by a consistency bonus (all signals agree → +5)
and a conflict penalty (signals disagree on which option to pick → -8).

Recommendation strength thresholds
------------------------------------
  90-100 → Very Strong
  75-89  → Strong
  58-74  → Moderate
  0-57   → Weak

Public API
----------
  build_recommendation(...)      -> RecommendationResult  (typed dataclass)
  build_recommendation_dict(...) -> dict                  (plain dict, exact output spec)

Input types accepted (all three inputs are plain dicts)
--------------------------------------------------------
  analyzer_output   : dict   — from analyzer.analyse_decision() .to_dict()
                               OR from app.py's analyze_decision() shim
  scorer_output     : dict   — from scorer.score_options_dict()
  simulation_output : dict   — from future_simulator.simulate_future_dict()
                               Pass {} to skip simulation signal.
"""

from __future__ import annotations

import json
import logging
import math
import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import anthropic

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────
logger = logging.getLogger("decisionpilot.recommendation")
logger.setLevel(logging.INFO)

# ─────────────────────────────────────────────────────────────────────────────
# MODEL CONFIG  (mirrors analyzer.py exactly)
# ─────────────────────────────────────────────────────────────────────────────
MODEL           = "claude-sonnet-4-20250514"
MAX_TOKENS      = 1800
MAX_RETRIES     = 2
RETRY_DELAY_SEC = 1.5

# ─────────────────────────────────────────────────────────────────────────────
# STRENGTH THRESHOLDS
# ─────────────────────────────────────────────────────────────────────────────
# Maps final confidence_score (0-100) to a recommendation_strength label.
# Ordered highest-first so the first match wins.
STRENGTH_THRESHOLDS: list[tuple[int, str]] = [
    (90, "Very Strong"),
    (75, "Strong"),
    (58, "Moderate"),
    (0,  "Weak"),
]

# Valid strength labels — consistent with analyzer.py's VALID_STRENGTHS
VALID_STRENGTHS: frozenset[str] = frozenset({"Weak", "Moderate", "Strong", "Very Strong"})

# Valid option labels
VALID_OPTIONS: frozenset[str] = frozenset({"A", "B"})

# ─────────────────────────────────────────────────────────────────────────────
# CONFIDENCE BLEND WEIGHTS
# ─────────────────────────────────────────────────────────────────────────────
# Must sum to 1.0. Change weights here to rebalance signal importance.
W_ANALYZER_CONFIDENCE    = 0.35
W_SCORER_CONFIDENCE      = 0.25
W_SCORE_GAP              = 0.20   # mapped through sigmoid before blending
W_SIMULATION_QUALITY     = 0.20   # 0 if no simulation, full weight if complete

_WEIGHT_SUM = (
    W_ANALYZER_CONFIDENCE +
    W_SCORER_CONFIDENCE   +
    W_SCORE_GAP           +
    W_SIMULATION_QUALITY
)
assert abs(_WEIGHT_SUM - 1.0) < 1e-9, f"Blend weights must sum to 1.0, got {_WEIGHT_SUM}"

# Consistency bonus/penalty applied after the weighted blend
CONSISTENCY_BONUS   =  5   # added when all signals agree on which option wins
CONFLICT_PENALTY    = -8   # applied when scorer and analyzer disagree on winner

# ─────────────────────────────────────────────────────────────────────────────
# DATA CLASS
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class RecommendationResult:
    """
    Final synthesised recommendation.

    All fields have safe defaults so callers never hit AttributeError.
    The five required output keys map exactly to the spec.
    """
    # ── Required outputs (spec) ──────────────────────────────────────────────
    recommended_option:       str = ""     # e.g. "Option A" or full option text
    confidence_score:         int = 50     # 0-100, blended from all signals
    recommendation_strength:  str = "Moderate"  # Weak / Moderate / Strong / Very Strong
    final_reasoning:          str = ""     # AI-generated multi-signal synthesis
    summary:                  str = ""     # one punchy sentence

    # ── Provenance — echoed from inputs for traceability ────────────────────
    option_a_label:           str = ""
    option_b_label:           str = ""
    analyzer_recommendation:  str = ""     # "A" or "B"
    scorer_leading:           str = ""     # "A" or "B"
    signals_agree:            bool = True

    # ── Intermediate confidence components (useful for debug / viz) ──────────
    analyzer_confidence:      int = 50
    scorer_confidence:        int = 50
    score_gap:                int = 0
    simulation_quality:       int = 0      # 0-100

    # ── Meta ─────────────────────────────────────────────────────────────────
    elapsed_ms:               int = 0
    raw_response:             str = field(default="", repr=False)

    def __post_init__(self) -> None:
        self.confidence_score        = _clamp(self.confidence_score, 0, 100)
        self.analyzer_confidence     = _clamp(self.analyzer_confidence, 0, 100)
        self.scorer_confidence       = _clamp(self.scorer_confidence, 0, 100)
        self.score_gap               = _clamp(self.score_gap, 0, 100)
        self.simulation_quality      = _clamp(self.simulation_quality, 0, 100)
        self.recommendation_strength = (
            self.recommendation_strength
            if self.recommendation_strength in VALID_STRENGTHS
            else "Moderate"
        )

    @property
    def winning_option_key(self) -> str:
        """Return 'A' or 'B' derived from the recommended_option string."""
        if "option a" in self.recommended_option.lower():
            return "A"
        if "option b" in self.recommended_option.lower():
            return "B"
        # Fall back to analyzer recommendation
        return self.analyzer_recommendation or "A"

    @property
    def is_decisive(self) -> bool:
        """True when confidence_score is at or above the Strong threshold."""
        return self.confidence_score >= 75

    def to_dict(self) -> dict:
        """
        Return the canonical plain dict matching the required output spec.

        Guaranteed top-level keys
        -------------------------
        recommended_option      : str
        confidence_score        : int   (0-100)
        recommendation_strength : str
        final_reasoning         : str
        summary                 : str
        -- plus provenance / debug enrichment --
        option_a_label          : str
        option_b_label          : str
        analyzer_recommendation : str
        scorer_leading          : str
        signals_agree           : bool
        analyzer_confidence     : int
        scorer_confidence       : int
        score_gap               : int
        simulation_quality      : int
        is_decisive             : bool
        elapsed_ms              : int
        """
        return {
            # Required outputs (spec)
            "recommended_option":       self.recommended_option,
            "confidence_score":         self.confidence_score,
            "recommendation_strength":  self.recommendation_strength,
            "final_reasoning":          self.final_reasoning,
            "summary":                  self.summary,
            # Provenance / enrichment
            "option_a_label":           self.option_a_label,
            "option_b_label":           self.option_b_label,
            "analyzer_recommendation":  self.analyzer_recommendation,
            "scorer_leading":           self.scorer_leading,
            "signals_agree":            self.signals_agree,
            "analyzer_confidence":      self.analyzer_confidence,
            "scorer_confidence":        self.scorer_confidence,
            "score_gap":                self.score_gap,
            "simulation_quality":       self.simulation_quality,
            "is_decisive":              self.is_decisive,
            "elapsed_ms":               self.elapsed_ms,
        }


# ─────────────────────────────────────────────────────────────────────────────
# ANTHROPIC CLIENT  (module-level singleton — same pattern as analyzer.py)
# ─────────────────────────────────────────────────────────────────────────────
_client: Optional[anthropic.Anthropic] = None


def _get_client() -> anthropic.Anthropic:
    """Return a cached Anthropic client. Reads ANTHROPIC_API_KEY from env."""
    global _client
    if _client is None:
        _client = anthropic.Anthropic()
    return _client


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL UTILITIES  (identical interface to analyzer.py)
# ─────────────────────────────────────────────────────────────────────────────
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
    Robustly extract a JSON object from model output.

    Three-stage strategy (mirrors analyzer.py):
      1. Direct parse after fence stripping
      2. Regex extraction of the outermost { ... }
      3. Return {} and log a warning — caller handles fallback
    """
    cleaned = _strip_fences(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    logger.warning(
        "recommendation: could not parse JSON. First 500 chars: %s", text[:500]
    )
    return {}


def _call_api(prompt: str, system: str, max_tokens: int) -> tuple[str, int]:
    """
    Call the Anthropic Messages API with automatic retry on transient errors.

    Signature matches analyzer.py exactly:
      _call_api(prompt, system, max_tokens) -> (raw_text, elapsed_ms)

    Raises the last caught exception if all retries are exhausted.
    """
    client    = _get_client()
    last_exc: Exception = RuntimeError("_call_api: no attempt was made")

    for attempt in range(1, MAX_RETRIES + 2):
        try:
            t0 = time.monotonic()
            response = client.messages.create(
                model      = MODEL,
                max_tokens = max_tokens,
                system     = system,
                messages   = [{"role": "user", "content": prompt}],
            )
            elapsed_ms = int((time.monotonic() - t0) * 1000)
            logger.info(
                "recommendation: API ok in %d ms (attempt %d)", elapsed_ms, attempt
            )
            return response.content[0].text, elapsed_ms

        except anthropic.RateLimitError as exc:
            logger.warning(
                "recommendation: rate limit (attempt %d/%d): %s",
                attempt, MAX_RETRIES + 1, exc,
            )
            last_exc = exc
            time.sleep(RETRY_DELAY_SEC * attempt)

        except anthropic.APIStatusError as exc:
            logger.error(
                "recommendation: API status error (attempt %d): %s", attempt, exc
            )
            last_exc = exc
            if attempt <= MAX_RETRIES:
                time.sleep(RETRY_DELAY_SEC)

        except Exception as exc:
            logger.error("recommendation: unexpected API error: %s", exc)
            raise

    raise last_exc


# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────────────────────────────────────────────
_SYSTEM_RECOMMENDER = """\
You are DecisionPilot AI's Final Recommendation Engine — the last and most
authoritative voice in a multi-stage decision analysis pipeline.

By the time you speak, three specialist modules have already run:
  1. The Analyzer   — assessed both options on pros, cons, risk, and opportunity cost.
  2. The Scorer     — quantified each option across six weighted dimensions.
  3. The Simulator  — projected concrete life outcomes at 1 month, 6 months, and 1 year.

Your job is not to redo their work. Your job is to synthesise their findings
into a single, decisive recommendation the user can act on today.

Synthesis principles:
  1. CROSS-SIGNAL COHERENCE — Look for where all three modules agree. That
     agreement is the strongest evidence. Flag any contradictions honestly.
  2. GOAL SUPREMACY — The user's stated goal is the ultimate tiebreaker. If
     Option B scores higher on five dimensions but Option A is the only one
     that plausibly achieves the goal, recommend A.
  3. ASYMMETRIC RISK WEIGHTING — A catastrophic downside outweighs many
     moderate upsides. If one option has a high-severity failure mode, say so.
  4. COGNITIVE BIAS AUDIT — Name any bias that may be colouring the user's
     framing or the data provided (anchoring, status quo, loss aversion).
  5. DECISIVE CLARITY — End with an unambiguous recommendation. "It depends"
     is not acceptable. If the evidence is genuinely mixed, say Moderate
     strength but still pick a winner.

Output rules (CRITICAL):
  - Respond ONLY with valid JSON. No markdown. No prose outside the JSON.
  - final_reasoning must be 4-6 sentences, specific to the inputs, cross-referencing
    at least two of the three upstream signals explicitly.
  - summary must be exactly one sentence, punchy, and actionable.
"""

# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL EXTRACTION  (safe reading from upstream dicts)
# ─────────────────────────────────────────────────────────────────────────────
def _safe_str(d: dict, *keys: str, default: str = "") -> str:
    """
    Walk a chain of keys into a nested dict, returning default on any miss.
    Accepts multiple keys tried in order (first hit wins).
    """
    for key in keys:
        val = d.get(key)
        if val is not None:
            return str(val).strip()
    return default


def _safe_int(d: dict, *keys: str, default: int = 0) -> int:
    """Walk a chain of keys, coercing the first hit to int."""
    for key in keys:
        val = d.get(key)
        if val is not None:
            try:
                return int(val)
            except (TypeError, ValueError):
                continue
    return default


def _safe_list(d: dict, key: str) -> list:
    """Return a list from d[key], or [] if missing / wrong type."""
    val = d.get(key, [])
    return val if isinstance(val, list) else []


def _extract_analyzer_signals(analyzer_output: dict) -> dict:
    """
    Pull all decision-relevant signals from an analyzer output dict.

    Compatible with both:
      - analyzer.AnalysisResult.to_dict()          (typed dataclass output)
      - app.py's analyze_decision() shim            (same structure)

    Returns a flat dict of normalised signal values.
    """
    opt_a = analyzer_output.get("option_a", {}) or {}
    opt_b = analyzer_output.get("option_b", {}) or {}

    return {
        "analyzer_confidence":      _safe_int(analyzer_output, "confidence", default=50),
        "analyzer_recommended":     _safe_str(analyzer_output, "recommended", default="A").upper(),
        "analyzer_strength":        _safe_str(analyzer_output, "recommendation_strength", default="Moderate"),
        "analyzer_reasoning":       _safe_str(analyzer_output, "reasoning"),
        "analyzer_summary":         _safe_str(analyzer_output, "summary"),
        "option_a_score":           _safe_int(opt_a, "score", default=0),
        "option_b_score":           _safe_int(opt_b, "score", default=0),
        "option_a_risk":            _safe_str(opt_a, "risk_level", default="Medium"),
        "option_b_risk":            _safe_str(opt_b, "risk_level", default="Medium"),
        "option_a_long_term":       _safe_str(opt_a, "long_term_impact"),
        "option_b_long_term":       _safe_str(opt_b, "long_term_impact"),
        "short_term_regret_risk":   _safe_str(analyzer_output, "short_term_regret_risk", default="Medium"),
        "long_term_regret_risk":    _safe_str(analyzer_output, "long_term_regret_risk", default="Medium"),
        "category":                 _safe_str(analyzer_output, "category"),
        "title":                    _safe_str(analyzer_output, "title"),
    }


def _extract_scorer_signals(scorer_output: dict) -> dict:
    """
    Pull all decision-relevant signals from a scorer output dict.

    Compatible with scorer.ScoringResult.to_dict() / score_options_dict().
    """
    opt_a = scorer_output.get("option_a", {}) or {}
    opt_b = scorer_output.get("option_b", {}) or {}

    return {
        "scorer_confidence":        _safe_int(scorer_output, "confidence_percentage", default=50),
        "scorer_score_a":           _safe_int(scorer_output, "score_a", default=0),
        "scorer_score_b":           _safe_int(scorer_output, "score_b", default=0),
        "scorer_gap":               _safe_int(scorer_output, "score_gap", default=0),
        "scorer_leading":           _safe_str(scorer_output, "leading", default="A").upper(),
        "scorer_is_decisive":       bool(scorer_output.get("is_decisive", False)),
        "option_a_strongest_dim":   _safe_str(opt_a, "strongest_dim"),
        "option_b_strongest_dim":   _safe_str(opt_b, "strongest_dim"),
        "option_a_weakest_dim":     _safe_str(opt_a, "weakest_dim"),
        "option_b_weakest_dim":     _safe_str(opt_b, "weakest_dim"),
    }


def _extract_simulation_signals(simulation_output: dict) -> dict:
    """
    Pull all decision-relevant signals from a simulation output dict.

    Compatible with future_simulator.SimulationResult.to_dict() /
    simulate_future_dict(). Returns zeros if simulation_output is empty.
    """
    if not simulation_output:
        return {
            "sim_available":    False,
            "sim_1_month":      "",
            "sim_6_months":     "",
            "sim_1_year":       "",
            "sim_growth":       "",
            "sim_challenges":   "",
            "sim_benefits":     [],
            "sim_risks":        [],
            "sim_quality":      0,
        }

    # Quality score: how complete is the simulation? (0-100)
    filled = sum([
        bool(_safe_str(simulation_output, "after_1_month")),
        bool(_safe_str(simulation_output, "after_6_months")),
        bool(_safe_str(simulation_output, "after_1_year")),
        bool(_safe_str(simulation_output, "growth")),
        bool(_safe_str(simulation_output, "challenges")),
        bool(_safe_list(simulation_output, "benefits")),
        bool(_safe_list(simulation_output, "risks")),
    ])
    quality = round((filled / 7) * 100)

    return {
        "sim_available":    True,
        "sim_1_month":      _safe_str(simulation_output, "after_1_month"),
        "sim_6_months":     _safe_str(simulation_output, "after_6_months"),
        "sim_1_year":       _safe_str(simulation_output, "after_1_year"),
        "sim_growth":       _safe_str(simulation_output, "growth"),
        "sim_challenges":   _safe_str(simulation_output, "challenges"),
        "sim_benefits":     _safe_list(simulation_output, "benefits"),
        "sim_risks":        _safe_list(simulation_output, "risks"),
        "sim_quality":      quality,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CONFIDENCE BLENDING
# ─────────────────────────────────────────────────────────────────────────────
def _sigmoid_gap(gap: int) -> float:
    """
    Map a score gap (0-100) onto (0, 1) using a sigmoid curve.

    Calibration:
      gap=5  → 0.21   (small gap — low confidence contribution)
      gap=10 → 0.38   (moderate)
      gap=20 → 0.62   (meaningful gap)
      gap=30 → 0.78   (large gap — high confidence)
      gap=50 → 0.92   (dominant)

    Uses  f(x) = 1 / (1 + e^(-(x-15)/8))
    so the inflection point is at gap=15 (equal weighting between low/high).
    """
    return 1.0 / (1.0 + math.exp(-(gap - 15) / 8.0))


def _blend_confidence(
    analyzer_signals:   dict,
    scorer_signals:     dict,
    sim_signals:        dict,
) -> tuple[int, bool]:
    """
    Compute the final blended confidence score and the signals_agree flag.

    Algorithm
    ---------
    1. Extract each signal's contribution to 0-100.
    2. Apply weights (W_*) to produce a raw weighted sum.
    3. Apply consistency_bonus if all signals agree on the winner.
    4. Apply conflict_penalty if scorer and analyzer disagree on winner.
    5. Clamp to [0, 100].

    Returns
    -------
    (confidence_score: int, signals_agree: bool)
    """
    # Normalise simulation quality: 0 if unavailable, else use its quality score
    sim_quality_norm = sim_signals["sim_quality"] if sim_signals["sim_available"] else 0

    # Sigmoid-mapped gap contribution (0-100 range)
    gap_contribution = _sigmoid_gap(scorer_signals["scorer_gap"]) * 100

    # Weighted blend
    raw = (
        W_ANALYZER_CONFIDENCE * analyzer_signals["analyzer_confidence"] +
        W_SCORER_CONFIDENCE   * scorer_signals["scorer_confidence"]     +
        W_SCORE_GAP           * gap_contribution                        +
        W_SIMULATION_QUALITY  * sim_quality_norm
    )

    # Determine if all signals agree on the same winner
    analyzer_wins = analyzer_signals["analyzer_recommended"]  # "A" or "B"
    scorer_wins   = scorer_signals["scorer_leading"]          # "A" or "B"
    signals_agree = (analyzer_wins == scorer_wins)

    # Consistency bonus / conflict penalty
    if signals_agree:
        raw += CONSISTENCY_BONUS
        logger.debug("recommendation: signals agree on Option %s → +%d bonus", analyzer_wins, CONSISTENCY_BONUS)
    else:
        raw += CONFLICT_PENALTY
        logger.debug(
            "recommendation: signals conflict — analyzer=%s scorer=%s → %d penalty",
            analyzer_wins, scorer_wins, CONFLICT_PENALTY,
        )

    return _clamp(round(raw), 0, 100), signals_agree


def _confidence_to_strength(confidence: int) -> str:
    """
    Derive recommendation_strength label from the blended confidence score.

    Thresholds (defined in STRENGTH_THRESHOLDS at module top):
      90-100 → Very Strong
      75-89  → Strong
      58-74  → Moderate
      0-57   → Weak
    """
    for threshold, label in STRENGTH_THRESHOLDS:
        if confidence >= threshold:
            return label
    return "Weak"


# ─────────────────────────────────────────────────────────────────────────────
# RECOMMENDED OPTION LABEL  (human-readable, not just "A" or "B")
# ─────────────────────────────────────────────────────────────────────────────
def _resolve_recommended_option_label(
    winning_key:      str,   # "A" or "B"
    option_a_label:   str,
    option_b_label:   str,
) -> str:
    """
    Build the human-readable recommended_option string.

    Format:  "Option A — {option_a_label}"
             "Option B — {option_b_label}"

    Falls back to "Option A" / "Option B" if labels are empty.
    """
    if winning_key == "A":
        label = option_a_label.strip()
        return f"Option A — {label}" if label else "Option A"
    label = option_b_label.strip()
    return f"Option B — {label}" if label else "Option B"


# ─────────────────────────────────────────────────────────────────────────────
# PROMPT BUILDER
# ─────────────────────────────────────────────────────────────────────────────
def _build_reasoning_prompt(
    analyzer_signals: dict,
    scorer_signals:   dict,
    sim_signals:      dict,
    option_a_label:   str,
    option_b_label:   str,
    winning_key:      str,
    confidence:       int,
    strength:         str,
    signals_agree:    bool,
) -> str:
    """
    Build the prompt that asks the AI to write final_reasoning and summary.

    The prompt deliberately provides the already-decided winning option and
    confidence so the AI focuses purely on *explaining* — not re-deciding.
    This keeps the reasoning grounded in the blended signal evidence, not
    in whatever the AI would pick independently.
    """
    winning_label = _resolve_recommended_option_label(winning_key, option_a_label, option_b_label)

    # Inline simulation highlights (first 200 chars of each checkpoint if available)
    if sim_signals["sim_available"]:
        sim_block = (
            f"  1 Month  : {sim_signals['sim_1_month'][:200]}\n"
            f"  6 Months : {sim_signals['sim_6_months'][:200]}\n"
            f"  1 Year   : {sim_signals['sim_1_year'][:200]}\n"
            f"  Growth   : {sim_signals['sim_growth'][:150]}\n"
        )
    else:
        sim_block = "  (Future simulation not available for this analysis)\n"

    conflict_note = (
        "NOTE: The analyzer and scorer disagree on the best option. "
        "Acknowledge this tension in your reasoning and explain why the "
        f"final decision still favours {winning_label}."
        if not signals_agree else ""
    )

    return f"""You are writing the final recommendation text for a completed decision analysis.
The winning option and confidence score have already been determined algorithmically.
Your ONLY job is to write the final_reasoning and summary fields.

WINNING OPTION    : {winning_label}
CONFIDENCE SCORE  : {confidence}/100
STRENGTH          : {strength}
SIGNALS AGREE     : {"Yes" if signals_agree else "No — see conflict note below"}
{conflict_note}

UPSTREAM SIGNALS
────────────────

Analyzer Module
  Recommended     : Option {analyzer_signals['analyzer_recommended']}
  Confidence      : {analyzer_signals['analyzer_confidence']}%
  Strength        : {analyzer_signals['analyzer_strength']}
  Option A score  : {analyzer_signals['option_a_score']}/100
  Option B score  : {analyzer_signals['option_b_score']}/100
  Option A risk   : {analyzer_signals['option_a_risk']}
  Option B risk   : {analyzer_signals['option_b_risk']}
  ST Regret Risk  : {analyzer_signals['short_term_regret_risk']}
  LT Regret Risk  : {analyzer_signals['long_term_regret_risk']}
  Reasoning       : {analyzer_signals['analyzer_reasoning'][:300]}

Scorer Module
  Leading option  : Option {scorer_signals['scorer_leading']}
  Score A         : {scorer_signals['scorer_score_a']}/100
  Score B         : {scorer_signals['scorer_score_b']}/100
  Gap             : {scorer_signals['scorer_gap']} pts  (decisive: {scorer_signals['scorer_is_decisive']})
  A strongest dim : {scorer_signals['option_a_strongest_dim']}
  B strongest dim : {scorer_signals['option_b_strongest_dim']}
  A weakest dim   : {scorer_signals['option_a_weakest_dim']}
  B weakest dim   : {scorer_signals['option_b_weakest_dim']}

Future Simulator
{sim_block}

REQUIRED JSON SCHEMA
{{
  "final_reasoning": "<4-6 sentences. Cross-reference at least 2 upstream signals explicitly. Be specific — name dimensions, scores, regret risks, timeline outcomes. Explain WHY {winning_label} wins despite any trade-offs. If signals conflict, address the tension directly.>",
  "summary": "<Exactly 1 sentence. Punchy, actionable, specific. Start with the winning option name.>"
}}

Return ONLY the JSON object. No markdown. No text outside the JSON.
"""


# ─────────────────────────────────────────────────────────────────────────────
# FALLBACK RESULT
# ─────────────────────────────────────────────────────────────────────────────
def _fallback_result(
    winning_key:    str,
    option_a_label: str,
    option_b_label: str,
    confidence:     int,
    strength:       str,
    signals_agree:  bool,
    analyzer_signals: dict,
    scorer_signals:   dict,
    sim_signals:      dict,
) -> RecommendationResult:
    """
    Return a valid RecommendationResult when the AI reasoning call fails.

    The recommendation is still computed algorithmically — only the
    AI-written reasoning and summary are replaced with safe fallback text.
    """
    logger.error("recommendation: using fallback reasoning due to API/parse failure.")
    recommended_label = _resolve_recommended_option_label(winning_key, option_a_label, option_b_label)
    return RecommendationResult(
        recommended_option       = recommended_label,
        confidence_score         = confidence,
        recommendation_strength  = strength,
        final_reasoning          = (
            f"Based on a multi-signal analysis, {recommended_label} is the stronger choice. "
            f"The analyzer reported {analyzer_signals['analyzer_confidence']}% confidence, "
            f"the scorer showed a {scorer_signals['scorer_gap']}-point gap in its favour, "
            f"and all available evidence aligns with this direction. "
            "Detailed AI reasoning was unavailable due to a network issue — please retry."
        ),
        summary                  = (
            f"{recommended_label} is the recommended path — retry for full reasoning."
        ),
        option_a_label           = option_a_label,
        option_b_label           = option_b_label,
        analyzer_recommendation  = analyzer_signals.get("analyzer_recommended", "A"),
        scorer_leading           = scorer_signals.get("scorer_leading", "A"),
        signals_agree            = signals_agree,
        analyzer_confidence      = analyzer_signals.get("analyzer_confidence", 50),
        scorer_confidence        = scorer_signals.get("scorer_confidence", 50),
        score_gap                = scorer_signals.get("scorer_gap", 0),
        simulation_quality       = sim_signals.get("sim_quality", 0),
    )


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────
def build_recommendation(
    analyzer_output:   dict,
    scorer_output:     dict,
    simulation_output: dict,
    option_a_label:    str = "",
    option_b_label:    str = "",
) -> RecommendationResult:
    """
    Synthesise outputs from all three upstream modules into a final recommendation.

    Parameters
    ----------
    analyzer_output : dict
        Output from analyzer.analyze_decision() or analyzer.analyse_decision().to_dict().
        Required keys: confidence, recommended, option_a.score, option_b.score, etc.

    scorer_output : dict
        Output from scorer.score_options_dict() or scorer.score_options().to_dict().
        Required keys: score_a, score_b, confidence_percentage, score_gap, leading.

    simulation_output : dict
        Output from future_simulator.simulate_future_dict() or .to_dict().
        Pass {} to skip the simulation signal (lowers confidence contribution by 20%).

    option_a_label : str, optional
        Human-readable label for Option A (e.g. "Join the startup as SDE-2").
        If empty, falls back to option_a from analyzer output if present.

    option_b_label : str, optional
        Human-readable label for Option B.

    Returns
    -------
    RecommendationResult
        Fully-populated dataclass. Never raises — returns a safe fallback result
        on API or parse failure. The recommendation itself is always deterministic
        (derived algorithmically); only the reasoning text may fall back.

    Notes
    -----
    - The winning option is determined by which side the analyzer and scorer
      both favour. When they disagree, the analyzer takes precedence (higher
      weight in the blend), but a conflict penalty reduces the confidence score.
    - confidence_score is computed from a weighted blend of four signals.
    - recommendation_strength is derived algorithmically from confidence_score.
    - final_reasoning and summary are generated by a single focused AI call.
    """
    logger.info("recommendation: building final recommendation …")

    # ── Extract signals from all three upstream modules ───────────────────────
    a_sig   = _extract_analyzer_signals(analyzer_output)
    s_sig   = _extract_scorer_signals(scorer_output)
    sim_sig = _extract_simulation_signals(simulation_output)

    # ── Resolve option labels (prefer explicit args, fall back to data) ───────
    # Try to pull labels from scorer output if not passed explicitly
    if not option_a_label:
        option_a_label = (
            scorer_output.get("option_a", {}).get("label", "") or
            a_sig.get("title", "Option A")
        )
    if not option_b_label:
        option_b_label = scorer_output.get("option_b", {}).get("label", "") or "Option B"

    # Clean up — strip trailing punctuation the model sometimes adds
    option_a_label = option_a_label.strip().rstrip(".,;:")
    option_b_label = option_b_label.strip().rstrip(".,;:")

    # ── Blend confidence across all signals ───────────────────────────────────
    confidence, signals_agree = _blend_confidence(a_sig, s_sig, sim_sig)

    # ── Determine winning option ──────────────────────────────────────────────
    # The analyzer carries the highest weight and is the primary tiebreaker.
    # When signals disagree the conflict penalty already reduces confidence.
    winning_key = a_sig["analyzer_recommended"]   # "A" or "B"
    if winning_key not in VALID_OPTIONS:
        winning_key = "A"

    # ── Derive strength from blended confidence ───────────────────────────────
    strength = _confidence_to_strength(confidence)

    # ── Build human-readable recommended_option string ────────────────────────
    recommended_label = _resolve_recommended_option_label(
        winning_key, option_a_label, option_b_label
    )

    # ── AI reasoning call ─────────────────────────────────────────────────────
    prompt = _build_reasoning_prompt(
        analyzer_signals = a_sig,
        scorer_signals   = s_sig,
        sim_signals      = sim_sig,
        option_a_label   = option_a_label,
        option_b_label   = option_b_label,
        winning_key      = winning_key,
        confidence       = confidence,
        strength         = strength,
        signals_agree    = signals_agree,
    )

    try:
        raw_text, elapsed_ms = _call_api(prompt, _SYSTEM_RECOMMENDER, MAX_TOKENS)
    except Exception as exc:
        logger.error("recommendation: AI reasoning call failed: %s", exc)
        return _fallback_result(
            winning_key, option_a_label, option_b_label,
            confidence, strength, signals_agree, a_sig, s_sig, sim_sig,
        )

    parsed = _parse_json(raw_text)
    if not parsed:
        return _fallback_result(
            winning_key, option_a_label, option_b_label,
            confidence, strength, signals_agree, a_sig, s_sig, sim_sig,
        )

    # ── Assemble final result ─────────────────────────────────────────────────
    result = RecommendationResult(
        recommended_option      = recommended_label,
        confidence_score        = confidence,
        recommendation_strength = strength,
        final_reasoning         = str(parsed.get("final_reasoning", "")).strip(),
        summary                 = str(parsed.get("summary", "")).strip(),
        option_a_label          = option_a_label,
        option_b_label          = option_b_label,
        analyzer_recommendation = a_sig["analyzer_recommended"],
        scorer_leading          = s_sig["scorer_leading"],
        signals_agree           = signals_agree,
        analyzer_confidence     = a_sig["analyzer_confidence"],
        scorer_confidence       = s_sig["scorer_confidence"],
        score_gap               = s_sig["scorer_gap"],
        simulation_quality      = sim_sig["sim_quality"],
        elapsed_ms              = elapsed_ms,
        raw_response            = raw_text,
    )

    logger.info(
        "recommendation: done — %s | confidence=%d | strength=%s | agree=%s | %d ms",
        result.recommended_option[:50],
        result.confidence_score,
        result.recommendation_strength,
        result.signals_agree,
        elapsed_ms,
    )
    return result


def build_recommendation_dict(
    analyzer_output:   dict,
    scorer_output:     dict,
    simulation_output: dict,
    option_a_label:    str = "",
    option_b_label:    str = "",
) -> dict:
    """
    Plain-dict wrapper around build_recommendation().

    Guaranteed top-level keys (matching the required output spec)
    -------------------------------------------------------------
    recommended_option      : str
    confidence_score        : int   (0-100)
    recommendation_strength : str   "Weak" | "Moderate" | "Strong" | "Very Strong"
    final_reasoning         : str
    summary                 : str
    -- plus enrichment --
    option_a_label          : str
    option_b_label          : str
    analyzer_recommendation : str
    scorer_leading          : str
    signals_agree           : bool
    analyzer_confidence     : int
    scorer_confidence       : int
    score_gap               : int
    simulation_quality      : int
    is_decisive             : bool
    elapsed_ms              : int
    """
    return build_recommendation(
        analyzer_output   = analyzer_output,
        scorer_output     = scorer_output,
        simulation_output = simulation_output,
        option_a_label    = option_a_label,
        option_b_label    = option_b_label,
    ).to_dict()


# ─────────────────────────────────────────────────────────────────────────────
# UI HELPERS  (rendering aids for app.py — zero logic required there)
# ─────────────────────────────────────────────────────────────────────────────
def get_strength_colour(strength: str) -> str:
    """
    Return a hex colour suitable for the recommendation strength badge.

    Colour scheme mirrors the risk badge palette used in app.py.
    """
    return {
        "Very Strong": "#10b981",   # green
        "Strong":      "#4f8ef7",   # blue
        "Moderate":    "#f59e0b",   # amber
        "Weak":        "#ef4444",   # red
    }.get(strength, "#94a3b8")


def get_confidence_colour(confidence: int) -> str:
    """
    Return a hex colour for confidence bar gradient based on score range.

    Range        Colour
    ───────────────────────────────
    75-100       Green  (#10b981)
    55-74        Blue   (#4f8ef7)
    35-54        Amber  (#f59e0b)
    0-34         Red    (#ef4444)
    """
    if confidence >= 75:
        return "#10b981"
    if confidence >= 55:
        return "#4f8ef7"
    if confidence >= 35:
        return "#f59e0b"
    return "#ef4444"


def result_to_badge_data(result: RecommendationResult) -> dict:
    """
    Return all data needed to render the recommendation badge in app.py
    without any conditional logic in the UI layer.

    Returns
    -------
    dict with keys: label, colour, confidence_colour, is_decisive, icon
    """
    return {
        "label":              result.recommendation_strength,
        "colour":             get_strength_colour(result.recommendation_strength),
        "confidence_colour":  get_confidence_colour(result.confidence_score),
        "is_decisive":        result.is_decisive,
        "icon":               "🏆" if result.is_decisive else "📊",
    }


# ─────────────────────────────────────────────────────────────────────────────
# MANUAL SMOKE-TEST  (run: python -m modules.recommendation)
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level  = logging.INFO,
        stream = sys.stdout,
        format = "%(levelname)s | %(name)s | %(message)s",
    )

    SEP = "=" * 68

    print(f"\n{SEP}")
    print("  DecisionPilot AI — recommendation.py smoke test")
    print(f"{SEP}\n")

    # ── Weight sanity ─────────────────────────────────────────────────────────
    assert abs(_WEIGHT_SUM - 1.0) < 1e-9
    print(f"OK  Blend weights sum: {_WEIGHT_SUM:.4f} (expected 1.0)")

    # ── Sigmoid gap mapping ───────────────────────────────────────────────────
    for gap, expected_range in [(0, (0.05, 0.20)), (15, (0.45, 0.60)), (40, (0.80, 0.99))]:
        val = _sigmoid_gap(gap)
        lo, hi = expected_range
        status = "OK" if lo <= val <= hi else "FAIL"
        print(f"   {status}  _sigmoid_gap({gap:>2}) = {val:.3f}  (expected {lo}-{hi})")

    # ── Strength thresholds ───────────────────────────────────────────────────
    print()
    for score, expected in [(93, "Very Strong"), (80, "Strong"), (65, "Moderate"), (50, "Weak")]:
        label = _confidence_to_strength(score)
        status = "OK" if label == expected else "FAIL"
        print(f"   {status}  _confidence_to_strength({score}) = {label} (expected {expected})")

    # ── Colour helpers ────────────────────────────────────────────────────────
    print()
    assert get_strength_colour("Very Strong") == "#10b981"
    assert get_confidence_colour(85) == "#10b981"
    assert get_confidence_colour(30) == "#ef4444"
    print("OK  Colour helper outputs verified")

    # ── Live API call ─────────────────────────────────────────────────────────
    print(f"\n{'-' * 68}")
    print("  Live API call with synthetic upstream signals …")
    print(f"{'-' * 68}\n")

    # Synthetic analyzer output (matches analyzer.AnalysisResult.to_dict() shape)
    mock_analyzer = {
        "confidence":             78,
        "recommended":            "A",
        "recommendation_strength":"Strong",
        "reasoning":              (
            "Option A offers a significantly higher growth trajectory aligned with "
            "the user's stated goal of reaching 30 LPA within 3 years. The equity "
            "component adds asymmetric upside that Option B cannot match."
        ),
        "summary":                "Join the startup for the growth trajectory and equity upside.",
        "short_term_regret_risk": "Medium",
        "long_term_regret_risk":  "Low",
        "category":               "Career",
        "title":                  "Startup vs TCS",
        "option_a": {
            "score":             82,
            "risk_level":        "Medium",
            "long_term_impact":  "Leadership track within 3 years, strong equity outcome if startup succeeds.",
        },
        "option_b": {
            "score":             61,
            "risk_level":        "Low",
            "long_term_impact":  "Stable but slow progression; unlikely to hit 30 LPA within 3 years.",
        },
    }

    # Synthetic scorer output (matches scorer.ScoringResult.to_dict() shape)
    mock_scorer = {
        "score_a":               79,
        "score_b":               63,
        "confidence_percentage": 72,
        "score_gap":             16,
        "leading":               "A",
        "is_decisive":           True,
        "option_a": {"label": "Join startup as SDE-2", "strongest_dim": "Career Growth Potential", "weakest_dim": "Risk Level"},
        "option_b": {"label": "Stay at TCS Senior Eng", "strongest_dim": "Risk Level", "weakest_dim": "Career Growth Potential"},
    }

    # Synthetic simulation output (matches simulate_future_dict() shape)
    mock_simulation = {
        "after_1_month":  "Onboarding complete, ramping up on core product codebase, first feature shipped.",
        "after_6_months": "Leading a sub-team, mid-year review positive, equity vesting begins, salary at 18 LPA.",
        "after_1_year":   "Promoted to SDE-2.5 equivalent, 22 LPA comp, strong conversion signal for equity event.",
        "growth":         "Strong upward trajectory — skill stack, comp, and network all compounding.",
        "challenges":     "Work intensity is high; requires active boundary-setting to avoid burnout.",
        "benefits":       ["Equity upside", "Faster career progression", "Broader ownership"],
        "risks":          ["Startup failure risk", "High intensity culture"],
    }

    result = build_recommendation(
        analyzer_output   = mock_analyzer,
        scorer_output     = mock_scorer,
        simulation_output = mock_simulation,
        option_a_label    = "Join Series-A startup as SDE-2 at 18 LPA",
        option_b_label    = "Stay at TCS as Senior Engineer at 14 LPA",
    )

    print(f"OK  build_recommendation() completed in {result.elapsed_ms} ms\n")
    print(f"   recommended_option      : {result.recommended_option}")
    print(f"   confidence_score        : {result.confidence_score}/100")
    print(f"   recommendation_strength : {result.recommendation_strength}")
    print(f"   signals_agree           : {result.signals_agree}")
    print(f"   is_decisive             : {result.is_decisive}")
    print(f"\n   final_reasoning:\n   {result.final_reasoning}")
    print(f"\n   summary:\n   {result.summary}")

    # ── Badge data helper ─────────────────────────────────────────────────────
    badge = result_to_badge_data(result)
    assert "colour" in badge and "icon" in badge
    print(f"\n   Badge data: {badge}")

    # ── Dict key check ────────────────────────────────────────────────────────
    d = result.to_dict()
    required = {"recommended_option", "confidence_score", "recommendation_strength", "final_reasoning", "summary"}
    missing  = required - d.keys()
    assert not missing, f"Missing required keys: {missing}"
    print(f"\nOK  to_dict() contains all required keys: {sorted(required)}")

    # ── build_recommendation_dict check ──────────────────────────────────────
    d2 = build_recommendation_dict(mock_analyzer, mock_scorer, {})
    assert "recommended_option" in d2
    print("OK  build_recommendation_dict() with no simulation output works")

    print(f"\n{SEP}")
    print("  All smoke tests passed.")
    print(f"{SEP}\n")
