"""
DecisionPilot AI — modules/future_simulator.py
Future outcome simulation engine.

Responsibilities
----------------
  - Simulate the real-world consequences of a recommended decision across
    three time checkpoints: 1 month, 6 months, and 1 year
  - Generate structured benefit, risk, growth, and challenge projections
  - Return a plain Python dict (exact output spec) and a typed SimulationResult
  - Stay fully compatible with the analyzer.py architecture:
      same MODEL, MAX_RETRIES, RETRY_DELAY_SEC, _get_client, _call_api,
      _strip_fences, _parse_json, and CATEGORY_HINTS patterns

How simulation is structured
------------------------------
The AI is not asked to "imagine" freely — it is given a structured narrative
framework that forces it to reason in three distinct phases:

  Phase 1 — Adjustment (1 month)
    Early friction, habit formation, initial wins/losses, psychological state.

  Phase 2 — Momentum (6 months)
    Compounding effects become visible; skill acquisition, social proof,
    financial reality, and course-correction opportunities.

  Phase 3 — Consolidation (1 year)
    Outcome crystallisation; measurable progress toward goal, identity shift,
    opportunity landscape compared to the unchosen path.

Each phase is grounded in the user's category and goal, so projections are
specific rather than generic.

Public API
----------
  simulate_future(...)      -> SimulationResult   (typed dataclass)
  simulate_future_dict(...) -> dict               (plain dict, exact output spec)

Output dict — guaranteed keys (matching spec)
----------------------------------------------
  after_1_month   : str
  after_6_months  : str
  after_1_year    : str
  benefits        : list[str]   (4-6 items)
  risks           : list[str]   (3-5 items)
  growth          : str
  challenges      : str         (also available as challenges_list: list[str])
  elapsed_ms      : int
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import anthropic

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────
logger = logging.getLogger("decisionpilot.future_simulator")
logger.setLevel(logging.INFO)

# ─────────────────────────────────────────────────────────────────────────────
# MODEL CONFIG  (mirrors analyzer.py exactly)
# ─────────────────────────────────────────────────────────────────────────────
MODEL           = "claude-sonnet-4-20250514"
MAX_TOKENS      = 2200
MAX_RETRIES     = 2
RETRY_DELAY_SEC = 1.5

# ─────────────────────────────────────────────────────────────────────────────
# CATEGORY HINTS  (mirrors analyzer.py — same keys, simulation-flavoured copy)
# ─────────────────────────────────────────────────────────────────────────────
# These hints nudge the model to produce domain-appropriate projections.
# "Career" projections focus on skills and comp; "Purchase" on utility decay, etc.
CATEGORY_SIMULATION_HINTS: dict[str, str] = {
    "Career": (
        "Project role progression, compensation milestones, skill stack expansion, "
        "network growth, performance review cycles, and visibility within the organisation."
    ),
    "Education": (
        "Project learning curve, credential acquisition timeline, peer network formation, "
        "practical project portfolio, and job-market positioning at each checkpoint."
    ),
    "Finance": (
        "Project capital growth or loss, liquidity position, compounding trajectory, "
        "lifestyle impact, and risk exposure change at each time horizon."
    ),
    "Purchase": (
        "Project utility delivered vs expectation, maintenance burden, total cost of "
        "ownership accumulated, resale value trajectory, and buyer's satisfaction arc."
    ),
    "Project": (
        "Project delivery milestones, team dynamics, scope creep risks, portfolio value "
        "gained, technical debt accumulated, and client or stakeholder feedback loops."
    ),
    "Internship": (
        "Project onboarding speed, skill absorption, mentor relationship quality, "
        "conversion probability growth, full-time offer likelihood, and brand value added."
    ),
    "Personal Growth": (
        "Project habit formation progress, mindset shift indicators, social capital built, "
        "identity consolidation, and measurable behavioural change at each checkpoint."
    ),
    "Other": (
        "Project concrete life changes, resource allocation shifts, relationship impacts, "
        "and measurable progress toward the stated personal goal at each time horizon."
    ),
}

# ─────────────────────────────────────────────────────────────────────────────
# BENEFIT & RISK TAXONOMY
# ─────────────────────────────────────────────────────────────────────────────
# These category-specific lenses guide which benefit and risk dimensions the
# model should prioritise when generating its lists.
BENEFIT_LENSES: dict[str, list[str]] = {
    "Career":        ["compensation growth", "skill acquisition", "network expansion", "role prestige", "autonomy"],
    "Education":     ["credential value", "knowledge depth", "peer network", "career pivot capability", "intellectual confidence"],
    "Finance":       ["capital appreciation", "passive income", "financial security", "liquidity improvement", "tax efficiency"],
    "Purchase":      ["utility delivered", "time saved", "enjoyment", "productivity gain", "status signal"],
    "Project":       ["portfolio asset", "technical mastery", "client relationship", "revenue potential", "team capability"],
    "Internship":    ["full-time conversion", "brand association", "mentorship", "live project experience", "stipend"],
    "Personal Growth":["habit permanence", "self-awareness", "resilience", "social skills", "clarity of purpose"],
    "Other":         ["goal progress", "resource efficiency", "wellbeing", "optionality", "confidence"],
}

RISK_LENSES: dict[str, list[str]] = {
    "Career":        ["role mismatch", "toxic culture", "compensation stagnation", "skill obsolescence", "job insecurity"],
    "Education":     ["credential inflation", "debt burden", "time cost", "poor institution fit", "theory-practice gap"],
    "Finance":       ["capital loss", "liquidity trap", "market volatility", "emotional trading", "over-concentration"],
    "Purchase":      ["buyer's remorse", "hidden costs", "rapid depreciation", "feature mismatch", "maintenance burden"],
    "Project":       ["scope creep", "resource overrun", "team conflict", "technical debt", "client churn"],
    "Internship":    ["no-conversion outcome", "low learning density", "stipend insufficiency", "cultural mismatch", "reference risk"],
    "Personal Growth":["motivation decay", "plateau effect", "social friction", "over-commitment", "false progress metrics"],
    "Other":         ["opportunity cost", "reversibility loss", "expectation mismatch", "resource drain", "goal drift"],
}

# ─────────────────────────────────────────────────────────────────────────────
# DATA CLASS
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class SimulationResult:
    """
    Full future simulation result.

    All fields have safe defaults so callers never hit AttributeError.
    The seven required output keys are present in to_dict() at the top level.
    """
    # ── Three time-checkpoint narratives ────────────────────────────────────
    after_1_month:   str = ""
    after_6_months:  str = ""
    after_1_year:    str = ""

    # ── Structured lists ────────────────────────────────────────────────────
    benefits:        list[str] = field(default_factory=list)   # 4-6 items
    risks:           list[str] = field(default_factory=list)   # 3-5 items
    challenges_list: list[str] = field(default_factory=list)   # 2-4 items

    # ── Narrative summaries ─────────────────────────────────────────────────
    growth:          str = ""   # growth trajectory paragraph
    challenges:      str = ""   # challenges summary paragraph (joined from list)

    # ── Meta ────────────────────────────────────────────────────────────────
    decision_title:     str = ""
    recommendation:     str = ""
    category:           str = ""
    elapsed_ms:         int = 0
    raw_response:       str = field(default="", repr=False)

    def __post_init__(self) -> None:
        """Normalise list fields and derive the challenges paragraph."""
        self.benefits        = _ensure_list(self.benefits)
        self.risks           = _ensure_list(self.risks)
        self.challenges_list = _ensure_list(self.challenges_list)

        # If the model returned challenges as a list but not a paragraph, join them
        if not self.challenges and self.challenges_list:
            self.challenges = " ".join(self.challenges_list)

        # If the model returned challenges as a paragraph but not a list, split it
        if not self.challenges_list and self.challenges:
            # Naive sentence split for list representation
            self.challenges_list = [
                s.strip() for s in re.split(r"(?<=[.!?])\s+", self.challenges) if s.strip()
            ]

    @property
    def has_complete_timeline(self) -> bool:
        """True when all three time-checkpoint narratives are non-empty."""
        return bool(self.after_1_month and self.after_6_months and self.after_1_year)

    @property
    def benefit_count(self) -> int:
        """Number of benefits identified."""
        return len(self.benefits)

    @property
    def risk_count(self) -> int:
        """Number of risks identified."""
        return len(self.risks)

    def to_dict(self) -> dict:
        """
        Return the canonical plain dict matching the required output spec.

        Guaranteed keys
        ---------------
        after_1_month   : str
        after_6_months  : str
        after_1_year    : str
        benefits        : list[str]
        risks           : list[str]
        growth          : str
        challenges      : str
        -- plus enrichment --
        challenges_list : list[str]
        decision_title  : str
        recommendation  : str
        category        : str
        elapsed_ms      : int
        """
        return {
            # Required outputs (spec)
            "after_1_month":   self.after_1_month,
            "after_6_months":  self.after_6_months,
            "after_1_year":    self.after_1_year,
            "benefits":        self.benefits,
            "risks":           self.risks,
            "growth":          self.growth,
            "challenges":      self.challenges,
            # Enrichment
            "challenges_list": self.challenges_list,
            "decision_title":  self.decision_title,
            "recommendation":  self.recommendation,
            "category":        self.category,
            "elapsed_ms":      self.elapsed_ms,
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


def _ensure_list(value: Any) -> list[str]:
    """
    Guarantee a list of non-empty strings regardless of what the model returned.

    Handles:
      - Correct list input              → returned as-is (strings only)
      - Single string                   → wrapped in a one-element list
      - None / empty string             → empty list
      - List containing non-string items → coerced to str
    """
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _strip_fences(text: str) -> str:
    """
    Remove markdown code fences the model sometimes emits despite instructions.

    Handles all three variants observed in practice:
      ```json ... ```   (most common)
      ``` ... ```
      ``` (orphan opener, no closing fence)
    """
    text = text.strip()
    # Full fenced block
    fence_re = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)
    m = fence_re.match(text)
    if m:
        return m.group(1).strip()
    # Orphan opener
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

    # Stage 1 — direct parse
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Stage 2 — extract outermost JSON object
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    logger.warning(
        "future_simulator: could not parse JSON. First 500 chars: %s", text[:500]
    )
    return {}


def _call_api(prompt: str, system: str, max_tokens: int) -> tuple[str, int]:
    """
    Call the Anthropic Messages API with automatic retry on transient errors.

    Signature matches analyzer.py exactly:
      _call_api(prompt, system, max_tokens) -> (raw_text, elapsed_ms)

    Retry strategy
    --------------
    - RateLimitError  → exponential backoff, retry up to MAX_RETRIES times
    - APIStatusError  → linear backoff, retry up to MAX_RETRIES times
    - Any other error → re-raised immediately (not retried)

    Raises
    ------
    Last caught exception if all retries are exhausted.
    """
    client    = _get_client()
    last_exc: Exception = RuntimeError("_call_api: no attempt was made")

    for attempt in range(1, MAX_RETRIES + 2):   # +2 gives MAX_RETRIES actual retries
        try:
            t0 = time.monotonic()
            response = client.messages.create(
                model     = MODEL,
                max_tokens= max_tokens,
                system    = system,
                messages  = [{"role": "user", "content": prompt}],
            )
            elapsed_ms = int((time.monotonic() - t0) * 1000)
            logger.info(
                "future_simulator: API ok in %d ms (attempt %d)", elapsed_ms, attempt
            )
            return response.content[0].text, elapsed_ms

        except anthropic.RateLimitError as exc:
            logger.warning(
                "future_simulator: rate limit hit (attempt %d/%d): %s",
                attempt, MAX_RETRIES + 1, exc,
            )
            last_exc = exc
            time.sleep(RETRY_DELAY_SEC * attempt)

        except anthropic.APIStatusError as exc:
            logger.error(
                "future_simulator: API status error (attempt %d): %s", attempt, exc
            )
            last_exc = exc
            if attempt <= MAX_RETRIES:
                time.sleep(RETRY_DELAY_SEC)

        except Exception as exc:
            logger.error("future_simulator: unexpected API error: %s", exc)
            raise

    raise last_exc


# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────────────────────────────────────────────
_SYSTEM_SIMULATOR = """\
You are DecisionPilot AI's Future Simulation Engine — a specialist in \
decision forecasting, life trajectory modelling, and outcome visualisation.

Your job is to make future consequences vivid, concrete, and believable.

Simulation principles:
  1. SPECIFICITY   — Use concrete nouns, numbers, and situations. Never be vague.
                     Bad: "You will grow professionally."
                     Good: "By month 6, you have shipped two production features,
                            received a positive mid-year review, and joined a
                            cross-functional AI taskforce."
  2. PHASE REALISM — Month 1 is always messy. Month 6 shows momentum. Year 1
                     shows whether the bet paid off. Reflect this arc honestly.
  3. GOAL ANCHORING — Every projection must trace back to the user's stated goal.
                      If the goal is financial, show numbers. If career, show
                      titles and responsibilities. If learning, show skills.
  4. BALANCED HONESTY — Show realistic challenges alongside benefits. Don't paint
                        an unrealistically rosy or dark picture.
  5. DOMAIN AWARENESS — Honour the category: career projections differ sharply
                        from purchase or education projections.

Output rules (CRITICAL):
  - Respond ONLY with valid JSON. No markdown. No prose outside the JSON object.
  - All narrative strings must be 2-4 sentences, ≥ 25 words, and specific.
  - benefits and risks must be lists of complete sentences (not fragments).
  - challenges must be a prose paragraph, not a list.
  - growth must be a forward-looking trajectory paragraph, not a list.
"""

# ─────────────────────────────────────────────────────────────────────────────
# PROMPT BUILDER
# ─────────────────────────────────────────────────────────────────────────────
def _build_simulation_prompt(
    decision_title: str,
    recommendation: str,
    goal:           str,
    category:       str,
    context:        str,
) -> str:
    """
    Construct the full simulation prompt with domain-specific guidance injected.

    The prompt uses a three-phase narrative framework to force the model to
    think about *time-ordered cause and effect*, not just list random outcomes.

    Parameters
    ----------
    decision_title : str  Short description of the decision being made.
    recommendation : str  The option that is recommended / being simulated.
    goal           : str  The user's personal goal this decision should serve.
    category       : str  Decision category (used to select domain hint).
    context        : str  Additional background provided by the user.
    """
    # Resolve domain hint — fall back to "Other" if category is unrecognised
    sim_hint = CATEGORY_SIMULATION_HINTS.get(category, CATEGORY_SIMULATION_HINTS["Other"])

    # Resolve benefit and risk lenses for the prompt
    b_lenses = ", ".join(BENEFIT_LENSES.get(category, BENEFIT_LENSES["Other"]))
    r_lenses = ", ".join(RISK_LENSES.get(category, RISK_LENSES["Other"]))

    return f"""Simulate the future outcomes for the following decision.

DECISION INPUTS
Decision Title  : {decision_title}
Chosen Option   : {recommendation}
Personal Goal   : {goal}
Category        : {category}
Extra Context   : {context or "None provided"}

DOMAIN GUIDANCE
{sim_hint}

PROJECTION FRAMEWORK
Think in three sequential phases before writing your simulation:

  Phase 1 — Adjustment (After 1 Month)
    What does daily life look like 30 days in? What early friction exists?
    What first signals — positive or negative — have appeared?
    What is the user's psychological state?

  Phase 2 — Momentum (After 6 Months)
    What compounding effects are now visible? What habits have formed?
    What measurable progress exists toward the goal? What problems have emerged?
    What did the user have to sacrifice or adapt?

  Phase 3 — Consolidation (After 1 Year)
    What is the final outcome picture? Has the goal been served?
    How does the user's situation compare to where they started?
    What new opportunities or constraints has this decision created?

BENEFIT LENSES (prioritise these for the benefits list)
  {b_lenses}

RISK LENSES (prioritise these for the risks list)
  {r_lenses}

REQUIRED JSON SCHEMA
{{
  "after_1_month"  : "<Phase 1 narrative — 2-4 sentences, ≥25 words, concrete and specific>",
  "after_6_months" : "<Phase 2 narrative — 2-4 sentences, ≥25 words, showing compounding effects>",
  "after_1_year"   : "<Phase 3 narrative — 2-4 sentences, ≥25 words, showing outcome crystallisation>",
  "benefits" : [
    "<complete sentence describing a specific benefit — ≥10 words>",
    "<another benefit>",
    "<another benefit>",
    "<another benefit>"
  ],
  "risks" : [
    "<complete sentence describing a specific risk — ≥10 words>",
    "<another risk>",
    "<another risk>"
  ],
  "growth"      : "<2-3 sentence paragraph describing the growth trajectory across the year — forward-looking, specific to goal>",
  "challenges"  : "<2-3 sentence paragraph describing the main challenges the user will face — honest, actionable>"
}}

Return ONLY the JSON object. No markdown. No text outside the JSON.
"""


# ─────────────────────────────────────────────────────────────────────────────
# RESULT BUILDER
# ─────────────────────────────────────────────────────────────────────────────
def _build_simulation_result(
    parsed:         dict,
    decision_title: str,
    recommendation: str,
    category:       str,
    elapsed_ms:     int,
    raw_response:   str,
) -> SimulationResult:
    """
    Convert a parsed API response dict into a validated SimulationResult.

    All string fields are stripped of leading/trailing whitespace.
    List fields are normalised via _ensure_list (handles strings, None, bad types).
    Missing fields fall back to empty strings / lists — never raises.

    Parameters
    ----------
    parsed         : dict  Parsed JSON from the model response.
    decision_title : str   Echoed from input for traceability.
    recommendation : str   Echoed from input for traceability.
    category       : str   Echoed from input for traceability.
    elapsed_ms     : int   API call duration in milliseconds.
    raw_response   : str   Raw model text, stored for debugging (not exposed in to_dict).
    """
    return SimulationResult(
        after_1_month   = str(parsed.get("after_1_month",  "")).strip(),
        after_6_months  = str(parsed.get("after_6_months", "")).strip(),
        after_1_year    = str(parsed.get("after_1_year",   "")).strip(),
        benefits        = parsed.get("benefits",  []),
        risks           = parsed.get("risks",     []),
        growth          = str(parsed.get("growth",     "")).strip(),
        challenges      = str(parsed.get("challenges", "")).strip(),
        challenges_list = parsed.get("challenges_list", []),
        decision_title  = decision_title,
        recommendation  = recommendation,
        category        = category,
        elapsed_ms      = elapsed_ms,
        raw_response    = raw_response,
    )


def _fallback_result(
    decision_title: str,
    recommendation: str,
    category:       str,
) -> SimulationResult:
    """
    Return a safe fallback SimulationResult when the API or parse fails.

    The fallback is clearly labelled as unavailable so the UI can display an
    appropriate message instead of silently showing empty content.
    """
    logger.error("future_simulator: returning fallback SimulationResult due to failure.")
    return SimulationResult(
        after_1_month   = (
            "Simulation could not be completed due to a network or API issue. "
            "Please retry to see your 1-month projection."
        ),
        after_6_months  = (
            "Simulation could not be completed. "
            "Please retry to see your 6-month projection."
        ),
        after_1_year    = (
            "Simulation could not be completed. "
            "Please retry to see your 1-year projection."
        ),
        benefits        = ["Retry the simulation to see projected benefits."],
        risks           = ["Retry the simulation to see projected risks."],
        growth          = "Growth trajectory unavailable. Please retry.",
        challenges      = "Challenges analysis unavailable. Please retry.",
        decision_title  = decision_title,
        recommendation  = recommendation,
        category        = category,
    )


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────
def simulate_future(
    decision_title: str,
    recommendation: str,
    goal:           str,
    category:       str = "Other",
    context:        str = "",
) -> SimulationResult:
    """
    Simulate the future outcomes of a recommended decision at three time horizons.

    Parameters
    ----------
    decision_title : str
        Short description of the decision (e.g. "Should I join the startup?").
    recommendation : str
        The option being simulated (e.g. "Join the startup as SDE-2").
        Can be the full option text or a label like "Option A".
    goal : str
        The user's personal goal this decision should serve.
        The more specific, the more grounded the simulation output.
    category : str, optional
        Decision category. One of: Career, Education, Finance, Purchase,
        Project, Internship, Personal Growth, Other. Default: "Other".
    context : str, optional
        Any additional background — constraints, values, current situation.
        Default: "".

    Returns
    -------
    SimulationResult
        Fully-populated dataclass. Never raises — returns a clearly-labelled
        fallback result on API or parse failure.

    Examples
    --------
    >>> result = simulate_future(
    ...     decision_title = "Accept the startup offer",
    ...     recommendation = "Join Series-A startup as SDE-2 at 18 LPA",
    ...     goal           = "Reach 30 LPA and Staff Engineer level in 3 years",
    ...     category       = "Career",
    ...     context        = "3 YOE, strong Python skills, no dependants",
    ... )
    >>> result.after_1_month
    '...'
    >>> result.benefits
    ['...', '...', ...]
    >>> result.to_dict()
    {'after_1_month': '...', 'after_6_months': '...', ...}
    """
    logger.info(
        "future_simulator: simulating '%s' → '%s' [%s]",
        decision_title[:50], recommendation[:50], category,
    )

    prompt = _build_simulation_prompt(
        decision_title = decision_title,
        recommendation = recommendation,
        goal           = goal,
        category       = category,
        context        = context,
    )

    # ── API call ─────────────────────────────────────────────────────────────
    try:
        raw_text, elapsed_ms = _call_api(prompt, _SYSTEM_SIMULATOR, MAX_TOKENS)
    except Exception as exc:
        logger.error("future_simulator: API call failed: %s", exc)
        return _fallback_result(decision_title, recommendation, category)

    # ── JSON parse ───────────────────────────────────────────────────────────
    parsed = _parse_json(raw_text)
    if not parsed:
        logger.error("future_simulator: empty parse result — using fallback.")
        return _fallback_result(decision_title, recommendation, category)

    # ── Validate presence of required keys ───────────────────────────────────
    required_keys = {"after_1_month", "after_6_months", "after_1_year", "benefits", "risks"}
    missing = required_keys - parsed.keys()
    if missing:
        logger.warning(
            "future_simulator: parsed response missing keys %s — proceeding with defaults.",
            missing,
        )

    # ── Build and return typed result ─────────────────────────────────────────
    result = _build_simulation_result(
        parsed         = parsed,
        decision_title = decision_title,
        recommendation = recommendation,
        category       = category,
        elapsed_ms     = elapsed_ms,
        raw_response   = raw_text,
    )

    logger.info(
        "future_simulator: complete — timeline_ok=%s | benefits=%d | risks=%d | %d ms",
        result.has_complete_timeline,
        result.benefit_count,
        result.risk_count,
        elapsed_ms,
    )
    return result


def simulate_future_dict(
    decision_title: str,
    recommendation: str,
    goal:           str,
    category:       str = "Other",
    context:        str = "",
) -> dict:
    """
    Plain-dict wrapper around simulate_future().

    Useful for app.py and other modules that consume raw dicts rather than
    typed dataclasses.

    Guaranteed top-level keys (matching the required output spec)
    -------------------------------------------------------------
    after_1_month   : str
    after_6_months  : str
    after_1_year    : str
    benefits        : list[str]
    risks           : list[str]
    growth          : str
    challenges      : str
    -- enrichment --
    challenges_list : list[str]
    decision_title  : str
    recommendation  : str
    category        : str
    elapsed_ms      : int
    """
    return simulate_future(
        decision_title = decision_title,
        recommendation = recommendation,
        goal           = goal,
        category       = category,
        context        = context,
    ).to_dict()


# ─────────────────────────────────────────────────────────────────────────────
# TIMELINE HELPERS  (for UI rendering in app.py)
# ─────────────────────────────────────────────────────────────────────────────
def get_timeline_checkpoints() -> list[dict[str, str]]:
    """
    Return the three time-checkpoint metadata records used by the UI
    to render the timeline section consistently.

    Returns
    -------
    list of dicts with keys: key, label, icon, horizon
    """
    return [
        {"key": "after_1_month",  "label": "After 1 Month",  "icon": "📅", "horizon": "short"},
        {"key": "after_6_months", "label": "After 6 Months", "icon": "📆", "horizon": "medium"},
        {"key": "after_1_year",   "label": "After 1 Year",   "icon": "🗓️", "horizon": "long"},
    ]


def result_to_timeline_items(result: SimulationResult) -> list[dict[str, str]]:
    """
    Convert a SimulationResult into an ordered list of timeline item dicts
    ready for rendering in the UI without any conditional logic in app.py.

    Returns
    -------
    list of dicts: [{label, icon, horizon, content}, ...]
    """
    checkpoints = get_timeline_checkpoints()
    result_dict = result.to_dict()
    return [
        {
            "label":   cp["label"],
            "icon":    cp["icon"],
            "horizon": cp["horizon"],
            "content": result_dict.get(cp["key"], ""),
        }
        for cp in checkpoints
    ]


# ─────────────────────────────────────────────────────────────────────────────
# MANUAL SMOKE-TEST  (run: python -m modules.future_simulator)
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level  = logging.INFO,
        stream = sys.stdout,
        format = "%(levelname)s | %(name)s | %(message)s",
    )

    SEP = "=" * 66

    print(f"\n{SEP}")
    print("  DecisionPilot AI — future_simulator.py smoke test")
    print(f"{SEP}\n")

    # ── Utility checks ────────────────────────────────────────────────────────
    assert len(CATEGORY_SIMULATION_HINTS) == 8, "Expected 8 category hints"
    print(f"OK  Category simulation hints : {len(CATEGORY_SIMULATION_HINTS)}")

    assert len(BENEFIT_LENSES) == len(RISK_LENSES) == 8, "Lens count mismatch"
    print(f"OK  Benefit lenses            : {len(BENEFIT_LENSES)}")
    print(f"OK  Risk lenses               : {len(RISK_LENSES)}")

    checkpoints = get_timeline_checkpoints()
    assert len(checkpoints) == 3
    print(f"OK  Timeline checkpoints      : {[c['label'] for c in checkpoints]}\n")

    # ── _ensure_list edge cases ───────────────────────────────────────────────
    assert _ensure_list(None)      == []
    assert _ensure_list("")        == []
    assert _ensure_list("hello")   == ["hello"]
    assert _ensure_list([1, 2, 3]) == ["1", "2", "3"]
    print("OK  _ensure_list edge cases pass\n")

    # ── Live API call ─────────────────────────────────────────────────────────
    print("-" * 66)
    print("  Live API call ...")
    print("-" * 66 + "\n")

    result = simulate_future(
        decision_title = "Should I accept the startup offer or stay at TCS?",
        recommendation = "Join Series-A startup as SDE-2 at 18 LPA plus 0.1% equity",
        goal           = "Reach 30 LPA compensation and Staff Engineer level in 3 years",
        category       = "Career",
        context        = "3 YOE, strong Python and AWS skills, 6 months savings, no dependants",
    )

    print(f"OK  simulate_future() completed in {result.elapsed_ms} ms")
    print(f"    has_complete_timeline : {result.has_complete_timeline}")
    print(f"    benefit_count         : {result.benefit_count}")
    print(f"    risk_count            : {result.risk_count}\n")

    print("  Timeline:")
    for item in result_to_timeline_items(result):
        print(f"  {item['icon']}  {item['label']}")
        print(f"     {item['content'][:120]}...")
        print()

    print("  Benefits:")
    for i, b in enumerate(result.benefits, 1):
        print(f"    {i}. {b}")

    print("\n  Risks:")
    for i, r in enumerate(result.risks, 1):
        print(f"    {i}. {r}")

    print(f"\n  Growth:\n    {result.growth}")
    print(f"\n  Challenges:\n    {result.challenges}")

    # ── to_dict key check ─────────────────────────────────────────────────────
    d = result.to_dict()
    required = {"after_1_month", "after_6_months", "after_1_year", "benefits", "risks", "growth", "challenges"}
    missing  = required - d.keys()
    assert not missing, f"Missing required keys: {missing}"
    print(f"\nOK  to_dict() contains all required keys: {sorted(required)}")

    # ── simulate_future_dict check ────────────────────────────────────────────
    d2 = simulate_future_dict(
        decision_title = "Switch from CS degree to online bootcamp",
        recommendation = "Enrol in a 6-month full-stack bootcamp",
        goal           = "Land a junior developer role within 8 months",
        category       = "Education",
    )
    assert "after_1_year" in d2
    print("OK  simulate_future_dict() returns correct keys")

    print(f"\n{SEP}")
    print("  All smoke tests passed.")
    print(f"{SEP}\n")
