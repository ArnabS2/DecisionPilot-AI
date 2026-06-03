"""
DecisionPilot AI — "Think Before You Act."
Production-grade Streamlit app. Single Gemini call per analysis.
No fake/demo fallbacks — real errors shown to the user.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import json
import os
import datetime
import google.generativeai as genai
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

# ─────────────────────────────────────────────
# PAGE CONFIG  (must be the very first ST call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="DecisionPilot AI",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# PATHS & CONSTANTS
# ─────────────────────────────────────────────
DATA_DIR    = Path("data")
HISTORY_CSV = DATA_DIR / "decision_history.csv"
USAGE_FILE = DATA_DIR / "usage.json"
MAX_FREE_ANALYSES = 3
DATA_DIR.mkdir(parents=True, exist_ok=True)

DECISION_CATEGORIES = [
    "Career",
    "Education",
    "Business",
    "Startup",
    "Investment",
    "Finance",
    "Technology",
    "AI & Machine Learning",
    "Project",
    "Job Offer",
    "Internship",
    "Freelancing",
    "Entrepreneurship",
    "Personal Growth",
    "Skill Development",
    "Health & Fitness",
    "Lifestyle",
    "Travel",
    "Relocation",
    "Relationships",
    "Family",
    "Purchases",
    "Content Creation",
    "Time Management",
    "Other"
]

HISTORY_COLS = [
    "id", "timestamp", "title", "category",
    "option_a", "option_b", "goal", "budget", "time_commitment",
    "recommended", "confidence", "risk_level", "summary",
]

# ─────────────────────────────────────────────
# GEMINI SCHEMA  (all sections in ONE call)
# ─────────────────────────────────────────────
MASTER_SCHEMA = """
{
  "option_a": {
    "score": <int 0-100>,
    "pros": ["<string>", ...],
    "cons": ["<string>", ...],
    "risk_level": "<Low|Medium|High>",
    "risk_details": "<string>",
    "long_term_impact": "<string>",
    "opportunity_cost": "<string>"
  },
  "option_b": {
    "score": <int 0-100>,
    "pros": ["<string>", ...],
    "cons": ["<string>", ...],
    "risk_level": "<Low|Medium|High>",
    "risk_details": "<string>",
    "long_term_impact": "<string>",
    "opportunity_cost": "<string>"
  },
  "confidence": <int 0-100>,
  "recommended": "<A|B>",
  "recommendation_strength": "<Weak|Moderate|Strong|Very Strong>",
  "reasoning": "<detailed multi-sentence reasoning>",
  "summary": "<one-sentence summary>",
  "short_term_regret_risk": "<Low|Medium|High>",
  "long_term_regret_risk": "<Low|Medium|High>",
  "regret_reason": "<string>",
  "comparison": {
    "dimensions": ["Growth Potential","Cost Efficiency","Time Investment","Learning Value","Career Impact","Financial Impact","Risk (Inverted)"],
    "option_a_scores": [<int 0-10>, ...],
    "option_b_scores": [<int 0-10>, ...],
    "winner": "<A|B>",
    "winner_label": "<string>",
    "margin": <int 0-100>
  },
  "future": {
    "chosen": {
      "label": "<string>",
      "month_1": "<string>",
      "month_6": "<string>",
      "year_1":  "<string>",
      "benefits":    ["<string>", ...],
      "risks":       ["<string>", ...],
      "challenges":  ["<string>", ...],
      "growth":      "<string>"
    },
    "other": {
      "label": "<string>",
      "month_1": "<string>",
      "month_6": "<string>",
      "year_1":  "<string>",
      "benefits":    ["<string>", ...],
      "risks":       ["<string>", ...],
      "challenges":  ["<string>", ...],
      "growth":      "<string>"
    }
  }
}
"""

REQUIRED_KEYS = {
    "option_a", "option_b", "confidence", "recommended",
    "recommendation_strength", "reasoning", "summary",
    "short_term_regret_risk", "long_term_regret_risk", "regret_reason",
    "comparison", "future",
}

SYSTEM_PROMPT = (
    "You are DecisionPilot AI — a rational, world-class decision strategist. "
    "Analyze decisions with surgical precision. Expose hidden trade-offs, "
    "opportunity costs, and second-order effects. Give clear, evidence-based "
    "recommendations. ALWAYS respond in pure JSON — no markdown fences, "
    "no extra text, no explanation outside the JSON object."
)

# ─────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;1,9..40,300&display=swap');

:root {
  --bg-base:       #080c14;
  --bg-panel:      rgba(255,255,255,0.04);
  --bg-card:       rgba(255,255,255,0.06);
  --border:        rgba(255,255,255,0.08);
  --border-bright: rgba(255,255,255,0.15);
  --accent-1:      #4f8ef7;
  --accent-2:      #7c3aed;
  --accent-3:      #06b6d4;
  --accent-green:  #10b981;
  --accent-yellow: #f59e0b;
  --accent-red:    #ef4444;
  --text-primary:  #f0f4ff;
  --text-secondary:#94a3b8;
  --text-muted:    #475569;
  --radius-lg:     16px;
  --radius-md:     12px;
  --radius-sm:     8px;
  --shadow-card:   0 8px 32px rgba(0,0,0,0.3);
  --shadow-lg:     0 25px 50px rgba(0,0,0,0.5);
  --glow-blue:     0 0 40px rgba(79,142,247,0.15);
  --glow-purple:   0 0 40px rgba(124,58,237,0.15);
}

html, body, [data-testid="stAppViewContainer"] {
  background: var(--bg-base) !important;
  color: var(--text-primary) !important;
  font-family: 'DM Sans', sans-serif !important;
}
[data-testid="stAppViewContainer"]::before {
  content: '';
  position: fixed; inset: 0; pointer-events: none; z-index: 0;
  background:
    radial-gradient(ellipse 80% 50% at 20% -10%, rgba(79,142,247,0.12) 0%, transparent 60%),
    radial-gradient(ellipse 60% 40% at 80% 110%, rgba(124,58,237,0.10) 0%, transparent 60%);
}
[data-testid="stHeader"]            { display: none !important; }
[data-testid="stMainBlockContainer"]{ padding: 0 !important; max-width: 100% !important; }
[data-testid="stMain"] > div        { padding: 0 !important; }
.block-container { padding: 1.5rem 1.5rem 4rem !important; max-width: 1200px !important; margin: 0 auto !important; }

h1,h2,h3,h4 { font-family: 'Syne', sans-serif !important; }
h1 { font-size: clamp(1.8rem,4vw,2.8rem) !important; font-weight: 800 !important; }
h2 { font-size: clamp(1.3rem,3vw,1.8rem) !important; font-weight: 700 !important; }
h3 { font-size: clamp(1rem,2.5vw,1.3rem) !important; font-weight: 600 !important; }
p, li { font-size: clamp(0.88rem,1.5vw,1rem) !important; line-height: 1.7 !important; color: var(--text-secondary) !important; }

.dp-navbar { display:flex; align-items:center; justify-content:space-between; padding:1.25rem 0 2rem; }
.dp-logo {
  font-family:'Syne',sans-serif; font-size:clamp(1.1rem,2.5vw,1.4rem); font-weight:800;
  background:linear-gradient(135deg,var(--accent-1),var(--accent-2));
  -webkit-background-clip:text; -webkit-text-fill-color:transparent; letter-spacing:-0.02em;
}
.dp-tagline { color:var(--text-muted); font-size:0.78rem; letter-spacing:0.08em; text-transform:uppercase; }

.dp-card {
  background:var(--bg-card); border:1px solid var(--border); border-radius:var(--radius-lg);
  padding:1.5rem; backdrop-filter:blur(20px); box-shadow:var(--shadow-card);
  transition:border-color 0.2s,box-shadow 0.2s; margin-bottom:1rem;
}
.dp-card:hover { border-color:var(--border-bright); box-shadow:var(--shadow-lg); }
.dp-card-accent-blue   { border-top:2px solid var(--accent-1);   box-shadow:var(--glow-blue); }
.dp-card-accent-purple { border-top:2px solid var(--accent-2);   box-shadow:var(--glow-purple); }
.dp-card-accent-green  { border-top:2px solid var(--accent-green); }
.dp-card-accent-yellow { border-top:2px solid var(--accent-yellow); }
.dp-card-accent-red    { border-top:2px solid var(--accent-red); }

.dp-metric {
  background:var(--bg-panel); border:1px solid var(--border);
  border-radius:var(--radius-md); padding:1rem 1.25rem; text-align:center;
}
.dp-metric-value {
  font-family:'Syne',sans-serif; font-size:clamp(1.6rem,4vw,2.2rem); font-weight:800;
  background:linear-gradient(135deg,var(--accent-1),var(--accent-3));
  -webkit-background-clip:text; -webkit-text-fill-color:transparent; line-height:1.1;
}
.dp-metric-label { color:var(--text-muted); font-size:0.75rem; text-transform:uppercase; letter-spacing:0.06em; margin-top:0.25rem; }

.dp-bar-wrap { background:rgba(255,255,255,0.06); border-radius:99px; height:8px; overflow:hidden; margin:0.5rem 0; }
.dp-bar-fill { height:100%; border-radius:99px; transition:width 0.6s ease; }

.dp-badge { display:inline-block; padding:0.2rem 0.75rem; border-radius:99px; font-size:0.75rem; font-weight:600; letter-spacing:0.04em; }
.dp-badge-green  { background:rgba(16,185,129,0.15); color:var(--accent-green); border:1px solid rgba(16,185,129,0.3); }
.dp-badge-yellow { background:rgba(245,158,11,0.15); color:var(--accent-yellow); border:1px solid rgba(245,158,11,0.3); }
.dp-badge-red    { background:rgba(239,68,68,0.15);  color:var(--accent-red);   border:1px solid rgba(239,68,68,0.3); }
.dp-badge-blue   { background:rgba(79,142,247,0.15); color:var(--accent-1);     border:1px solid rgba(79,142,247,0.3); }

.dp-divider { border:none; border-top:1px solid var(--border); margin:1.5rem 0; }

.dp-timeline { position:relative; padding-left:1.5rem; }
.dp-timeline::before {
  content:''; position:absolute; left:0; top:0.5rem; bottom:0.5rem; width:2px;
  background:linear-gradient(to bottom,var(--accent-1),var(--accent-2),transparent); border-radius:99px;
}
.dp-timeline-item { position:relative; margin-bottom:1.25rem; }
.dp-timeline-item::before {
  content:''; position:absolute; left:-1.85rem; top:0.4rem; width:10px; height:10px;
  border-radius:50%; background:var(--accent-1); box-shadow:0 0 8px var(--accent-1);
}
.dp-timeline-title { font-family:'Syne',sans-serif; font-weight:700; font-size:0.9rem; color:var(--text-primary); margin-bottom:0.25rem; }
.dp-timeline-body  { font-size:0.88rem; color:var(--text-secondary); }

[data-testid="stTabs"] [data-baseweb="tab-list"] {
  background:var(--bg-panel) !important; border:1px solid var(--border) !important;
  border-radius:var(--radius-md) !important; padding:4px !important; gap:2px !important;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
  background:transparent !important; border-radius:var(--radius-sm) !important;
  color:var(--text-secondary) !important; font-family:'DM Sans',sans-serif !important;
  font-weight:500 !important; padding:0.5rem 1rem !important; border:none !important;
  transition:all 0.2s !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
  background:linear-gradient(135deg,var(--accent-1),var(--accent-2)) !important;
  color:white !important; font-weight:600 !important;
}
[data-testid="stTabs"] [data-baseweb="tab-highlight"],
[data-testid="stTabs"] [data-baseweb="tab-border"] { display:none !important; }

[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea,
[data-testid="stSelectbox"] > div,
[data-testid="stNumberInput"] input {
  background:var(--bg-panel) !important; border:1px solid var(--border) !important;
  border-radius:var(--radius-sm) !important; color:var(--text-primary) !important;
  font-family:'DM Sans',sans-serif !important;
}
[data-testid="stTextInput"] input:focus,
[data-testid="stTextArea"] textarea:focus {
  border-color:var(--accent-1) !important; box-shadow:0 0 0 2px rgba(79,142,247,0.2) !important;
}
label { color:var(--text-secondary) !important; font-size:0.875rem !important; font-weight:500 !important; }

[data-testid="stButton"] > button {
  background:linear-gradient(135deg,var(--accent-1),var(--accent-2)) !important;
  color:white !important; border:none !important; border-radius:var(--radius-sm) !important;
  font-family:'Syne',sans-serif !important; font-weight:600 !important; font-size:0.95rem !important;
  padding:0.65rem 1.5rem !important; width:100% !important;
  transition:opacity 0.2s,transform 0.1s !important; letter-spacing:0.02em !important;
}
[data-testid="stButton"] > button:hover  { opacity:0.88 !important; transform:translateY(-1px) !important; }
[data-testid="stButton"] > button:active { transform:translateY(0) !important; }

[data-testid="stDownloadButton"] > button {
  background:var(--bg-panel) !important; color:var(--accent-1) !important;
  border:1px solid var(--accent-1) !important; border-radius:var(--radius-sm) !important;
  font-family:'DM Sans',sans-serif !important; font-weight:600 !important; width:100% !important;
}

[data-testid="stExpander"] {
  background:var(--bg-card) !important; border:1px solid var(--border) !important;
  border-radius:var(--radius-md) !important;
}
[data-testid="stExpander"] summary { color:var(--text-primary) !important; }

[data-testid="stInfo"]    { background:rgba(79,142,247,0.1)  !important; border:1px solid rgba(79,142,247,0.2)  !important; border-radius:var(--radius-md) !important; }
[data-testid="stSuccess"] { background:rgba(16,185,129,0.1)  !important; border:1px solid rgba(16,185,129,0.2)  !important; border-radius:var(--radius-md) !important; }
[data-testid="stWarning"] { background:rgba(245,158,11,0.1)  !important; border:1px solid rgba(245,158,11,0.2)  !important; border-radius:var(--radius-md) !important; }
[data-testid="stError"]   { background:rgba(239,68,68,0.1)   !important; border:1px solid rgba(239,68,68,0.2)   !important; border-radius:var(--radius-md) !important; }

[data-testid="stDataFrame"] { background:var(--bg-card) !important; border-radius:var(--radius-md) !important; }
.js-plotly-plot .plotly    { background:transparent !important; }

::-webkit-scrollbar       { width:6px; height:6px; }
::-webkit-scrollbar-track { background:var(--bg-base); }
::-webkit-scrollbar-thumb { background:var(--border-bright); border-radius:99px; }

@media (max-width:768px) {
  .block-container { padding:1rem 0.75rem 3rem !important; }
  .dp-card         { padding:1rem; }
}
</style>
"""

# ─────────────────────────────────────────────
# PLOTLY THEME
# ─────────────────────────────────────────────
_PLOTLY = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor ="rgba(0,0,0,0)",
    font=dict(family="DM Sans", color="#94a3b8"),
    margin=dict(l=10, r=10, t=30, b=10),
    showlegend=True,
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#94a3b8")),
)

# ─────────────────────────────────────────────
# API KEY RESOLUTION
# ─────────────────────────────────────────────
def _resolve_key() -> str:
    """Return active Gemini API key: user key > env > Streamlit secrets."""
    user_key = st.session_state.get("user_gemini_key", "").strip()
    if user_key:
        return user_key
    env_key = os.getenv("GEMINI_API_KEY", "")
    if env_key:
        return env_key
    try:
        return st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        return ""


# ─────────────────────────────────────────────
# SINGLE GEMINI CALL
# ─────────────────────────────────────────────
def run_analysis(inputs: dict) -> dict:
    """
    One Gemini call returning the full master schema.
    Raises ValueError with a user-friendly message on any failure.
    """
    api_key = _resolve_key()
    if not api_key:
        raise ValueError(
            "No Gemini API key found. Add your key in the '🔑 Gemini API Key' expander above, "
            "or set GEMINI_API_KEY in your environment / Streamlit secrets."
        )

    prompt = f"""{SYSTEM_PROMPT}

Analyze this decision and return ONLY a JSON object matching this exact schema:
{MASTER_SCHEMA}

DECISION INPUTS:
- Title:            {inputs["title"]}
- Option A:         {inputs["option_a"]}
- Option B:         {inputs["option_b"]}
- Personal Goal:    {inputs["goal"]}
- Category:         {inputs["category"]}
- Budget:           {inputs.get("budget") or "Not specified"}
- Time Commitment:  {inputs.get("time_commitment") or "Not specified"}
- Extra Context:    {inputs.get("context") or "None"}

Rules:
- future.chosen = simulation for the recommended option
- future.other  = simulation for the other option
- comparison.winner_label = the actual option text, not just "A" or "B"
- All list fields must have at least 2 items
- Return pure JSON only — no markdown fences, no commentary
"""

    try:
        genai.configure(api_key=api_key)

        last_error = None
        raw = ""

        for model_name in ("gemini-2.5-flash", "gemini-1.5-flash"):
            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(prompt)
                raw = response.text.strip()
                break
            except Exception as exc:
                last_error = exc

        if not raw:
            raise ValueError(last_error or "Gemini returned an empty response.")

    except Exception as exc:
        raise ValueError(f"Gemini API error: {exc}") from exc

    # Strip markdown fences if present
    if raw.startswith("```"):
        raw = raw.split("```", 2)[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.rsplit("```", 1)[0].strip()

    # Parse JSON
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}") + 1
        if start != -1 and end > start:
            try:
                data = json.loads(raw[start:end])
            except json.JSONDecodeError as exc:
                raise ValueError(
                    "Gemini returned a response that could not be parsed as JSON. "
                    "Please try again."
                ) from exc
        else:
            raise ValueError(
                "Gemini response contained no valid JSON object. Please try again."
            )

    # Validate required top-level keys
    missing = REQUIRED_KEYS - set(data.keys())
    if missing:
        raise ValueError(
            f"Gemini response was missing required fields: {', '.join(sorted(missing))}. "
            "Please try again."
        )

    # Validate important nested structures so blank/partial JSON cannot render as 0-score UI.
    for option_key in ("option_a", "option_b"):
        option_data = data.get(option_key)
        if not isinstance(option_data, dict):
            raise ValueError(f"Gemini response field '{option_key}' must be an object. Please try again.")

        for required_field in (
            "score", "pros", "cons", "risk_level",
            "risk_details", "long_term_impact", "opportunity_cost"
        ):
            if required_field not in option_data:
                raise ValueError(
                    f"Gemini response field '{option_key}.{required_field}' is missing. "
                    "Please try again."
                )

        if not isinstance(option_data.get("score"), int):
            raise ValueError(f"Gemini response field '{option_key}.score' must be an integer. Please try again.")

    comparison = data.get("comparison")
    if not isinstance(comparison, dict):
        raise ValueError("Gemini response field 'comparison' must be an object. Please try again.")

    for required_field in ("dimensions", "option_a_scores", "option_b_scores", "winner", "winner_label", "margin"):
        if required_field not in comparison:
            raise ValueError(f"Gemini response field 'comparison.{required_field}' is missing. Please try again.")

    future = data.get("future")
    if not isinstance(future, dict) or "chosen" not in future or "other" not in future:
        raise ValueError("Gemini response field 'future' must include chosen and other paths. Please try again.")

    return data


# ─────────────────────────────────────────────
# HISTORY HELPERS
# ─────────────────────────────────────────────
def load_history() -> pd.DataFrame:
    if HISTORY_CSV.exists():
        try:
            df = pd.read_csv(HISTORY_CSV)
            for col in HISTORY_COLS:
                if col not in df.columns:
                    df[col] = ""
            return df[HISTORY_COLS]
        except Exception:
            pass
    return pd.DataFrame(columns=HISTORY_COLS)


def save_to_history(inputs: dict, result: dict) -> None:
    df  = load_history()
    rec = {col: "" for col in HISTORY_COLS}

    recommended_key = "option_a" if result.get("recommended") == "A" else "option_b"
    recommended_risk = result.get(recommended_key, {}).get("risk_level", "")

    rec.update({
        "id":              len(df) + 1,
        "timestamp":       datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "title":           inputs.get("title", ""),
        "category":        inputs.get("category", ""),
        "option_a":        inputs.get("option_a", ""),
        "option_b":        inputs.get("option_b", ""),
        "goal":            inputs.get("goal", ""),
        "budget":          inputs.get("budget", ""),
        "time_commitment": inputs.get("time_commitment", ""),
        "recommended": (
            f"Option {result['recommended']}: "
            f"{inputs['option_a'] if result['recommended'] == 'A' else inputs['option_b']}"
        ),
        "confidence":  result.get("confidence", ""),
        "risk_level":  recommended_risk,
        "summary":     result.get("summary", ""),
    })
    df = pd.concat([df, pd.DataFrame([rec])], ignore_index=True)[HISTORY_COLS]
    df.to_csv(HISTORY_CSV, index=False)

def load_usage() -> dict:
    if USAGE_FILE.exists():
        try:
            with open(USAGE_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass

    return {"analyses_used": 0}


def save_usage(usage: dict) -> None:
    with open(USAGE_FILE, "w") as f:
        json.dump(usage, f)


def free_analyses_remaining() -> int:
    usage = load_usage()
    used = usage.get("analyses_used", 0)
    return max(0, MAX_FREE_ANALYSES - used)


def increment_free_usage() -> None:
    usage = load_usage()
    usage["analyses_used"] = usage.get("analyses_used", 0) + 1
    save_usage(usage)

# ─────────────────────────────────────────────
# PDF GENERATION
# ─────────────────────────────────────────────
def generate_pdf(inputs: dict, result: dict) -> bytes:
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.lib import colors
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer,
            Table, TableStyle, HRFlowable,
        )
        import io

        ACCENT  = colors.HexColor("#4f8ef7")
        ACCENT2 = colors.HexColor("#7c3aed")
        TEXT_P  = colors.HexColor("#f0f4ff")
        TEXT_S  = colors.HexColor("#94a3b8")
        ROW_A   = colors.HexColor("#0d1525")
        ROW_B   = colors.HexColor("#111927")
        GRID_C  = colors.HexColor("#1e2d45")

        buf = io.BytesIO()
        doc = SimpleDocTemplate(
            buf, pagesize=A4,
            leftMargin=2*cm, rightMargin=2*cm,
            topMargin=2*cm, bottomMargin=2*cm,
        )
        SS = getSampleStyleSheet()

        def sty(name, **kw):
            return ParagraphStyle(name, parent=SS["Normal"], **kw)

        H1   = sty("H1",  fontSize=22, textColor=TEXT_P,  fontName="Helvetica-Bold", spaceAfter=6)
        H2   = sty("H2",  fontSize=14, textColor=ACCENT,  fontName="Helvetica-Bold", spaceAfter=4)
        H3   = sty("H3",  fontSize=11, textColor=TEXT_P,  fontName="Helvetica-Bold", spaceAfter=3)
        BODY = sty("BD",  fontSize=9,  textColor=TEXT_S,  fontName="Helvetica",      leading=14, spaceAfter=4)
        TAG  = sty("TG",  fontSize=8,  textColor=ACCENT2, fontName="Helvetica-Bold")

        def make_table(data, col_widths):
            t = Table(data, colWidths=col_widths)
            t.setStyle(TableStyle([
                ("FONTSIZE",  (0,0),(-1,-1), 8),
                ("PADDING",   (0,0),(-1,-1), 6),
                ("VALIGN",    (0,0),(-1,-1), "TOP"),
                ("GRID",      (0,0),(-1,-1), 0.3, GRID_C),
                ("ROWBACKGROUNDS", (0,0),(-1,-1), [ROW_A, ROW_B]),
                ("TEXTCOLOR", (0,0),(0,-1),  ACCENT),
                ("TEXTCOLOR", (1,0),(1,-1),  TEXT_S),
                ("FONTNAME",  (0,0),(0,-1),  "Helvetica-Bold"),
            ]))
            return t

        def hr():
            return HRFlowable(width="100%", thickness=0.5, color=GRID_C, spaceAfter=8)

        a    = result.get("option_a", {})
        b    = result.get("option_b", {})
        rec  = result.get("recommended", "?")
        fut  = result.get("future", {})

        story = [
            Paragraph("🧭 DecisionPilot AI", H1),
            Paragraph("Think Before You Act. — Full Decision Report", BODY),
            Paragraph(f"Generated: {datetime.datetime.now().strftime('%B %d, %Y at %H:%M')}", TAG),
            HRFlowable(width="100%", thickness=1, color=ACCENT, spaceAfter=12),
            Paragraph(f"Decision: {inputs.get('title','')}", H2),
            Spacer(1, 6),
            make_table([
                ["Option A",  inputs.get("option_a","")],
                ["Option B",  inputs.get("option_b","")],
                ["Goal",      inputs.get("goal","")],
                ["Category",  inputs.get("category","")],
                ["Budget",    inputs.get("budget","N/A") or "N/A"],
                ["Time",      inputs.get("time_commitment","N/A") or "N/A"],
            ], [4*cm, 13*cm]),
            Spacer(1, 12),
            Paragraph("Analysis Scores", H2),
            make_table([
                ["Metric", "Value"],
                ["Option A Score",    f"{a.get('score','—')}/100"],
                ["Option B Score",    f"{b.get('score','—')}/100"],
                ["Confidence",        f"{result.get('confidence','—')}%"],
                ["Recommended",       f"Option {rec}"],
                ["Strength",          result.get("recommendation_strength","—")],
                ["Short-Term Regret", result.get("short_term_regret_risk","—")],
                ["Long-Term Regret",  result.get("long_term_regret_risk","—")],
            ], [8*cm, 9*cm]),
            Spacer(1, 12),
            Paragraph("Strategic Reasoning", H2),
            Paragraph(result.get("reasoning",""), BODY),
            Spacer(1, 8),
        ]

        for opt, label in [
            (a, inputs.get("option_a","")),
            (b, inputs.get("option_b","")),
        ]:
            story += [
                hr(),
                Paragraph(f"{label} — Pros & Cons", H3),
                Paragraph("Pros: " + " | ".join(opt.get("pros",[])), BODY),
                Paragraph("Cons: " + " | ".join(opt.get("cons",[])), BODY),
                Paragraph(f"Long-Term Impact: {opt.get('long_term_impact','')}", BODY),
                Paragraph(f"Opportunity Cost: {opt.get('opportunity_cost','')}", BODY),
                Spacer(1, 6),
            ]

        story += [
            hr(),
            Paragraph("Regret Predictor", H2),
            Paragraph(f"Short-Term Risk: {result.get('short_term_regret_risk','—')}", BODY),
            Paragraph(f"Long-Term Risk:  {result.get('long_term_regret_risk','—')}", BODY),
            Paragraph(f"Reason: {result.get('regret_reason','')}", BODY),
            Spacer(1, 8),
        ]

        if fut:
            story.append(hr())
            story.append(Paragraph("Future Self Simulation", H2))
            for key in ["chosen", "other"]:
                sec = fut.get(key, {})
                story += [
                    Paragraph(f"Path: {sec.get('label','')}", H3),
                    Paragraph(f"1 Month:  {sec.get('month_1','')}", BODY),
                    Paragraph(f"6 Months: {sec.get('month_6','')}", BODY),
                    Paragraph(f"1 Year:   {sec.get('year_1','')}", BODY),
                    Paragraph(f"Growth:   {sec.get('growth','')}", BODY),
                    Spacer(1, 6),
                ]

        story += [
            HRFlowable(width="100%", thickness=1, color=ACCENT, spaceBefore=12, spaceAfter=6),
            Paragraph("DecisionPilot AI — Think Before You Act.", TAG),
            Paragraph("AI-generated for decision support only. Apply your own judgment.", BODY),
        ]

        doc.build(story)
        return buf.getvalue()

    except ImportError:
        return b""


# ─────────────────────────────────────────────
# CHART HELPERS
# ─────────────────────────────────────────────
def chart_bar_scores(a_score, b_score, a_label, b_label) -> go.Figure:
    fig = go.Figure(go.Bar(
        x=[a_label[:25], b_label[:25]],
        y=[a_score, b_score],
        marker=dict(color=["#4f8ef7","#7c3aed"], line=dict(width=0)),
        text=[str(a_score), str(b_score)],
        textposition="outside",
        textfont=dict(color="#f0f4ff", size=14, family="Syne"),
    ))
    fig.update_layout(
        **_PLOTLY,
        yaxis=dict(range=[0,115], gridcolor="rgba(255,255,255,0.05)"),
        xaxis=dict(gridcolor="rgba(0,0,0,0)"),
    )
    return fig


def chart_radar(comp: dict) -> go.Figure:
    dims     = comp.get("dimensions", [])
    scores_a = comp.get("option_a_scores", [])
    scores_b = comp.get("option_b_scores", [])
    fig = go.Figure()
    if dims and scores_a and scores_b:
        fig.add_trace(go.Scatterpolar(
            r=scores_a + [scores_a[0]], theta=dims + [dims[0]],
            fill="toself", name="Option A",
            line=dict(color="#4f8ef7"), fillcolor="rgba(79,142,247,0.15)",
        ))
        fig.add_trace(go.Scatterpolar(
            r=scores_b + [scores_b[0]], theta=dims + [dims[0]],
            fill="toself", name="Option B",
            line=dict(color="#7c3aed"), fillcolor="rgba(124,58,237,0.15)",
        ))
    fig.update_layout(
        **_PLOTLY,
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(
                visible=True, range=[0,10],
                gridcolor="rgba(255,255,255,0.07)",
                linecolor="rgba(255,255,255,0.07)",
                tickfont=dict(color="#475569"),
            ),
            angularaxis=dict(
                gridcolor="rgba(255,255,255,0.07)",
                linecolor="rgba(255,255,255,0.07)",
            ),
        ),
    )
    return fig


def chart_timeline(future: dict) -> go.Figure:
    months = ["Now", "1 Month", "6 Months", "1 Year"]
    chosen = future.get("chosen", {})
    other  = future.get("other",  {})
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=months, y=[50,58,72,85], mode="lines+markers",
        name=f"✅ {chosen.get('label','Recommended')[:20]}",
        line=dict(color="#10b981", width=3), marker=dict(size=8),
    ))
    fig.add_trace(go.Scatter(
        x=months, y=[50,52,60,68], mode="lines+markers",
        name=f"❌ {other.get('label','Other')[:20]}",
        line=dict(color="#ef4444", width=2, dash="dot"), marker=dict(size=6),
    ))
    fig.update_layout(
        **_PLOTLY,
        yaxis=dict(title="Satisfaction (est.)", gridcolor="rgba(255,255,255,0.05)", range=[30,100]),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
    )
    return fig


def chart_history_donut(df: pd.DataFrame) -> go.Figure:
    cats = df["category"].value_counts().reset_index()
    cats.columns = ["category", "count"]
    fig = go.Figure(go.Pie(
        labels=cats["category"], values=cats["count"], hole=0.65,
        marker=dict(colors=["#4f8ef7","#7c3aed","#06b6d4","#10b981","#f59e0b","#ef4444","#ec4899","#8b5cf6"]),
        textinfo="label+percent", textfont=dict(color="#94a3b8"),
    ))
    fig.update_layout(**_PLOTLY)
    return fig


def chart_risk_bar(df: pd.DataFrame) -> go.Figure:
    risks = df["risk_level"].value_counts().reindex(["Low","Medium","High"], fill_value=0)
    fig = go.Figure(go.Bar(
        x=risks.index, y=risks.values,
        marker=dict(color=["#10b981","#f59e0b","#ef4444"], line=dict(width=0)),
        text=risks.values, textposition="outside",
        textfont=dict(color="#f0f4ff"),
    ))
    fig.update_layout(
        **_PLOTLY,
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        xaxis=dict(gridcolor="rgba(0,0,0,0)"),
    )
    return fig


# ─────────────────────────────────────────────
# UI HELPERS
# ─────────────────────────────────────────────
def section_header(icon: str, title: str, subtitle: str = "") -> None:
    sub = (
        f"<p style='margin:0.25rem 0 0;color:#475569;font-size:0.85rem;'>{subtitle}</p>"
        if subtitle else ""
    )

    st.markdown(f"""
    <div style="margin:2rem 0 1rem;">
      <div style="display:flex;align-items:center;gap:0.55rem;">
        <span style="font-size:clamp(1.2rem,3vw,1.6rem);line-height:1;">
          {icon}
        </span>
        <h2 style="margin:0;font-family:'Syne',sans-serif;font-size:clamp(1.2rem,3vw,1.6rem);
                   background:linear-gradient(135deg,#4f8ef7,#7c3aed);
                   -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
          {title}
        </h2>
      </div>
      {sub}
    </div>
    """, unsafe_allow_html=True)


def risk_badge(level: str) -> str:
    cls = {"Low":"green","Medium":"yellow","High":"red"}.get(level,"blue")
    return f'<span class="dp-badge dp-badge-{cls}">{level} Risk</span>'


def conf_bar(pct: int, color: str = "#4f8ef7") -> str:
    return (
        f'<div class="dp-bar-wrap">'
        f'<div class="dp-bar-fill" style="width:{pct}%;'
        f'background:linear-gradient(90deg,{color},{color}99);"></div></div>'
    )


def metric_pill(value: str, label: str) -> str:
    return (
        f'<div class="dp-metric">'
        f'<div class="dp-metric-value">{value}</div>'
        f'<div class="dp-metric-label">{label}</div>'
        f'</div>'
    )


def title_slug(t: str) -> str:
    return t[:20].replace(" ", "_")


# ─────────────────────────────────────────────
# TAB 1 — DECISION ANALYZER
# ─────────────────────────────────────────────
def tab_analyzer() -> None:
    section_header("🧠", "Decision Analyzer", "Fill in the details below for a full AI-powered analysis.")

    st.markdown('<div class="dp-card dp-card-accent-blue">', unsafe_allow_html=True)
    st.markdown("#### 📝 Decision Details")

    c1, c2 = st.columns([2, 1])
    with c1:
        title = st.text_input("Decision Title", placeholder="e.g. Should I accept this job offer?")
    with c2:
        category = st.selectbox("Category", DECISION_CATEGORIES)

    c3, c4 = st.columns(2)
    with c3:
        option_a = st.text_input("Option A", placeholder="e.g. Accept the offer in Bangalore")
    with c4:
        option_b = st.text_input("Option B", placeholder="e.g. Stay at current company")

    goal = st.text_input("Your Personal Goal", placeholder="e.g. Reach ₹30 LPA and grow technically by 30")

    c5, c6 = st.columns(2)
    with c5:
        budget = st.text_input("Budget / Financial Context (optional)", placeholder="e.g. ₹5 LPA savings")
    with c6:
        time_commitment = st.text_input("Time Commitment (optional)", placeholder="e.g. 2 years")

    context = st.text_area(
        "Additional Context (optional)",
        height=90,
        placeholder="Constraints, values, fears, current situation…",
    )

    st.markdown("</div>", unsafe_allow_html=True)

    analyze_btn = st.button("⚡ Analyze My Decision", use_container_width=True)

    if not analyze_btn:
        return

    if not all([title, option_a, option_b, goal]):
        st.warning("Please fill in the Decision Title, both options, and your personal goal.")
        return

    inputs = dict(
        title=title,
        option_a=option_a,
        option_b=option_b,
        goal=goal,
        category=category,
        budget=budget,
        time_commitment=time_commitment,
        context=context,
    )

    using_user_key = bool(
        st.session_state.get("user_gemini_key", "").strip()
    )

    if not using_user_key and free_analyses_remaining() <= 0:
        st.error(
            "Free analysis limit reached. Please enter your own Gemini API key above to continue."
        )
        return

    with st.spinner("🧭 Analyzing your decision with AI… this may take a moment."):
        try:
            result = run_analysis(inputs)
        except ValueError as exc:
            st.error(f"**Analysis failed:** {exc}")
            return

    st.session_state["last_result"] = result
    st.session_state["last_inputs"] = inputs

    save_to_history(inputs, result)

    if not using_user_key:
        increment_free_usage()

    _render_results(inputs, result)


def _render_results(inputs: dict, result: dict) -> None:
    a    = result["option_a"]
    b    = result["option_b"]
    rec  = result["recommended"]
    conf = result["confidence"]
    comp = result.get("comparison", {})
    fut  = result.get("future", {})
    rec_label = inputs["option_a"] if rec == "A" else inputs["option_b"]

    # ── Metrics ──────────────────────────────────────────────
    section_header("📊", "Analysis Results")
    m1, m2, m3, m4 = st.columns(4)
    for col, val, lbl in [
        (m1, str(a.get("score","—")), "Option A Score"),
        (m2, str(b.get("score","—")), "Option B Score"),
        (m3, f"{conf}%",             "Confidence"),
        (m4, f"Option {rec}",        "Recommended"),
    ]:
        with col:
            st.markdown(metric_pill(val, lbl), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="dp-card" style="padding:1rem 1.25rem;">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.5rem;">
        <span style="font-family:'Syne',sans-serif;font-weight:700;color:#f0f4ff;">Confidence Meter</span>
        <span style="color:#4f8ef7;font-weight:700;">{conf}%</span>
      </div>
      {conf_bar(conf)}
      <div style="display:flex;justify-content:space-between;margin-top:0.35rem;">
        <span style="font-size:0.75rem;color:#475569;">Uncertain</span>
        <span style="font-size:0.75rem;color:#475569;">Very Confident</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.plotly_chart(
        chart_bar_scores(a.get("score",0), b.get("score",0), inputs["option_a"], inputs["option_b"]),
        use_container_width=True, config={"displayModeBar": False},
    )

    # ── Option cards ─────────────────────────────────────────
    st.markdown("#### Option Breakdown")
    ca, cb = st.columns(2)
    for col, opt, label, accent in [
        (ca, a, inputs["option_a"], "blue"),
        (cb, b, inputs["option_b"], "purple"),
    ]:
        with col:
            pros_li = "".join(f"<li>✅ {p}</li>" for p in opt.get("pros",[]))
            cons_li = "".join(f"<li>⚠️ {c}</li>" for c in opt.get("cons",[]))
            st.markdown(f"""
            <div class="dp-card dp-card-accent-{accent}">
              <h3 style="margin:0 0 0.5rem;">{label[:35]}</h3>
              {risk_badge(opt.get("risk_level","—"))}
              <br><br>
              <b style="color:#f0f4ff;font-size:0.85rem;">Pros</b>
              <ul style="padding-left:1rem;margin:0.25rem 0 0.75rem;">{pros_li}</ul>
              <b style="color:#f0f4ff;font-size:0.85rem;">Cons</b>
              <ul style="padding-left:1rem;margin:0.25rem 0 0.75rem;">{cons_li}</ul>
              <div style="border-top:1px solid rgba(255,255,255,0.08);padding-top:0.75rem;margin-top:0.5rem;">
                <p><b style="color:#4f8ef7;">Long-Term Impact:</b><br>{opt.get("long_term_impact","—")}</p>
                <p><b style="color:#7c3aed;">Opportunity Cost:</b><br>{opt.get("opportunity_cost","—")}</p>
                <p><b style="color:#ef4444;">Risk Details:</b><br>{opt.get("risk_details","—")}</p>
              </div>
            </div>
            """, unsafe_allow_html=True)

    # ── Regret predictor ─────────────────────────────────────
    st.markdown("#### 😬 Regret Predictor")
    r1, r2 = st.columns(2)
    for col, key, lbl in [
        (r1, "short_term_regret_risk", "Short-Term"),
        (r2, "long_term_regret_risk",  "Long-Term"),
    ]:
        lvl   = result.get(key,"—")
        color = {"Low":"#10b981","Medium":"#f59e0b","High":"#ef4444"}.get(lvl,"#4f8ef7")
        with col:
            st.markdown(f"""
            <div class="dp-card" style="border-top:2px solid {color};">
              <div style="font-family:'Syne',sans-serif;font-weight:700;font-size:1.5rem;color:{color};">{lvl}</div>
              <div style="color:#475569;font-size:0.8rem;text-transform:uppercase;letter-spacing:0.05em;">{lbl} Regret Risk</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="dp-card">
      <p><b style="color:#f59e0b;">Why you might regret this decision:</b></p>
      <p>{result.get("regret_reason","—")}</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Recommendation ────────────────────────────────────────
    strength = result.get("recommendation_strength","—")
    st.markdown(f"""
    <div class="dp-card dp-card-accent-green" style="margin-top:1rem;">
      <div style="display:flex;align-items:center;gap:0.75rem;margin-bottom:0.75rem;">
        <div style="font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:800;color:#10b981;">
          🏆 Recommendation — Option {rec}
        </div>
        <span class="dp-badge dp-badge-blue">{strength}</span>
      </div>
      <div style="font-size:1.05rem;color:#f0f4ff;font-weight:600;margin-bottom:0.5rem;">{rec_label}</div>
      <p>{result.get("reasoning","")}</p>
      <hr class="dp-divider">
      <p style="font-style:italic;color:#4f8ef7;">{result.get("summary","")}</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Comparison radar ──────────────────────────────────────
    if comp and comp.get("dimensions"):
        section_header("⚖️", "Decision Comparison Engine")
        w1, w2 = st.columns([2, 1])
        with w1:
            st.plotly_chart(chart_radar(comp), use_container_width=True, config={"displayModeBar":False})
        with w2:
            wl     = comp.get("winner_label", f"Option {comp.get('winner','?')}")
            margin = comp.get("margin", 0)
            st.markdown(f"""
            <div class="dp-card dp-card-accent-blue" style="text-align:center;">
              <div style="font-size:0.8rem;color:#475569;text-transform:uppercase;letter-spacing:0.06em;">Comparison Winner</div>
              <div style="font-family:'Syne',sans-serif;font-size:1.4rem;font-weight:800;
                          background:linear-gradient(135deg,#4f8ef7,#06b6d4);
                          -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin:0.5rem 0;">
                {wl[:30]}
              </div>
              <div style="color:#10b981;font-weight:700;">Wins by {margin} pts</div>
            </div>
            """, unsafe_allow_html=True)

    # ── Future preview ────────────────────────────────────────
    if fut and fut.get("chosen"):
        chosen = fut["chosen"]
        section_header("🔮","Future Self Preview","See the full simulation in the Future Simulation tab.")
        st.markdown(f"""
        <div class="dp-card dp-card-accent-purple">
          <div style="font-family:'Syne',sans-serif;font-weight:700;font-size:1rem;margin-bottom:1rem;">
            If you choose: <span style="color:#7c3aed;">{chosen.get("label","")}</span>
          </div>
          <div class="dp-timeline">
            <div class="dp-timeline-item">
              <div class="dp-timeline-title">1 Month</div>
              <div class="dp-timeline-body">{chosen.get("month_1","")}</div>
            </div>
            <div class="dp-timeline-item">
              <div class="dp-timeline-title">6 Months</div>
              <div class="dp-timeline-body">{chosen.get("month_6","")}</div>
            </div>
            <div class="dp-timeline-item">
              <div class="dp-timeline-title">1 Year</div>
              <div class="dp-timeline-body">{chosen.get("year_1","")}</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # ── PDF download ──────────────────────────────────────────
    section_header("📄", "Download Report")
    pdf = generate_pdf(inputs, result)
    if pdf:
        st.download_button(
            "⬇️ Download PDF Report", data=pdf,
            file_name=f"DecisionPilot_{title_slug(inputs['title'])}.pdf",
            mime="application/pdf",
        )
    else:
        st.info("Install `reportlab` to enable PDF export: `pip install reportlab`")


# ─────────────────────────────────────────────
# TAB 2 — FUTURE SIMULATION
# ─────────────────────────────────────────────
def tab_future() -> None:
    section_header("🔮","Future Self Simulation","Visualize where each path leads.")

    result = st.session_state.get("last_result",{})
    if not result:
        st.markdown("""
        <div class="dp-card" style="text-align:center;padding:3rem 2rem;">
          <div style="font-size:3rem;margin-bottom:1rem;">🔮</div>
          <h3>No simulation yet</h3>
          <p>Run a Decision Analysis first, then come back here for the full simulation.</p>
        </div>
        """, unsafe_allow_html=True)
        return

    fut    = result.get("future",{})
    chosen = fut.get("chosen",{})
    other  = fut.get("other", {})

    if not chosen:
        st.warning("The last analysis did not include a future simulation. Please re-run the analysis.")
        return

    st.plotly_chart(chart_timeline(fut), use_container_width=True, config={"displayModeBar":False})

    c1, c2 = st.columns(2)
    for col, sec, accent, icon in [(c1,chosen,"green","✅"),(c2,other,"red","❌")]:
        with col:
            ben_li = "".join(f"<li>{x}</li>" for x in sec.get("benefits",[]))
            rsk_li = "".join(f"<li>{x}</li>" for x in sec.get("risks",[]))
            chl_li = "".join(f"<li>{x}</li>" for x in sec.get("challenges",[]))
            st.markdown(f"""
            <div class="dp-card dp-card-accent-{accent}">
              <h3>{icon} {sec.get("label","")[:35]}</h3>
              <div class="dp-timeline">
                <div class="dp-timeline-item">
                  <div class="dp-timeline-title">After 1 Month</div>
                  <div class="dp-timeline-body">{sec.get("month_1","")}</div>
                </div>
                <div class="dp-timeline-item">
                  <div class="dp-timeline-title">After 6 Months</div>
                  <div class="dp-timeline-body">{sec.get("month_6","")}</div>
                </div>
                <div class="dp-timeline-item">
                  <div class="dp-timeline-title">After 1 Year</div>
                  <div class="dp-timeline-body">{sec.get("year_1","")}</div>
                </div>
              </div>
              <hr class="dp-divider">
              <b style="color:#10b981;font-size:0.85rem;">Potential Benefits</b>
              <ul style="padding-left:1.2rem;">{ben_li}</ul>
              <b style="color:#ef4444;font-size:0.85rem;">Potential Risks</b>
              <ul style="padding-left:1.2rem;">{rsk_li}</ul>
              <b style="color:#f59e0b;font-size:0.85rem;">Expected Challenges</b>
              <ul style="padding-left:1.2rem;">{chl_li}</ul>
              <hr class="dp-divider">
              <p><b style="color:#4f8ef7;">Growth Trajectory:</b><br>{sec.get("growth","")}</p>
            </div>
            """, unsafe_allow_html=True)

def tab_analytics() -> None:
    section_header(
        "📊",
        "Analytics",
        "Insights from your decision history."
    )

    df = load_history()

    if df.empty:
        st.info("No decision history available yet.")
        return

    total_decisions = len(df)

    st.metric(
        "Total Decisions",
        total_decisions
    )

    if "category" in df.columns:
        st.subheader("Category Distribution")

        category_counts = (
            df["category"]
            .fillna("Unknown")
            .value_counts()
        )

        st.bar_chart(category_counts)

    if "confidence" in df.columns:
        try:
            avg_confidence = pd.to_numeric(
                df["confidence"],
                errors="coerce"
            ).mean()

            if pd.notna(avg_confidence):
                st.metric(
                    "Average Confidence",
                    f"{avg_confidence:.1f}%"
                )
        except Exception:
            pass

# ─────────────────────────────────────────────
# TAB 3 — DECISION HISTORY
# ─────────────────────────────────────────────
def tab_history() -> None:
    section_header("📚","Decision History","All your past analyses, stored locally.")

    df = load_history()
    if df.empty:
        st.markdown("""
        <div class="dp-card" style="text-align:center;padding:3rem;">
          <div style="font-size:3rem;">📭</div>
          <h3>No decisions yet</h3>
          <p>Your decision history will appear here after your first analysis.</p>
        </div>
        """, unsafe_allow_html=True)
        return

    total    = len(df)
    conf_num = pd.to_numeric(df["confidence"], errors="coerce")
    avg_conf = f"{round(conf_num.mean(),1)}%" if conf_num.notna().any() else "—"
    top_cat  = df["category"].mode()[0] if not df["category"].dropna().empty else "—"

    m1, m2, m3 = st.columns(3)
    for col, val, lbl in [
        (m1, str(total), "Total Decisions"),
        (m2, avg_conf,   "Avg Confidence"),
        (m3, top_cat,    "Top Category"),
    ]:
        with col:
            st.markdown(metric_pill(val, lbl), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    ch1, ch2 = st.columns(2)
    with ch1:
        st.markdown('<div class="dp-card"><b style="color:#f0f4ff;">Category Distribution</b>', unsafe_allow_html=True)
        if not df["category"].dropna().empty:
            st.plotly_chart(chart_history_donut(df), use_container_width=True, config={"displayModeBar":False})
        st.markdown("</div>", unsafe_allow_html=True)
    with ch2:
        st.markdown('<div class="dp-card"><b style="color:#f0f4ff;">Risk Distribution</b>', unsafe_allow_html=True)
        if not df["risk_level"].dropna().empty:
            st.plotly_chart(chart_risk_bar(df), use_container_width=True, config={"displayModeBar":False})
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("#### All Decisions")
    display = [c for c in ["id","timestamp","title","category","confidence","recommended","risk_level","summary"] if c in df.columns]
    st.dataframe(
        df[display].sort_values("id", ascending=False).reset_index(drop=True),
        use_container_width=True, hide_index=True,
    )

    d1, d2 = st.columns(2)
    with d1:
        st.download_button(
            "⬇️ Export CSV", data=df.to_csv(index=False).encode(),
            file_name="DecisionPilot_History.csv", mime="text/csv",
        )
    with d2:
        if st.button("🗑️ Clear History"):
            HISTORY_CSV.unlink(missing_ok=True)
            st.success("History cleared. Refresh the page.")


# ─────────────────────────────────────────────
# TAB 4 — REPORTS
# ─────────────────────────────────────────────
def tab_reports() -> None:
    section_header("📄","Reports","Download the full PDF report for your last analysis.")

    result = st.session_state.get("last_result",{})
    inputs = st.session_state.get("last_inputs",{})

    if not result:
        st.markdown("""
        <div class="dp-card" style="text-align:center;padding:3rem;">
          <div style="font-size:3rem;">📋</div>
          <h3>No analysis yet</h3>
          <p>Run a Decision Analysis first, then come back to download the PDF report.</p>
        </div>
        """, unsafe_allow_html=True)
        return

    title = inputs.get("title","decision")
    st.markdown(f"""
    <div class="dp-card dp-card-accent-blue">
      <h3>📄 Report Ready</h3>
      <p><b style="color:#f0f4ff;">Decision:</b> {title}</p>
      <p><b style="color:#f0f4ff;">Recommended:</b> Option {result.get("recommended","?")} — Confidence {result.get("confidence","—")}%</p>
      <p><b style="color:#f0f4ff;">Includes:</b> Scores · Pros &amp; Cons · Risk Analysis · Regret Prediction · Future Simulation · Full Reasoning</p>
    </div>
    """, unsafe_allow_html=True)

    pdf = generate_pdf(inputs, result)
    if pdf:
        st.download_button(
            "⬇️ Download Full PDF Report", data=pdf,
            file_name=f"DecisionPilot_{title_slug(title)}.pdf",
            mime="application/pdf", use_container_width=True,
        )
    else:
        st.warning("PDF export requires `reportlab`. Run: `pip install reportlab`")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main() -> None:
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    with st.expander("🔑 Gemini API Key", expanded=False):
        st.markdown(
            "Enter your own Gemini API key for unlimited use. "
            "If left blank, the app will use the server-side key (if configured)."
        )
        user_key = st.text_input(
            "Gemini API Key (optional)",
            value=st.session_state.get("user_gemini_key",""),
            type="password",
            placeholder="AIza…",
        )
        st.session_state["user_gemini_key"] = user_key.strip()

        has_env = bool(os.getenv("GEMINI_API_KEY",""))
        has_secret = False
        try:
            has_secret = bool(st.secrets.get("GEMINI_API_KEY",""))
        except Exception:
            pass

        if st.session_state["user_gemini_key"]:
            st.success("✅ Using your Gemini API key.")
        elif has_env or has_secret:
            remaining = free_analyses_remaining()
            st.info(
                f"ℹ️ Using the server-configured Gemini API key. Free analyses remaining: {remaining}/{MAX_FREE_ANALYSES}"
            )
        else:
            st.warning(
                "⚠️ No Gemini API key found. Add yours above, or set GEMINI_API_KEY "
                "in your environment / Streamlit secrets."
            )

    st.markdown("""
    <div class="dp-navbar">
      <div>
        <div class="dp-logo">🧭 DecisionPilot AI</div>
        <div class="dp-tagline">Think Before You Act.</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    tabs = st.tabs(["🧠 Decision Analyzer","🔮 Future Simulation","📊 Analytics","📚 Decision History","📄 Reports","ℹ️ About"])
    with tabs[0]: tab_analyzer()
    with tabs[1]: tab_future()
    with tabs[2]: tab_analytics()
    with tabs[3]: tab_history()
    with tabs[4]: tab_reports()
    with tabs[5]:  # About tab
        st.markdown("## ℹ️ About DecisionPilot AI")
        st.markdown("""
### 👨‍💻 Created By
**Arnab Sahoo**

### 🚀 Version
v1.0

### 🧠 What is DecisionPilot AI?
DecisionPilot AI helps users make smarter decisions using AI-powered analysis, comparison scoring, future simulations, and PDF reports.

### ⚙️ Tech Stack
- Python
- Streamlit
- Gemini AI
- Plotly
- Pandas
- ReportLab

### 🎯 Features
- AI Decision Analysis
- Future Outcome Simulation
- Decision History Tracking
- PDF Report Generation
- Comparison Charts
""")

if __name__ == "__main__":
    main()
