"""
IPL Strategic Decision Support System
======================================
Real-time win probability · Player profile integration · Gemini AI tactical brief.

Layout:
  Left  : 3 sliders + BATTER AT CREASE dropdown + badge
  Right : live circular probability gauge
  Bottom: Gemini / rule-based tactical terminal

Run: streamlit run app.py
"""

import os
import hashlib
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import google.generativeai as genai
from datetime import datetime, timezone

# ── CONFIGURATION ──────────────────────────────────────────────────────────────
MODEL_PATH           = "notebooks/ipl_model.pkl"
SCALER_PATH          = "notebooks/ipl_scaler.pkl"
PLAYER_PROFILES_PATH = "player_profiles.csv"

GEMINI_API_KEY    = os.environ.get("GEMINI_API_KEY",
    "AQ.Ab8RN6KCbiuvkzLrHaSqasN7QOVKkVqFo4CH0opnuFgj-HeCNA")
GEMINI_MODEL_NAME = os.environ.get("GEMINI_MODEL_NAME", "gemini-2.5-flash")

MAX_BALLS       = 120
MAX_WICKETS     = 10
MAX_RUNS        = 350
DEFAULT_CRR     = 7.0    # neutral T20 average — avoids circular RRR-based estimate
DEFAULT_PLAYER_SR  = 130.0
DEFAULT_PLAYER_DOT = 35.0

# Feature order MUST match scaler.feature_names_in_
FEATURE_COLUMNS = [
    "runs_needed", "balls_left", "wickets_left",
    "crr", "rrr", "player_strike_rate", "player_dot_percent",
]

# ── PAGE CONFIG ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="IPL Strategic Decision Support System",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ─────────────────────────────────────────────────────────────────────────
def inject_styles() -> None:
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;500;600;700&family=Share+Tech+Mono&display=swap');

html, body { margin:0; padding:0; }
[data-testid="stAppViewContainer"] {
    background-color: #0c0e12;
    background-image: radial-gradient(circle, rgba(255,255,255,0.045) 1px, transparent 1px);
    background-size: 28px 28px;
    color: #ddd;
    font-family: 'Rajdhani', sans-serif;
}
[data-testid="stHeader"] { background: transparent !important; }
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] { display:none !important; }
.main .block-container { padding: 1rem 1.8rem 2rem; max-width: 1300px; }

/* ── Title ── */
.ipl-title {
    font-family: 'Rajdhani', sans-serif;
    font-size: clamp(1.5rem, 3.8vw, 3rem);
    font-weight: 700;
    letter-spacing: 0.06em;
    text-align: center;
    background: linear-gradient(135deg,#f5c518 0%,#ffe066 45%,#c8870a 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
    margin: 0.3rem 0 0.15rem;
}

/* ── Nav buttons styled as breadcrumb links ── */
.nav-row .stButton > button {
    background: transparent !important;
    border: none !important;
    color: rgba(255,255,255,0.35) !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.05em !important;
    padding: 0 0.3rem 0.4rem !important;
    height: auto !important;
    font-family: 'Rajdhani', sans-serif !important;
    box-shadow: none !important;
    text-decoration: none !important;
}
.nav-row .stButton > button:hover { color: rgba(245,197,24,0.8) !important; }

/* ── Panel headers ── */
.panel-header {
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: rgba(245,197,24,0.65);
    border-bottom: 1px solid rgba(245,197,24,0.15);
    padding-bottom: 0.4rem;
    margin-bottom: 0.75rem;
}

/* ── Slider rows ── */
.slider-label {
    font-size: 0.77rem;
    color: rgba(255,255,255,0.55);
    padding-top: 0.5rem;
}
.val-box {
    background: rgba(245,197,24,0.08);
    border: 1px solid rgba(245,197,24,0.38);
    border-radius: 4px;
    padding: 0.25rem 0.5rem;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.88rem;
    color: #f5c518;
    text-align: center;
    margin-top: 0.35rem;
}

/* ── Slider skin ── */
[data-testid="stSlider"] { padding:0 !important; }
[data-testid="stSlider"] > label { display:none !important; }
[data-baseweb="slider"] [role="slider"] {
    background: #f5c518 !important; border-color: #f5c518 !important;
    box-shadow: 0 0 6px rgba(245,197,24,0.65) !important;
}
[data-baseweb="slider"] [data-testid="stSliderTrackActive"] {
    background: linear-gradient(90deg,#c8870a,#f5c518) !important;
}
[data-baseweb="slider"] > div > div { background: rgba(255,255,255,0.1) !important; }
[data-testid="stSliderTickBarMin"], [data-testid="stSliderTickBarMax"] { display:none !important; }

/* ── Selectbox ── */
[data-testid="stSelectbox"] > label { display:none !important; }
[data-baseweb="select"] > div {
    background: rgba(245,197,24,0.06) !important;
    border: 1px solid rgba(245,197,24,0.3) !important;
    border-radius: 6px !important;
    color: #f5c518 !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.8rem !important;
}
[data-baseweb="select"] svg { fill: rgba(245,197,24,0.5) !important; }
[data-baseweb="popover"] ul { background: #10141c !important; border:1px solid rgba(245,197,24,0.2) !important; }
[data-baseweb="popover"] li { color: #ccc !important; font-size:0.8rem !important; }
[data-baseweb="popover"] li:hover { background: rgba(245,197,24,0.1) !important; color:#f5c518 !important; }

/* ── Player badge ── */
.player-badge {
    background: rgba(245,197,24,0.07);
    border: 1px solid rgba(245,197,24,0.22);
    border-radius: 6px;
    padding: 0.45rem 0.7rem;
    font-size: 0.72rem;
    color: rgba(255,255,255,0.5);
    margin-top: 0.5rem;
    line-height: 1.6;
}
.player-badge .pname { color: #f5c518; font-weight:600; font-size:0.82rem; }

/* ── Gauge ── */
.gauge-wrap {
    display:flex; flex-direction:column; align-items:center;
    justify-content:center; padding:0.2rem 0 0.6rem;
}
.gauge-ring { position:relative; width:240px; height:240px; }
.gauge-ring svg { display:block; }
.gauge-center {
    position:absolute; top:50%; left:50%;
    transform:translate(-50%,-50%);
    text-align:center; pointer-events:none;
}
.gauge-pct {
    font-family:'Rajdhani',sans-serif; font-size:3.4rem; font-weight:700; line-height:1;
    background:linear-gradient(135deg,#f5c518,#ffe066);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;
}
.gauge-sublabel {
    font-size:0.58rem; letter-spacing:0.14em; text-transform:uppercase;
    color:rgba(255,255,255,0.35); margin-top:0.3rem;
}
.gauge-stats {
    font-size:0.72rem; color:rgba(255,255,255,0.4);
    text-align:center; margin-top:0.3rem; letter-spacing:0.04em;
}
.gauge-stats .stat-val { color:#f5c518; font-weight:600; }

/* ── Terminal ── */
.terminal-header {
    font-size:0.63rem; font-weight:600; letter-spacing:0.18em; text-transform:uppercase;
    color:rgba(245,197,24,0.55);
    border:1px solid rgba(245,197,24,0.2); border-bottom:none;
    border-radius:6px 6px 0 0;
    padding:0.3rem 0.9rem;
    background:rgba(245,197,24,0.04);
}
.terminal-body {
    background:#080c10;
    border:1px solid rgba(245,197,24,0.2);
    border-radius:0 0 6px 6px;
    padding:0.8rem 1rem;
    font-family:'Share Tech Mono',monospace;
    font-size:0.77rem; line-height:1.75; color:#b0c8b0;
    min-height:85px; white-space:pre-wrap; word-break:break-word;
}
.terminal-body .cmd { color:#f5c518; }
.terminal-body .key { color:#f5c518; font-weight:700; }
.terminal-body .cursor {
    display:inline-block; width:7px; height:0.9em;
    background:#f5c518; vertical-align:text-bottom;
    animation:blink 1.1s step-end infinite;
}
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0} }

/* ── Mobile ── */
@media (max-width:768px) {
    .main .block-container { padding:0.7rem 0.8rem 1.5rem; }
    .ipl-title { font-size:1.35rem; }
    .gauge-ring { width:180px; height:180px; }
    .gauge-pct { font-size:2.5rem; }
}
</style>
""", unsafe_allow_html=True)


# ── ASSET LOADERS ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        return joblib.load(MODEL_PATH)
    except Exception as exc:
        st.error(f"Model load failed — {MODEL_PATH}: {exc}")
        return None


@st.cache_resource
def load_scaler():
    try:
        return joblib.load(SCALER_PATH)
    except Exception as exc:
        st.error(f"Scaler load failed — {SCALER_PATH}: {exc}")
        return None


@st.cache_data
def load_player_profiles() -> pd.DataFrame:
    try:
        full = pd.read_csv(PLAYER_PROFILES_PATH)
        df = full[["batter", "player_strike_rate", "player_dot_percent"]].dropna()
        if "balls_faced" in full.columns:
            df = df[full["balls_faced"] >= 200]
        return df.sort_values("batter").reset_index(drop=True)
    except Exception:
        return pd.DataFrame(columns=["batter", "player_strike_rate", "player_dot_percent"])


# ── SESSION STATE ───────────────────────────────────────────────────────────────
def init_session() -> None:
    defaults = {
        "selected_player":  "— Select a player —",
        "last_analysis":    "",
        "last_inputs_hash": "",
        "page":             "match",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ── CORE LOGIC ──────────────────────────────────────────────────────────────────
def compute_win_probability(
    model, scaler,
    runs_needed: int, balls_remaining: int, wickets_left: int,
    player_sr: float, player_dot: float,
) -> float:
    """
    Compute chasing-team win probability.
    CRR is fixed at the historical T20 average (7.0) — deriving it from RRR
    caused systematic over-pessimism because it told the model the team was
    already scoring at an impossibly high rate.
    """
    if model is None or scaler is None:
        return 50.0
    if balls_remaining <= 0:
        return 0.0
    if runs_needed <= 0:
        return 100.0

    rrr = float(np.clip((runs_needed * 6.0) / balls_remaining, 0.0, 36.0))
    feature_df = pd.DataFrame(
        [[runs_needed, balls_remaining, wickets_left,
          DEFAULT_CRR, rrr, player_sr, player_dot]],
        columns=FEATURE_COLUMNS,
    )
    scaled = scaler.transform(feature_df)
    return round(float(model.predict_proba(scaled)[0][1] * 100.0), 1)


def inputs_hash(runs: int, balls: int, wickets: int, sr: float, dot: float) -> str:
    return hashlib.md5(f"{runs}|{balls}|{wickets}|{sr:.1f}|{dot:.1f}".encode()).hexdigest()


def build_gemini_prompt(
    runs_needed: int, balls_remaining: int, wickets_left: int,
    rrr: float, win_probability: float,
    player_label: str, player_sr: float, player_dot: float,
) -> str:
    overs_left  = balls_remaining // 6
    extra_balls = balls_remaining % 6
    overs_str = f"{overs_left}ov {extra_balls}b" if extra_balls else f"{overs_left} overs"
    return (
        "You are an IPL Head Coach. Give a direct 2-3 sentence tactical brief. "
        "No emojis. No filler. Pure cricket intelligence.\n\n"
        f"MATCH STATE: {runs_needed} runs needed, {overs_str} remaining, "
        f"{wickets_left} wickets in hand. RRR: {rrr:.2f}. "
        f"Win probability: {win_probability:.1f}%.\n"
        f"BATTER ({player_label}): Strike Rate {player_sr:.0f}, "
        f"Dot Ball {player_dot:.0f}%.\n\n"
        "Address: (1) urgency vs patience call, (2) one specific tactical instruction. Max 55 words."
    )


def rule_based_analysis(
    runs_needed: int, balls_remaining: int, wickets_left: int,
    rrr: float, win_probability: float, player_label: str,
) -> str:
    urgency = "critical" if rrr > 12 else ("high" if rrr > 9 else "moderate")
    if win_probability > 60:
        tactic = "maintain partnerships and accelerate gradually through the field"
    elif win_probability < 35:
        tactic = "attack immediately — dot balls at this rate are match-losing"
    else:
        tactic = "rotate strike aggressively and target the 5th/6th bowler"
    return (
        f"Required run rate {rrr:.2f} — pressure is {urgency}. "
        f"With {wickets_left} wickets and {balls_remaining} balls remaining, "
        f"{player_label} must {tactic}. Win probability: {win_probability:.1f}%."
    )


def call_gemini(llm, prompt: str):
    try:
        return llm.generate_content(prompt).text.strip()
    except Exception:
        return None


# ── RENDER HELPERS ──────────────────────────────────────────────────────────────
def render_gauge(probability: float) -> None:
    r, sw = 100, 16
    circ = 2 * np.pi * r
    fill = (probability / 100.0) * circ
    gap  = circ - fill

    if probability >= 60:
        col, glow = "#f5c518", "rgba(245,197,24,0.5)"
    elif probability >= 35:
        col, glow = "#e8a000", "rgba(232,160,0,0.4)"
    else:
        col, glow = "#c0392b", "rgba(192,57,43,0.45)"

    st.markdown(f"""
<div class="gauge-wrap">
  <div class="gauge-ring">
    <svg width="240" height="240" viewBox="0 0 240 240">
      <defs>
        <filter id="gf">
          <feGaussianBlur stdDeviation="5" result="b"/>
          <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
        </filter>
      </defs>
      <circle cx="120" cy="120" r="{r}"
        fill="none" stroke="rgba(255,255,255,0.07)" stroke-width="{sw}"
        transform="rotate(-90 120 120)"/>
      <circle cx="120" cy="120" r="{r}"
        fill="none" stroke="{col}" stroke-width="{sw}" stroke-linecap="round"
        stroke-dasharray="{fill:.2f} {gap:.2f}"
        transform="rotate(-90 120 120)"
        style="filter:drop-shadow(0 0 8px {glow});"/>
    </svg>
    <div class="gauge-center">
      <div class="gauge-pct">{probability:.0f}%</div>
      <div class="gauge-sublabel">Victory<br>Probability</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


def render_terminal(
    analysis: str, rrr: float, probability: float,
    player_label: str, player_sr: float, player_dot: float,
) -> None:
    now = datetime.now(timezone.utc).strftime("%H:%M:%S GMT")

    if not analysis:
        body = (
            f'<span class="cmd">&gt; </span>'
            f'<span class="key">SYSTEM STATUS:</span> Initialising analysis...<br>'
            f'<span class="cmd">&gt; </span>'
            f'<span class="key">LAST UPDATE:</span> {now} '
            f'<span class="cursor"></span>'
        )
    else:
        lines = analysis.split("\n")
        first = lines[0]
        rest  = "<br>  ".join(lines[1:]) if len(lines) > 1 else ""
        body = (
            f'<span class="cmd">&gt; </span>'
            f'<span class="key">ANALYSIS COMPLETE:</span> {first}<br>'
        )
        if rest.strip():
            body += f'  {rest}<br>'
        body += (
            f'<span class="cmd">&gt; </span>'
            f'<span class="key">BATTER:</span> {player_label} '
            f'| SR {player_sr:.0f} | Dot {player_dot:.0f}%<br>'
            f'<span class="cmd">&gt; </span>'
            f'<span class="key">WIN PROBABILITY:</span> {probability:.1f}% '
            f'| RRR {rrr:.2f}<br>'
            f'<span class="cmd">&gt; </span>'
            f'<span class="key">LAST UPDATE:</span> {now} '
            f'<span class="cursor"></span>'
        )

    st.markdown(
        f'<div class="terminal-header">Strategic Insight &amp; Command Center</div>'
        f'<div class="terminal-body">{body}</div>',
        unsafe_allow_html=True,
    )


# ── MAIN ────────────────────────────────────────────────────────────────────────
def main() -> None:
    inject_styles()
    init_session()

    model     = load_model()
    scaler    = load_scaler()
    player_df = load_player_profiles()

    # Configure Gemini (silently — no UI warning if it fails)
    llm = None
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        llm = genai.GenerativeModel(GEMINI_MODEL_NAME)
    except Exception:
        pass

    # ── TITLE ──────────────────────────────────────────────────────────────────
    st.markdown(
        '<div class="ipl-title">IPL STRATEGIC DECISION SUPPORT SYSTEM</div>',
        unsafe_allow_html=True,
    )

    # ── NAV BUTTONS (breadcrumb style) ─────────────────────────────────────────
    st.markdown('<div class="nav-row">', unsafe_allow_html=True)
    nav_cols = st.columns([0.7, 0.9, 1.2, 7])
    with nav_cols[0]:
        if st.button("Home", key="nav_home"):
            st.session_state.page = "home"
            st.rerun()
    with nav_cols[1]:
        if st.button("Dashboard", key="nav_dash"):
            st.session_state.page = "dashboard"
            st.rerun()
    with nav_cols[2]:
        if st.button("Match Analysis", key="nav_match"):
            st.session_state.page = "match"
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<hr style="margin:0.2rem 0 0.8rem;border-color:rgba(245,197,24,0.1);">', unsafe_allow_html=True)

    page = st.session_state.page

    # ── HOME ───────────────────────────────────────────────────────────────────
    if page == "home":
        st.markdown("""
<div style="text-align:center; padding:3rem 2rem;">
  <div style="font-family:'Rajdhani',sans-serif; font-size:1.8rem; font-weight:700;
              background:linear-gradient(135deg,#f5c518,#ffe066);
              -webkit-background-clip:text; -webkit-text-fill-color:transparent;
              margin-bottom:1rem;">
    Welcome to the IPL SDSS
  </div>
  <div style="color:rgba(255,255,255,0.45); font-size:0.92rem;
              max-width:600px; margin:0 auto; line-height:1.9;">
    A real-time decision support system powered by a machine-learning win-probability
    model trained on 260,920 IPL deliveries and enhanced with Gemini AI tactical
    intelligence.<br><br>
    Navigate to <strong style="color:#f5c518;">Match Analysis</strong> to start a live
    prediction, or <strong style="color:#f5c518;">Dashboard</strong> to explore player stats.
  </div>
</div>
""", unsafe_allow_html=True)
        return

    # ── DASHBOARD ──────────────────────────────────────────────────────────────
    if page == "dashboard":
        st.markdown(
            '<div class="panel-header">Player Intelligence — Career Stats</div>',
            unsafe_allow_html=True,
        )
        if not player_df.empty:
            display_df = player_df.copy()
            display_df["player_strike_rate"] = display_df["player_strike_rate"].round(1)
            display_df["player_dot_percent"] = display_df["player_dot_percent"].round(1)
            display_df.columns = ["Batter", "Strike Rate", "Dot Ball %"]
            search = st.text_input("Search player", placeholder="e.g. Kohli, Dhoni, Rohit")
            if search:
                display_df = display_df[
                    display_df["Batter"].str.contains(search, case=False, na=False)
                ]
            st.dataframe(
                display_df.sort_values("Strike Rate", ascending=False).reset_index(drop=True),
                use_container_width=True, height=420,
            )
        else:
            st.info("player_profiles.csv not found.")
        return

    # ── MATCH ANALYSIS (default) ───────────────────────────────────────────────
    left_col, right_col = st.columns([2, 2.5], gap="large")

    # ── LEFT: INPUTS ──────────────────────────────────────────────────────────
    with left_col:
        st.markdown(
            '<div class="panel-header">Game Scenario Inputs</div>',
            unsafe_allow_html=True,
        )

        # Runs Needed
        la, lb, lc = st.columns([2.2, 3.8, 1])
        with la:
            st.markdown('<div class="slider-label">Runs Needed</div>', unsafe_allow_html=True)
        with lb:
            runs_needed = st.slider("rn", 1, MAX_RUNS, 125, label_visibility="collapsed")
        with lc:
            st.markdown(f'<div class="val-box">{runs_needed}</div>', unsafe_allow_html=True)

        # Balls Remaining
        ma, mb, mc = st.columns([2.2, 3.8, 1])
        with ma:
            st.markdown('<div class="slider-label">Balls Remaining</div>', unsafe_allow_html=True)
        with mb:
            balls_remaining = st.slider("br", 1, MAX_BALLS, 80, label_visibility="collapsed")
        with mc:
            st.markdown(f'<div class="val-box">{balls_remaining}</div>', unsafe_allow_html=True)

        # Wickets in Hand
        na, nb, nc = st.columns([2.2, 3.8, 1])
        with na:
            st.markdown('<div class="slider-label">Wickets in Hand</div>', unsafe_allow_html=True)
        with nb:
            wickets_left = st.slider("wl", 1, MAX_WICKETS, 6, label_visibility="collapsed")
        with nc:
            st.markdown(f'<div class="val-box">{wickets_left}</div>', unsafe_allow_html=True)

        # ── Batter at Crease ────────────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            '<div style="font-size:0.65rem;letter-spacing:0.15em;'
            'color:rgba(245,197,24,0.55);margin-bottom:0.35rem;">'
            'BATTER AT CREASE</div>',
            unsafe_allow_html=True,
        )
        player_options = ["— Select a player —"] + player_df["batter"].tolist()
        current_player = st.session_state.selected_player
        if current_player not in player_options:
            current_player = "— Select a player —"

        new_player = st.selectbox(
            "player_select",
            options=player_options,
            index=player_options.index(current_player),
            label_visibility="collapsed",
        )
        if new_player != st.session_state.selected_player:
            st.session_state.selected_player = new_player
            st.rerun()

    # ── Resolve batter profile ──────────────────────────────────────────────
    selected_player = st.session_state.selected_player
    player_row = (
        player_df[player_df["batter"] == selected_player]
        if selected_player not in ("— Select a player —", None, "")
        else pd.DataFrame()
    )

    if not player_row.empty:
        player_sr    = float(player_row.iloc[0]["player_strike_rate"])
        player_dot   = float(player_row.iloc[0]["player_dot_percent"])
        player_label = selected_player
    else:
        player_sr    = DEFAULT_PLAYER_SR
        player_dot   = DEFAULT_PLAYER_DOT
        player_label = "Average batter"

    # Show badge in left panel
    with left_col:
        if not player_row.empty:
            st.markdown(
                f'<div class="player-badge">'
                f'<span class="pname">{selected_player}</span><br>'
                f'Strike Rate: <strong>{player_sr:.1f}</strong>'
                f'&nbsp;|&nbsp;Dot Ball: <strong>{player_dot:.1f}%</strong>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # ── Compute win probability ────────────────────────────────────────────
    match_won  = runs_needed <= 0
    match_lost = balls_remaining <= 0 and runs_needed > 0

    if match_won:
        win_probability = 100.0
    elif match_lost:
        win_probability = 0.0
    else:
        win_probability = compute_win_probability(
            model, scaler,
            runs_needed, balls_remaining, wickets_left,
            player_sr, player_dot,
        )

    rrr = float(np.clip(
        (runs_needed * 6.0 / balls_remaining) if balls_remaining > 0 else 0.0,
        0.0, 36.0,
    ))

    # ── RIGHT: GAUGE ───────────────────────────────────────────────────────
    with right_col:
        render_gauge(win_probability)
        st.markdown(
            f'<div class="gauge-stats">'
            f'Required run rate: <span class="stat-val">{rrr:.2f}</span> per over<br>'
            f'Batter: <span class="stat-val">{player_label}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── TERMINAL ───────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)

    h = inputs_hash(runs_needed, balls_remaining, wickets_left, player_sr, player_dot)

    if h != st.session_state.last_inputs_hash and not match_won and not match_lost:
        if llm is not None:
            with st.spinner(""):
                prompt   = build_gemini_prompt(
                    runs_needed, balls_remaining, wickets_left,
                    rrr, win_probability, player_label, player_sr, player_dot,
                )
                ai_reply = call_gemini(llm, prompt)
                if ai_reply is None:
                    ai_reply = rule_based_analysis(
                        runs_needed, balls_remaining, wickets_left,
                        rrr, win_probability, player_label,
                    )
        else:
            ai_reply = rule_based_analysis(
                runs_needed, balls_remaining, wickets_left,
                rrr, win_probability, player_label,
            )
        st.session_state.last_analysis    = ai_reply
        st.session_state.last_inputs_hash = h

    display_analysis = st.session_state.last_analysis
    if match_won:
        display_analysis = "Target reached. Batting team wins the match."
    elif match_lost:
        display_analysis = "Innings complete. Defending team wins — target not achieved."

    render_terminal(
        display_analysis, rrr, win_probability,
        player_label, player_sr, player_dot,
    )


if __name__ == "__main__":
    main()