"""
OHL Matchday Attendance Predictor — Streamlit web app.

Run with: streamlit run streamlit_app.py
"""
import datetime
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# ── Config ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="OHL Attendance Predictor",
    page_icon="⚽",
    layout="wide",
)

MODELS_DIR = Path(__file__).parent / "models"

NUMERIC_FEATURES = [
    'kickoff_hour', 'is_playoff', 'is_weekend', 'is_school_holiday_flanders',
    'has_promotion', 'weather_score', 'avg_ohl_interest_7d',
    'pre_match_article_count', 'home_win_rate_last3', 'ohl_season_points',
    'opp_ppg_vs_ohl', 'rolling_avg_attendance_last3', 'matchday_normalized',
    'last_h2h_goal_margin', 'ohl_points_per_game', 'has_campaign',
    'pre_match_interest_ratio',
]

FEAT_14 = [
    'rolling_avg_attendance_last3', 'ohl_points_per_game', 'home_win_rate_last3',
    'matchday_normalized', 'is_weekend', 'is_school_holiday_flanders',
    'academic_week', 'has_promotion', 'weather_score', 'weather_temp_deviation',
    'avg_ohl_interest_7d', 'pre_match_interest_ratio', 'opp_ppg_vs_ohl',
    'last_h2h_goal_margin',
]

ALL_OPPONENTS = [
    'Anderlecht', 'Antwerp', 'Cercle Brugge', 'Club Brugge', 'Dender',
    'Eupen', 'Genk', 'Gent', 'Kortrijk', 'Mechelen', 'Other',
    'Sint-Truiden', 'Sporting Charleroi', 'Standard Liège',
    'Union Saint-Gilloise', 'Westerlo', 'Zulte Waregem',
]

# Historical opp_ppg_vs_ohl per opponent (derived from training data)
OPPONENT_PPG = {
    'Anderlecht': 1.0, 'Antwerp': 3.0, 'Cercle Brugge': 3.0,
    'Club Brugge': 1.52, 'Dender': 1.0, 'Eupen': 3.0,
    'Genk': 1.52, 'Gent': 1.52, 'Kortrijk': 0.0, 'Mechelen': 1.0,
    'Other': 1.52, 'Sint-Truiden': 1.0, 'Sporting Charleroi': 1.52,
    'Standard Liège': 0.0, 'Union Saint-Gilloise': 1.52,
    'Westerlo': 1.52, 'Zulte Waregem': 0.0,
}

# Buzz presets → (avg_ohl_interest_7d, pre_match_interest_ratio)
# Mapped to p25 / p50 / p75 / p90 of training data
BUZZ_PRESETS = {
    "Low":       (4.02, 0.749),
    "Normal":    (5.57, 0.967),
    "High":      (7.99, 1.201),
    "Very High": (10.23, 1.677),
}

# Weather presets → weather_score (p10 / p25 / p50 / p75 / p90)
WEATHER_PRESETS = {
    "Stormy":    -3.83,
    "Poor":      -1.71,
    "Average":   -1.04,
    "Good":      -0.21,
    "Excellent":  0.18,
}

# Historical reference values (from training data)
HIST_MEAN = 7122
HIST_MIN = 4523
HIST_MAX = 10029
STADIUM_CAPACITY = 11700


# ── Load models (cached — loaded once per session) ─────────────────────────
@st.cache_resource
def load_models():
    if not MODELS_DIR.exists():
        st.error("Models not found. Run: `python src/model.py`", icon="🚨")
        st.stop()
    return {
        'ridge':      joblib.load(MODELS_DIR / 'ridge_model.pkl'),
        'qr35':       joblib.load(MODELS_DIR / 'qr35_model.pkl'),
        'qr25':       joblib.load(MODELS_DIR / 'qr25_model.pkl'),
        'cols_ridge': joblib.load(MODELS_DIR / 'feature_columns_ridge.pkl'),
        'cols_qr':    joblib.load(MODELS_DIR / 'feature_columns_qr.pkl'),
    }


# ── Build input vectors ────────────────────────────────────────────────────
def build_ridge_input(inputs: dict, feature_columns: list) -> pd.DataFrame:
    row = {col: 0.0 for col in feature_columns}
    for feat in NUMERIC_FEATURES:
        row[feat] = inputs[feat]
    opp_col = f"opponent_grouped_{inputs['opponent_grouped']}"
    if opp_col in row:
        row[opp_col] = 1.0
    h2h_col = f"last_h2h_result_{inputs['last_h2h_result']}"
    if h2h_col in row:
        row[h2h_col] = 1.0
    return pd.DataFrame([row])[feature_columns]


def build_qr_input(inputs: dict, feature_columns: list) -> pd.DataFrame:
    row = {feat: inputs[feat] for feat in FEAT_14}
    return pd.DataFrame([row])[feature_columns]


# ── Attendance range chart ─────────────────────────────────────────────────
def make_chart(ridge_pred, qr35_pred, qr25_pred):
    fig, ax = plt.subplots(figsize=(8, 2.2))
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')

    ax.barh(y=0, width=STADIUM_CAPACITY, left=0, height=0.5,
            color='#1e2530', zorder=1)

    low  = min(qr25_pred, qr35_pred, ridge_pred)
    high = max(qr25_pred, qr35_pred, ridge_pred)
    ax.barh(y=0, width=(high - low), left=low, height=0.5,
            color='#2563eb', alpha=0.35, zorder=2)

    ax.axvline(ridge_pred, color='#60a5fa', lw=2.5, zorder=4,
               label=f'Estimate  {ridge_pred:,.0f}')
    ax.axvline(qr35_pred,  color='#f59e0b', lw=1.8, ls='--', zorder=4,
               label=f'Conservative  {qr35_pred:,.0f}')
    ax.axvline(qr25_pred,  color='#ef4444', lw=1.8, ls=':', zorder=4,
               label=f'Min. plan  {qr25_pred:,.0f}')
    ax.axvline(HIST_MEAN,  color='#6b7280', lw=1.2, zorder=3,
               label=f'Hist. avg  {HIST_MEAN:,}')

    ax.set_xlim(0, STADIUM_CAPACITY)
    ax.set_ylim(-0.5, 0.8)
    ax.set_xlabel('Tickets', color='#9ca3af', fontsize=9)
    ax.tick_params(axis='x', colors='#9ca3af', labelsize=8)
    ax.tick_params(axis='y', left=False, labelleft=False)
    for spine in ax.spines.values():
        spine.set_edgecolor('#374151')

    ax.legend(loc='upper right', fontsize=8, facecolor='#1e2530',
              edgecolor='#374151', labelcolor='#d1d5db', ncol=4, framealpha=0.9)
    ax.text(HIST_MIN, 0.42, f'Historical range\n{HIST_MIN:,} – {HIST_MAX:,}',
            color='#6b7280', fontsize=7, va='bottom', ha='left')
    ax.text(STADIUM_CAPACITY - 50, 0.42, f'Capacity\n{STADIUM_CAPACITY:,}',
            color='#6b7280', fontsize=7, va='bottom', ha='right')

    plt.tight_layout()
    return fig


# ── Sidebar ────────────────────────────────────────────────────────────────
def sidebar_inputs():
    st.sidebar.title("Match Parameters")

    advanced = st.sidebar.toggle("Advanced mode", value=False)
    st.sidebar.divider()

    inputs = {}

    # ── Match Setup ────────────────────────────────────────────────────────
    with st.sidebar.expander("Match Setup", expanded=True):
        opponent = st.selectbox("Opponent", ALL_OPPONENTS, index=0)
        inputs['opponent_grouped'] = opponent
        # Auto-fill historical strength — editable only in advanced mode
        if advanced:
            inputs['opp_ppg_vs_ohl'] = st.slider(
                "Opponent historical strength vs OHL", 0.0, 3.0,
                float(OPPONENT_PPG[opponent]), step=0.1,
                help="Points per game this opponent has historically earned against OHL.",
            )
        else:
            inputs['opp_ppg_vs_ohl'] = OPPONENT_PPG[opponent]
            st.caption(f"Opponent strength vs OHL: **{OPPONENT_PPG[opponent]:.2f}** (auto-filled)")

        inputs['kickoff_hour'] = st.selectbox(
            "Kickoff time", [16, 17, 18, 19, 20, 21],
            index=2, format_func=lambda h: f"{h}:00",
        )
        inputs['is_playoff'] = int(st.checkbox("Play-off match", value=False))

        # Date picker → derives is_weekend and academic_week automatically
        match_date = st.date_input(
            "Match date",
            value=datetime.date.today(),
            help="Used to determine if it's a weekend and the academic calendar week.",
        )
        inputs['is_weekend'] = int(match_date.weekday() >= 5)
        inputs['academic_week'] = match_date.isocalendar()[1]
        if not advanced:
            day_label = "Weekend" if inputs['is_weekend'] else "Weekday"
            st.caption(f"Detected: **{day_label}** · Week **{inputs['academic_week']}**")
        else:
            inputs['is_weekend'] = int(st.checkbox(
                "Weekend match", value=bool(inputs['is_weekend'])
            ))
            inputs['academic_week'] = st.number_input(
                "Academic calendar week", 1, 53, int(inputs['academic_week']),
            )

    # ── Team Form ──────────────────────────────────────────────────────────
    with st.sidebar.expander("Team Form", expanded=True):
        matchday = st.slider("Matchday", 1, 33, 16)
        inputs['matchday_normalized'] = matchday / 33.0

        inputs['ohl_season_points'] = st.number_input(
            "OHL season points so far", min_value=0, max_value=99, value=19,
        )
        inputs['ohl_points_per_game'] = st.slider(
            "OHL avg points per game this season", 0.0, 3.0, 1.0, step=0.05,
        )

        home_wins = st.radio(
            "OHL wins in last 3 home games", [0, 1, 2, 3],
            index=1, horizontal=True,
        )
        inputs['home_win_rate_last3'] = home_wins / 3.0

        inputs['last_h2h_result'] = st.radio(
            "Last head-to-head result", ['W', 'D', 'L', 'Unknown'],
            index=1, horizontal=True,
            help="Result of the last time OHL hosted this opponent.",
        )
        inputs['last_h2h_goal_margin'] = st.slider(
            "Last H2H goal difference (OHL − Opponent)", -5, 3, 0,
        )

    # ── Promotions & Buzz ──────────────────────────────────────────────────
    with st.sidebar.expander("Promotions & Buzz", expanded=True):
        inputs['rolling_avg_attendance_last3'] = st.slider(
            "Recent home attendance (3-game avg)", 4000, 10500, 6533, step=100,
            help="Average tickets scanned at OHL's last 3 home games.",
        )
        inputs['is_school_holiday_flanders'] = int(
            st.checkbox("Flemish school holiday week", value=False)
        )
        inputs['has_promotion'] = int(st.checkbox("Ticket promotion active", value=False))
        inputs['has_campaign'] = int(st.checkbox("Marketing campaign active", value=False))

        if advanced:
            inputs['pre_match_article_count'] = st.slider(
                "Press articles (7 days pre-match)", 1, 50, 12,
            )
            inputs['avg_ohl_interest_7d'] = st.slider(
                "Google Trends interest (7-day avg)", 1.8, 22.0, 5.6, step=0.1,
            )
            inputs['pre_match_interest_ratio'] = st.slider(
                "Buzz vs season average (ratio)", 0.3, 3.0, 1.0, step=0.05,
                help="1.0 = average, 2.0 = twice the usual interest.",
            )
        else:
            inputs['pre_match_article_count'] = st.slider(
                "Expected media coverage", 1, 50, 12,
                help="Roughly: 5 = quiet week, 15 = normal, 30+ = big match.",
            )
            buzz_label = st.select_slider(
                "Expected online buzz about this match",
                options=list(BUZZ_PRESETS.keys()),
                value="Normal",
            )
            inputs['avg_ohl_interest_7d'], inputs['pre_match_interest_ratio'] = (
                BUZZ_PRESETS[buzz_label]
            )

    # ── Weather ────────────────────────────────────────────────────────────
    with st.sidebar.expander("Weather", expanded=False):
        if advanced:
            inputs['weather_score'] = st.slider(
                "Weather quality score (composite)", -6.0, 0.6, -1.0, step=0.1,
                help="−6 = very poor conditions, +0.6 = excellent.",
            )
        else:
            weather_label = st.select_slider(
                "Expected weather conditions",
                options=list(WEATHER_PRESETS.keys()),
                value="Average",
            )
            inputs['weather_score'] = WEATHER_PRESETS[weather_label]

        inputs['weather_temp_deviation'] = st.slider(
            "Temperature vs seasonal norm (°C)", -7.0, 7.0, 0.0, step=0.5,
            help="0 = typical for the time of year. Negative = colder than usual.",
        )

    return inputs


# ── Main app ───────────────────────────────────────────────────────────────
def main():
    models = load_models()

    st.title("OHL Matchday Attendance Predictor")
    st.caption("Adjust match parameters in the sidebar to get an attendance forecast.")

    inputs = sidebar_inputs()

    # ── Predict ────────────────────────────────────────────────────────────
    X_ridge = build_ridge_input(inputs, models['cols_ridge'])
    X_qr    = build_qr_input(inputs, models['cols_qr'])

    ridge_pred = max(0, float(models['ridge'].predict(X_ridge)[0]))
    qr35_pred  = max(0, float(models['qr35'].predict(X_qr)[0]))
    qr25_pred  = max(0, float(models['qr25'].predict(X_qr)[0]))

    # ── Warning ────────────────────────────────────────────────────────────
    st.warning(
        "**Guidance only:** Trained on 71 matches. Typical error is ±1,000 tickets. "
        "Use for directional planning, not precise forecasting.",
        icon="⚠️",
    )

    # ── Metric cards ───────────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="Expected Attendance",
            value=f"{ridge_pred:,.0f}",
            delta=f"{ridge_pred - HIST_MEAN:+,.0f} vs historical average",
        )
        st.caption("Best single-number estimate")

    with col2:
        st.metric(label="Conservative Estimate", value=f"{qr35_pred:,.0f}")
        st.caption("Plan staffing & catering around this — 65% of matches exceeded this level")

    with col3:
        st.metric(label="Minimum Planning Figure", value=f"{qr25_pred:,.0f}")
        st.caption("Worst-case baseline — 75% of matches exceeded this level")

    st.divider()

    # ── Chart ──────────────────────────────────────────────────────────────
    st.subheader("Prediction range")
    fig = make_chart(ridge_pred, qr35_pred, qr25_pred)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # ── About expander ─────────────────────────────────────────────────────
    with st.expander("About this model"):
        st.markdown("""
**Models trained on 71 OHL home matches** (seasons 2022/23 – 2025/26).

| Output | What it means |
|--------|--------------|
| Expected Attendance | Ridge regression point estimate — best single number |
| Conservative Estimate | 65% of past matches had higher attendance than this |
| Minimum Planning Figure | 75% of past matches had higher attendance than this |

**Accuracy:** Average error is approximately ±1,000 tickets on held-out matches.
The 2025/26 season had 18% lower attendance than earlier seasons — the model
doesn't fully capture this shift, so treat predictions as directional guidance.

**What the model considers most important (in order):**
1. Recent home attendance (3-game average)
2. How far into the season the match is (late-season games draw more)
3. OHL's current league form (points per game)
4. Whether there's a ticket promotion active
5. The opponent
        """)


if __name__ == "__main__":
    main()
