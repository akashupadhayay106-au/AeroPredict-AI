import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from src.config import SENSOR_COLUMNS, SETTING_COLUMNS, RISK_THRESHOLDS
from src.inference import EnginePredictor
from src.explainability import explain_prediction
from llm.gemini_client import GeminiAdvisor
from src.feature_engineering import add_rolling_features

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="AeroPredict AI — Engine Health & Maintenance",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# GLOBAL CSS
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Hero */
.hero-container {
    text-align: center;
    padding: 2rem 1rem 1.5rem 1rem;
}
.hero-title {
    font-size: 2.8rem;
    font-weight: 900;
    background: linear-gradient(135deg, #60A5FA, #A78BFA, #F472B6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.3rem;
    letter-spacing: -1px;
}
.hero-subtitle {
    font-size: 1.15rem;
    color: #94A3B8;
    font-weight: 400;
    margin-bottom: 1.5rem;
}

/* Feature Cards */
.feature-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 1rem;
    margin: 1.5rem 0;
}
.feature-card {
    background: linear-gradient(145deg, #1E293B, #0F172A);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
    transition: transform 0.2s, border-color 0.2s;
}
.feature-card:hover {
    transform: translateY(-4px);
    border-color: #60A5FA;
}
.feature-icon {
    font-size: 2rem;
    margin-bottom: 0.5rem;
}
.feature-card h4 {
    color: #E2E8F0;
    font-size: 1rem;
    font-weight: 700;
    margin: 0 0 0.4rem 0;
}
.feature-card p {
    color: #94A3B8;
    font-size: 0.85rem;
    margin: 0;
    line-height: 1.4;
}

/* Result Cards */
.result-card {
    background: linear-gradient(145deg, #1E293B, #0F172A);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
    margin-bottom: 1rem;
}
.result-card.healthy { border-left: 5px solid #22C55E; }
.result-card.warning { border-left: 5px solid #F59E0B; }
.result-card.critical { border-left: 5px solid #EF4444; }
.result-card.monitoring { border-left: 5px solid #3B82F6; }
.result-label {
    color: #94A3B8;
    font-size: 0.8rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 0.3rem;
}
.result-value {
    font-size: 2.2rem;
    font-weight: 900;
    color: #F1F5F9;
}
.result-value.healthy-text { color: #22C55E; }
.result-value.warning-text { color: #F59E0B; }
.result-value.critical-text { color: #EF4444; }
.result-value.monitoring-text { color: #3B82F6; }

/* Steps */
.step-row {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-bottom: 0.5rem;
}
.step-num {
    background: linear-gradient(135deg, #3B82F6, #8B5CF6);
    color: #FFF;
    border-radius: 50%;
    width: 32px;
    height: 32px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 800;
    font-size: 0.85rem;
    flex-shrink: 0;
}
.step-text {
    color: #CBD5E1;
    font-size: 0.9rem;
}

/* Status pill */
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
}
.status-green { background: rgba(34,197,94,0.15); color: #22C55E; }
.status-yellow { background: rgba(245,158,11,0.15); color: #F59E0B; }
.status-red { background: rgba(239,68,68,0.15); color: #EF4444; }

/* Disclaimer */
.disclaimer {
    background: rgba(59, 130, 246, 0.08);
    border: 1px solid rgba(59, 130, 246, 0.2);
    border-radius: 8px;
    padding: 1rem;
    color: #94A3B8;
    font-size: 0.82rem;
    line-height: 1.5;
    margin-top: 1rem;
}

/* Chat context */
.chat-context {
    background: #1E293B;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 0.8rem;
    font-size: 0.78rem;
    color: #94A3B8;
    margin-bottom: 1rem;
}
.chat-context strong { color: #E2E8F0; }

/* Tech badge */
.tech-badges {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin: 0.5rem 0;
}
.tech-badge {
    background: #1E293B;
    border: 1px solid #334155;
    border-radius: 6px;
    padding: 4px 10px;
    font-size: 0.78rem;
    color: #94A3B8;
    font-weight: 500;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# BACKEND INITIALIZATION
# ============================================================
@st.cache_resource
def load_predictor():
    return EnginePredictor()

predictor = load_predictor()
advisor = GeminiAdvisor()  # Never cached — always checks st.secrets fresh

# ============================================================
# SIDEBAR NAVIGATION
# ============================================================
with st.sidebar:
    st.markdown("### ✈️ AeroPredict AI")
    page = st.radio(
        "Navigate",
        ["🏠 Welcome", "🔧 Engine Analysis", "📘 User Guide", "🛠 Technical Overview"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    
    # Gemini Status (always visible)
    if advisor.client:
        st.markdown('<span class="status-pill status-green">🤖 AI Advisor: Available</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-pill status-yellow">🤖 AI Advisor: Not Configured</span>', unsafe_allow_html=True)
        st.caption("Add GEMINI_API_KEY in Streamlit Secrets to enable AI recommendations.")
    
    model_ok = predictor.model is not None
    if model_ok:
        st.markdown('<span class="status-pill status-green">🧠 ML Model: Ready</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-pill status-red">🧠 ML Model: Offline</span>', unsafe_allow_html=True)


# ████████████████████████████████████████████████████████████
#  PAGE: WELCOME / LANDING
# ████████████████████████████████████████████████████████████
if page == "🏠 Welcome":

    # Hero
    st.markdown("""
    <div class="hero-container">
        <div class="hero-title">✈️ AeroPredict AI</div>
        <div class="hero-subtitle">
            Intelligent Aircraft Engine Health & Predictive Maintenance System
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    AeroPredict AI estimates how many operating cycles an aircraft engine may have
    remaining before it requires maintenance attention. It combines **machine learning**,
    **deep learning**, **explainable AI (SHAP)**, and **generative AI (Gemini)** into
    a single, easy-to-use dashboard.
    """)

    # Feature Cards
    st.markdown("""
    <div class="feature-grid">
        <div class="feature-card">
            <div class="feature-icon">🔮</div>
            <h4>RUL Prediction</h4>
            <p>Estimate remaining engine operating cycles using a trained GRU deep learning model.</p>
        </div>
        <div class="feature-card">
            <div class="feature-icon">🧠</div>
            <h4>AI Explainability</h4>
            <p>Understand which sensors and operating conditions influence each individual prediction via SHAP.</p>
        </div>
        <div class="feature-card">
            <div class="feature-icon">⚠️</div>
            <h4>Health & Risk</h4>
            <p>Convert the numeric prediction into a clear health status and risk level for rapid decision-making.</p>
        </div>
        <div class="feature-card">
            <div class="feature-icon">🤖</div>
            <h4>AI Maintenance Advisor</h4>
            <p>Ask questions and receive context-aware maintenance guidance powered by Google Gemini.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # How It Works
    st.markdown("### How It Works")
    st.markdown("""
    <div class="step-row"><div class="step-num">1</div><div class="step-text"><strong>Engine Data</strong> — Provide operating conditions and sensor measurements (simulated or manual).</div></div>
    <div class="step-row"><div class="step-num">2</div><div class="step-text"><strong>Feature Processing</strong> — Rolling statistics and engineered features are computed automatically.</div></div>
    <div class="step-row"><div class="step-num">3</div><div class="step-text"><strong>ML/DL Prediction</strong> — A trained ML model predicts the Remaining Useful Life (RUL).</div></div>
    <div class="step-row"><div class="step-num">4</div><div class="step-text"><strong>Explanation & Insight</strong> — SHAP explains the prediction; Gemini AI provides maintenance guidance.</div></div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # What Are We Predicting?
    st.markdown("### 🎯 What Is AeroPredict AI Predicting?")
    st.markdown("""
    **Remaining Useful Life (RUL)** represents the estimated number of operating cycles
    an engine can continue operating before reaching the defined degradation threshold
    used by the predictive model.

    > **Example:** *Predicted RUL = 42 cycles* means the model estimates approximately
    > 42 remaining operating cycles under the modelled scenario.
    """)
    st.markdown("""
    <div class="disclaimer">
    ⚠️ <strong>Important:</strong> This is an AI-based predictive estimate, not a certified aviation
    maintenance decision or substitute for approved engineering procedures. Predictions should be
    interpreted as decision-support estimates only.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Dataset
    st.markdown("### 📊 What Data Does the Model Use?")
    st.markdown("""
    The model is built around the **NASA C-MAPSS** (Commercial Modular Aero-Propulsion
    System Simulation) turbofan engine degradation dataset. Each record contains:
    - **Engine ID** and **operating cycle** count
    - **3 operational settings** (altitude, Mach number, throttle resolver angle)
    - **21 sensor measurements** (temperatures, pressures, speeds, flow rates)
    - **Historical degradation patterns** learned from run-to-failure trajectories
    """)
    with st.expander("View Technical Details"):
        st.markdown(f"""
        | Detail | Value |
        |---|---|
        | Dataset | NASA C-MAPSS FD001 |
        | Engines (train) | ~100 run-to-failure units |
        | Features (raw) | 3 settings + 21 sensors = 24 |
        | Engineered features | Rolling mean/std (windows 5, 10) |
        | Total model features | {len(predictor.features) if predictor.features else 'N/A'} |
        | Target | RUL (capped at 125 cycles) |
        | Production model | LightGBM (best ML) |
        | Preprocessing | Constant-column removal, rolling aggregation |
        """)

    st.markdown("---")

    # Model Info
    st.markdown("### 🧠 AI Model Behind the Prediction")
    st.markdown("""
    The production inference model is a **tree-based ML model** (best performer from the comparison study)
    trained on engineered time-series features from the C-MAPSS dataset.
    """)

    st.markdown("**Models Compared During Development:**")
    comparison_data = {
        "Model": ["Linear Regression", "Random Forest", "XGBoost", "LightGBM", "GRU (Deep Learning)"],
        "Type": ["Classical ML", "Classical ML", "Classical ML", "Classical ML", "Deep Learning"],
        "Role": ["Comparison", "Comparison", "Comparison", "Production Inference", "Comparison (Sequence)"]
    }
    st.dataframe(pd.DataFrame(comparison_data), use_container_width=True, hide_index=True)

    with st.expander("📈 Model Evaluation Metrics"):
        st.markdown("""
        | Metric | LightGBM (ML) | GRU (DL) |
        |---|---|---|
        | MAE | 43.90 | 36.17 |
        | RMSE | 59.10 | 41.85 |
        | R² | −0.004 | −0.014 |

        **What do these metrics mean?**
        - **MAE** (Mean Absolute Error) — average absolute prediction error in cycles.
        - **RMSE** (Root Mean Squared Error) — penalises large errors more strongly.
        - **R²** (Coefficient of Determination) — goodness-of-fit measure. A negative value
          indicates the model has not outperformed a naïve mean predictor on the test set,
          which is common in C-MAPSS due to early-cycle sensor noise.
        """)
        st.markdown("""
        <div class="disclaimer">
        Model performance depends on the training data and operating scenario. Predictions
        should be interpreted as decision-support estimates, not precise engineering forecasts.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    if st.button("🚀 Start Engine Analysis →", type="primary", use_container_width=True):
        st.session_state['nav_to_analysis'] = True
        st.rerun()


# ████████████████████████████████████████████████████████████
#  PAGE: ENGINE ANALYSIS
# ████████████████████████████████████████████████████████████
elif page == "🔧 Engine Analysis":

    if predictor.model is None:
        st.error("⚠️ Prediction model could not be loaded. Ensure model files exist in `models/ml/`.")
        st.stop()

    st.markdown("## 🔧 Engine Analysis")
    st.caption("Configure, simulate, and predict the Remaining Useful Life of an engine.")

    # ── Step 1: Engine & Mode ──
    with st.sidebar:
        st.markdown("### 🔧 Analysis Config")
        engine_id = st.number_input("Engine ID", min_value=1, max_value=1000, value=1,
                                    help="Unique identifier for the engine asset.")
        input_mode = st.radio("Data Source", ["🔄 Simulation", "✏️ Manual Input"])
        
        if input_mode == "🔄 Simulation":
            cycles_to_sim = st.slider("Cycles to Simulate", 5, 250, 50,
                                      help="Number of synthetic degradation cycles to generate.")
            degradation_rate = st.select_slider("Degradation Profile",
                                                ["Low", "Normal", "High"], value="Normal")
        else:
            current_cycle = st.number_input("Current Cycle", min_value=1, value=100)

    # ── Data Input Area ──
    df = None

    if input_mode == "🔄 Simulation":
        st.info("**Simulation Mode** — Generates a synthetic run-to-failure scenario with realistic C-MAPSS sensor distributions. Data is clearly labelled as simulated.")

        if st.button("🔮 Simulate & Predict Remaining Useful Life", type="primary", use_container_width=True):
            with st.spinner("Generating scenario data & running inference pipeline..."):
                progress = st.progress(0, "Validating engine configuration...")
                history = []
                rate_mult = {"Low": 0.05, "Normal": 0.15, "High": 0.3}[degradation_rate]

                for i in range(1, cycles_to_sim + 1):
                    row = {"time_cycles": float(i), "unit_number": engine_id}
                    row["setting_1"] = -0.0015 + np.random.normal(0, 0.001)
                    row["setting_2"] = 0.0003 + np.random.normal(0, 0.0001)
                    row["setting_3"] = 100.0
                    for col in SENSOR_COLUMNS:
                        if col in ['sensor_2', 'sensor_3', 'sensor_4', 'sensor_8',
                                    'sensor_11', 'sensor_13', 'sensor_15', 'sensor_17']:
                            row[col] = 10.0 + (i * rate_mult) + np.random.normal(0, 0.5)
                        elif col in ['sensor_7', 'sensor_12', 'sensor_20', 'sensor_21']:
                            row[col] = 50.0 - (i * rate_mult) + np.random.normal(0, 0.5)
                        else:
                            row[col] = 20.0 + np.random.normal(0, 0.1)
                    history.append(row)

                progress.progress(30, "Preparing features...")
                df = pd.DataFrame(history)
                st.session_state['sim_df'] = df
                st.session_state['sim_mode'] = True
                progress.progress(50, "Running model inference...")

    else:
        st.info("**Manual Input Mode** — Enter the latest telemetry readings. Sensors you don't specify will use nominal defaults.")

        with st.form("manual_input_form"):
            st.markdown("#### Step 1 — Operating Conditions")
            oc1, oc2, oc3 = st.columns(3)
            s1 = oc1.number_input("Setting 1 (Altitude)", value=0.0, format="%.4f",
                                  help="Operational altitude setting.")
            s2 = oc2.number_input("Setting 2 (Mach Number)", value=0.0, format="%.4f",
                                  help="Mach number setting.")
            s3 = oc3.number_input("Setting 3 (TRA)", value=100.0, format="%.1f",
                                  help="Throttle Resolver Angle.")

            st.markdown("#### Step 2 — Key Sensors")
            sc1, sc2, sc3, sc4 = st.columns(4)
            sens_inputs = {}
            sens_inputs['sensor_2']  = sc1.number_input("Sensor 2 (T24 — Total temp. at LPC outlet)", value=642.1)
            sens_inputs['sensor_3']  = sc2.number_input("Sensor 3 (T30 — Total temp. at HPC outlet)", value=1589.7)
            sens_inputs['sensor_4']  = sc3.number_input("Sensor 4 (T50 — Total temp. at LPT outlet)", value=1400.2)
            sens_inputs['sensor_7']  = sc4.number_input("Sensor 7 (P50 — Total pressure at LPT outlet)", value=553.3)
            sens_inputs['sensor_11'] = sc1.number_input("Sensor 11 (Ps30 — Static pressure at HPC outlet)", value=47.5)
            sens_inputs['sensor_12'] = sc2.number_input("Sensor 12 (phi — Fuel/air ratio)", value=521.3)
            sens_inputs['sensor_15'] = sc3.number_input("Sensor 15 (BPR — Bypass ratio)", value=8.4)
            sens_inputs['sensor_21'] = sc4.number_input("Sensor 21 (W32 — HPT bleed enthalpy)", value=39.0)

            with st.expander("🔬 Advanced Sensor Inputs"):
                adv1, adv2, adv3 = st.columns(3)
                for idx, col in enumerate(SENSOR_COLUMNS):
                    if col not in sens_inputs:
                        target_col = [adv1, adv2, adv3][idx % 3]
                        sens_inputs[col] = target_col.number_input(f"{col}", value=20.0, key=f"adv_{col}")

            submitted = st.form_submit_button("🔮 Predict Remaining Useful Life",
                                              type="primary", use_container_width=True)
            if submitted:
                row = {"time_cycles": float(current_cycle), "unit_number": engine_id}
                row["setting_1"] = s1
                row["setting_2"] = s2
                row["setting_3"] = s3
                for col in SENSOR_COLUMNS:
                    row[col] = sens_inputs.get(col, 20.0)
                df = pd.DataFrame([row] * 10)
                df['time_cycles'] = [current_cycle - 9 + i for i in range(10)]
                st.session_state['sim_df'] = df
                st.session_state['sim_mode'] = False

    # ── PREDICTION RESULTS ──
    if 'sim_df' in st.session_state:
        df_to_predict = st.session_state['sim_df']
        is_sim = st.session_state.get('sim_mode', True)

        with st.spinner("Executing inference pipeline..."):
            risk_info = predictor.predict(df_to_predict)

        if "error" in risk_info:
            st.error("**Prediction could not be completed.**")
            st.warning(f"What happened: {risk_info.get('error')}")
            if 'details' in risk_info:
                with st.expander("Technical Details"):
                    st.code(risk_info['details'])
        else:
            # Update progress if simulation
            if 'progress' in dir():
                try:
                    progress.progress(80, "Calculating explainability...")
                except Exception:
                    pass

            rul_val = risk_info.get('rul', 0)
            health_cat = risk_info.get('health_status', 'UNKNOWN')
            risk_level = risk_info.get('risk_level', 'UNKNOWN')

            # Map to CSS classes
            health_css_map = {
                "HEALTHY": ("healthy", "healthy-text"),
                "MONITORING_SUGGESTED": ("monitoring", "monitoring-text"),
                "ATTENTION_REQUIRED": ("warning", "warning-text"),
                "IMMEDIATE_MAINTENANCE_REQUIRED": ("critical", "critical-text"),
            }
            card_css, text_css = health_css_map.get(health_cat, ("", ""))

            # Friendly health labels
            health_label_map = {
                "HEALTHY": "🟢 Healthy",
                "MONITORING_SUGGESTED": "🔵 Monitor",
                "ATTENTION_REQUIRED": "🟡 Warning",
                "IMMEDIATE_MAINTENANCE_REQUIRED": "🔴 Critical",
            }
            health_label = health_label_map.get(health_cat, health_cat)

            if is_sim:
                st.caption("⚠️ Results below are based on **simulated scenario data**, not real aircraft telemetry.")

            st.markdown("---")
            st.markdown("### Engine Health Result")

            rc1, rc2, rc3, rc4 = st.columns(4)
            with rc1:
                st.markdown(f"""<div class="result-card {card_css}">
                    <div class="result-label">Remaining Useful Life</div>
                    <div class="result-value {text_css}">{rul_val:.0f} Cycles</div>
                </div>""", unsafe_allow_html=True)
            with rc2:
                st.markdown(f"""<div class="result-card {card_css}">
                    <div class="result-label">Health Status</div>
                    <div class="result-value {text_css}">{health_label}</div>
                </div>""", unsafe_allow_html=True)
            with rc3:
                st.markdown(f"""<div class="result-card {card_css}">
                    <div class="result-label">Risk Level</div>
                    <div class="result-value">{risk_level}</div>
                </div>""", unsafe_allow_html=True)
            with rc4:
                st.markdown(f"""<div class="result-card">
                    <div class="result-label">Model</div>
                    <div class="result-value" style="font-size:1.4rem;">LightGBM</div>
                </div>""", unsafe_allow_html=True)

            # Summary table
            with st.expander("📋 Full Prediction Summary"):
                summary_df = pd.DataFrame({
                    "Metric": ["Engine ID", "Predicted RUL", "Health Status", "Risk Level",
                               "Current Cycle", "Data Source"],
                    "Result": [
                        str(risk_info.get('engine_id')),
                        f"{rul_val:.1f} cycles",
                        health_label,
                        risk_level,
                        str(risk_info.get('current_cycle')),
                        "Simulation" if is_sim else "Manual Input"
                    ]
                })
                st.dataframe(summary_df, use_container_width=True, hide_index=True)

            # Health threshold info
            st.info(f"""
            **Health Category Thresholds:**  
            🟢 Healthy — RUL > {RISK_THRESHOLDS['LOW']} · 
            🔵 Monitor — RUL > {RISK_THRESHOLDS['MEDIUM']} · 
            🟡 Warning — RUL > {RISK_THRESHOLDS['HIGH']} · 
            🔴 Critical — RUL ≤ {RISK_THRESHOLDS['HIGH']}  
            *These categories are rule-based interpretations of the predicted RUL, not labels generated by the model.*
            """)

            # ── SHAP, Health Story, AI Advisor, Chat, Trends ──
            # Pre-compute SHAP
            processed_df = add_rolling_features(df_to_predict)
            current_features = processed_df.iloc[-1:][predictor.features]
            top_features = explain_prediction(predictor.model, current_features)
            if not top_features:
                top_features = []

            tab_why, tab_story, tab_ai, tab_chat, tab_trend = st.tabs([
                "🔍 Why This Prediction?",
                "📝 Engine Health Story",
                "🤖 AI Advisor",
                "💬 AI Chat Assistant",
                "📈 Historical Trend"
            ])

            # Tab: SHAP
            with tab_why:
                st.markdown("#### Why did the model predict this?")
                st.markdown("These factors indicate which input features contributed most strongly to this individual prediction.")
                if top_features:
                    df_exp = pd.DataFrame(top_features)
                    fig = px.bar(
                        df_exp, x='impact', y='name', orientation='h',
                        color='impact',
                        color_continuous_scale=px.colors.diverging.RdBu[::-1],
                        title='Feature Contribution to Predicted RUL'
                    )
                    fig.update_layout(
                        yaxis={'categoryorder': 'total ascending'},
                        height=380,
                        xaxis_title="SHAP Impact (positive = increases RUL, negative = decreases RUL)",
                        yaxis_title="",
                        coloraxis_showscale=False,
                        template="plotly_dark"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Explainability not available for this model type.")

            # Tab: Engine Health Story
            with tab_story:
                st.markdown("#### Engine Health Summary")
                top_str = ", ".join([f['name'] for f in top_features[:3]]) if top_features else "various sensors"
                st.markdown(f"""
                Engine **{risk_info.get('engine_id')}** is currently estimated to have approximately
                **{rul_val:.0f} remaining cycles** under the {'simulated' if is_sim else 'provided'} scenario.

                The prediction is primarily influenced by: **{top_str}**.

                The current state falls within the **{health_label}** range based on the configured
                RUL thresholds (Healthy > {RISK_THRESHOLDS['LOW']}, Warning > {RISK_THRESHOLDS['HIGH']},
                Critical ≤ {RISK_THRESHOLDS['HIGH']}).
                """)
                st.markdown("""
                <div class="disclaimer">
                This summary is auto-generated from model output and SHAP analysis.
                It should be validated against approved maintenance procedures before any engineering action is taken.
                </div>
                """, unsafe_allow_html=True)

            # Tab: AI Advisor
            with tab_ai:
                st.markdown("#### Engineering Recommendation")
                if not advisor.client:
                    st.warning("🤖 AI Advisor is **not configured**. Add `GEMINI_API_KEY` to Streamlit Secrets to enable this feature.")
                    st.info("Core ML prediction, SHAP analysis, and health assessment remain fully operational.")
                else:
                    if st.button("Generate AI Recommendation", key="gen_rec"):
                        with st.spinner("Consulting Gemini AI..."):
                            try:
                                recommendation = advisor.get_recommendation(risk_info, top_features)
                                st.session_state['last_recommendation'] = recommendation
                            except Exception as e:
                                st.session_state['last_recommendation'] = None
                                st.warning(f"🟡 AI Advisor temporarily unavailable. Please try again shortly.")
                                st.caption(f"Technical detail: {str(e)[:150]}")

                    if 'last_recommendation' in st.session_state and st.session_state['last_recommendation']:
                        st.markdown(st.session_state['last_recommendation'])
                    
                    st.markdown("""
                    <div class="disclaimer">
                    AI recommendations are informational and should be validated against approved maintenance procedures.
                    </div>
                    """, unsafe_allow_html=True)

            # Tab: Chat
            with tab_chat:
                st.markdown("#### Chat with AeroPredict AI")
                st.markdown("Ask questions about the current engine status, SHAP factors, or maintenance guidance.")

                # Context panel
                ctx_features = ", ".join([f['name'] for f in top_features[:3]]) if top_features else "N/A"
                st.markdown(f"""
                <div class="chat-context">
                <strong>Current Context:</strong> Engine {risk_info.get('engine_id')} · 
                RUL: {rul_val:.0f} cycles · Risk: {risk_level} · 
                Health: {health_label} · Top Factors: {ctx_features}
                </div>
                """, unsafe_allow_html=True)

                if not advisor.client:
                    st.warning("🤖 Chat requires a configured Gemini API key. Add `GEMINI_API_KEY` in Streamlit Secrets.")
                else:
                    if "messages" not in st.session_state:
                        st.session_state.messages = []

                    for msg in st.session_state.messages:
                        with st.chat_message(msg["role"]):
                            st.markdown(msg["content"])

                    if prompt := st.chat_input("E.g., Which sensor should I monitor most closely?"):
                        st.session_state.messages.append({"role": "user", "content": prompt})
                        with st.chat_message("user"):
                            st.markdown(prompt)

                        with st.chat_message("assistant"):
                            with st.spinner("Thinking..."):
                                try:
                                    response = advisor.chat_with_context(
                                        prompt, st.session_state.messages[:-1],
                                        risk_info, top_features
                                    )
                                except Exception as e:
                                    response = f"I'm temporarily unable to respond. Please try again shortly. ({str(e)[:80]})"
                                st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})

                    st.markdown("""
                    <div class="disclaimer">
                    AI recommendations are generated by Google Gemini and are informational only.
                    They should be validated against approved maintenance procedures.
                    </div>
                    """, unsafe_allow_html=True)

            # Tab: Trend
            with tab_trend:
                st.markdown("#### Scenario Telemetry Trend")
                if is_sim and len(df_to_predict) > 1:
                    st.caption("⚠️ This chart shows **simulated scenario data**, not real aircraft telemetry.")
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=df_to_predict['time_cycles'], y=df_to_predict['sensor_2'],
                        name='Sensor 2 (T24)', line=dict(color='#60A5FA')
                    ))
                    fig.add_trace(go.Scatter(
                        x=df_to_predict['time_cycles'], y=df_to_predict['sensor_11'],
                        name='Sensor 11 (Ps30)', line=dict(color='#F472B6')
                    ))
                    fig.add_trace(go.Scatter(
                        x=df_to_predict['time_cycles'], y=df_to_predict['sensor_7'],
                        name='Sensor 7 (P50)', yaxis="y2", line=dict(color='#34D399', dash='dot')
                    ))
                    fig.update_layout(
                        title="Simulated Engine Degradation Profile",
                        xaxis_title="Operating Cycle",
                        yaxis_title="Sensors 2 & 11",
                        yaxis2=dict(title="Sensor 7", overlaying="y", side="right"),
                        height=420,
                        template="plotly_dark",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Trend visualisation requires Simulation mode with multiple cycles.")


# ████████████████████████████████████████████████████████████
#  PAGE: USER GUIDE
# ████████████████████████████████████████████████████████████
elif page == "📘 User Guide":
    st.markdown("## 📘 User Guide")
    st.markdown("A step-by-step walkthrough of how to use AeroPredict AI.")

    steps = [
        ("1", "Select an Engine", "Choose an engine ID to identify the asset you're analysing."),
        ("2", "Provide Data", "Use **Simulation** to generate a synthetic degradation scenario, or **Manual Input** to enter the latest sensor readings."),
        ("3", "Predict RUL", "Click **Predict Remaining Useful Life**. The system validates your data, engineers features, and runs the ML model."),
        ("4", "Understand the Result", "Review the predicted **RUL** (in cycles), the **Health Status**, and the **Risk Level**."),
        ("5", "Understand Why", "Check the **SHAP** tab to see which sensors and settings contributed most to the prediction."),
        ("6", "Ask the AI", "Open the **AI Chat Assistant** tab to ask follow-up questions about the prediction or maintenance actions."),
        ("7", "Take Engineering Action", "Use the insights as **decision-support**. Validate all recommendations against approved maintenance procedures before acting."),
    ]

    for num, title, desc in steps:
        st.markdown(f"""
        <div class="step-row">
            <div class="step-num">{num}</div>
            <div class="step-text"><strong>{title}</strong> — {desc}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Frequently Asked Questions")
    with st.expander("What does RUL mean?"):
        st.markdown("**Remaining Useful Life** is the predicted number of operating cycles before the engine reaches the degradation threshold defined during model training.")
    with st.expander("Is this real aircraft data?"):
        st.markdown("No. The simulation mode generates synthetic data modelled after NASA C-MAPSS turbofan engine patterns. It is clearly labelled as simulated.")
    with st.expander("What happens if the AI Advisor is offline?"):
        st.markdown("All core predictions (RUL, Health, Risk, SHAP) continue to work normally. Only the Gemini-powered recommendations are affected.")
    with st.expander("How are health categories determined?"):
        st.markdown(f"They are rule-based thresholds: Healthy (RUL > {RISK_THRESHOLDS['LOW']}), Monitor (RUL > {RISK_THRESHOLDS['MEDIUM']}), Warning (RUL > {RISK_THRESHOLDS['HIGH']}), Critical (RUL ≤ {RISK_THRESHOLDS['HIGH']}). These are not model-generated labels.")


# ████████████████████████████████████████████████████████████
#  PAGE: TECHNICAL OVERVIEW
# ████████████████████████████████████████████████████████████
elif page == "🛠 Technical Overview":
    st.markdown("## 🛠 Technical Overview")
    st.markdown("Architecture, technology stack, and model details for technical stakeholders.")

    st.markdown("### System Architecture")
    st.markdown("""
    ```
    NASA C-MAPSS / Engine Data
            ↓
    Data Preparation & Feature Engineering
            ↓
    ML Model (LightGBM — Production)
            ↓
    RUL Prediction
            ↓
    ┌───────────────┬─────────────────────┐
    │ SHAP Explain. │ Gemini AI Advisor    │
    └───────────────┴─────────────────────┘
            ↓
    Streamlit Application (Public)
    ```
    """)

    st.markdown("### Technology Stack")
    st.markdown("""
    <div class="tech-badges">
        <span class="tech-badge">Python</span>
        <span class="tech-badge">Pandas</span>
        <span class="tech-badge">NumPy</span>
        <span class="tech-badge">Scikit-learn</span>
        <span class="tech-badge">XGBoost</span>
        <span class="tech-badge">LightGBM</span>
        <span class="tech-badge">PyTorch</span>
        <span class="tech-badge">SHAP</span>
        <span class="tech-badge">Plotly</span>
        <span class="tech-badge">Streamlit</span>
        <span class="tech-badge">Google Gemini</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Feature Pipeline")
    st.markdown(f"""
    | Stage | Detail |
    |---|---|
    | Raw inputs | 3 settings + 21 sensors per cycle |
    | Rolling features | Mean & Std with windows 5 and 10 |
    | Constant removal | Sensors with near-zero variance dropped |
    | Final feature count | {len(predictor.features) if predictor.features else 'N/A'} |
    """)

    st.markdown("### Model Feature Schema")
    if predictor.features:
        with st.expander(f"View all {len(predictor.features)} features"):
            for i, f in enumerate(predictor.features):
                st.text(f"{i+1:3d}. {f}")
    else:
        st.warning("Model features not loaded.")

    st.markdown("### Known Limitations")
    st.markdown("""
    - The model exhibits a negative R² on the test set, indicating high baseline variance relative to predictions.
      This is common in C-MAPSS due to early-cycle sensor noise before degradation manifests.
    - Predictions are based on the NASA C-MAPSS simulated dataset and should not be directly applied to
      real-world engine maintenance without retraining on operational data.
    - The Gemini AI Advisor depends on an external API and may be temporarily unavailable during high-demand periods.
    - Health status categories are rule-based heuristics, not model-generated classifications.
    """)

# ── Handle navigation from Welcome CTA ──
if st.session_state.get('nav_to_analysis'):
    st.session_state['nav_to_analysis'] = False
