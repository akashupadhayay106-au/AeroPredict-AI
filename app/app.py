import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from src.config import SENSOR_COLUMNS, SETTING_COLUMNS
from src.inference import EnginePredictor
from src.explainability import explain_prediction
from llm.gemini_client import GeminiAdvisor
from src.feature_engineering import add_rolling_features

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="AeroPredict AI",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .metric-card {
        background-color: #1E1E1E;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        margin-bottom: 20px;
        border-top: 4px solid #4CAF50;
    }
    .metric-card.warning { border-top-color: #FFC107; }
    .metric-card.critical { border-top-color: #F44336; }
    .metric-title { font-size: 1rem; color: #888; font-weight: bold; margin-bottom: 5px; }
    .metric-value { font-size: 2rem; font-weight: 800; color: #FFF; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# BACKEND INITIALIZATION
# ==========================================
@st.cache_resource
def load_predictor():
    return EnginePredictor()

predictor = load_predictor()
advisor = GeminiAdvisor() # Do not cache this, so it always checks st.secrets dynamically

# ==========================================
# HEADER & SYSTEM STATUS
# ==========================================
st.title("✈️ AeroPredict AI")
st.markdown("##### Intelligent Aircraft Engine Health & Predictive Maintenance System")

sys_col1, sys_col2, sys_col3, sys_col4 = st.columns(4)
model_status = "🟢 Ready (GRU)" if predictor.model else "🔴 Offline"
api_status = "🟢 Connected" if advisor.client else "🟡 Optional (Offline)"

sys_col1.metric("Model Status", model_status)
sys_col2.metric("Inference Engine", "🟢 Ready")
sys_col3.metric("Data Pipeline", "🟢 Ready")
sys_col4.metric("AI Advisor", api_status)

st.divider()

# ==========================================
# SIDEBAR CONFIGURATION
# ==========================================
with st.sidebar:
    st.header("⚙️ Engine Configuration")
    engine_id = st.number_input("Engine ID", min_value=1, max_value=1000, value=1, help="Unique identifier for the engine asset.")
    
    st.header("🔄 Input Mode")
    input_mode = st.radio("Select Data Source:", ["Simulation Mode", "Manual Input"])
    
    if input_mode == "Simulation Mode":
        cycles_to_sim = st.slider("Cycles to Simulate", min_value=5, max_value=250, value=50, help="Generate synthetic degradation history.")
        degradation_rate = st.select_slider("Degradation Profile", options=["Low", "Normal", "High"], value="Normal")
    else:
        current_cycle = st.number_input("Current Cycle", min_value=1, value=100)

    st.markdown("---")
    st.info("AeroPredict AI runs a monolithic inference pipeline directly within Streamlit for rapid, serverless deployment.")

# ==========================================
# MAIN INTERFACE
# ==========================================
if predictor.model is None:
    st.error("Critical System Failure: Prediction models could not be loaded. Please ensure models are trained and present in `models/ml/`.")
    st.stop()

st.subheader("📊 Engine Sensor Telemetry")

df = None

if input_mode == "Simulation Mode":
    st.write("Synthetic run-to-failure scenario generation active.")
    if st.button("🔮 Simulate & Predict RUL", type="primary", use_container_width=True):
        with st.spinner("Generating scenario data & running inference pipeline..."):
            history = []
            rate_mult = {"Low": 0.05, "Normal": 0.15, "High": 0.3}[degradation_rate]
            
            for i in range(1, cycles_to_sim + 1):
                row = {"time_cycles": float(i), "unit_number": engine_id}
                
                # C-MAPSS typical settings (mocked constant for simplicity)
                row["setting_1"] = -0.0015 + np.random.normal(0, 0.001)
                row["setting_2"] = 0.0003 + np.random.normal(0, 0.0001)
                row["setting_3"] = 100.0
                
                # Mock sensors (some go up, some go down as in C-MAPSS)
                for col in SENSOR_COLUMNS:
                    if col in ['sensor_2', 'sensor_3', 'sensor_4', 'sensor_8', 'sensor_11', 'sensor_13', 'sensor_15', 'sensor_17']:
                        row[col] = 10.0 + (i * rate_mult) + np.random.normal(0, 0.5) # Degrading upwards
                    elif col in ['sensor_7', 'sensor_12', 'sensor_20', 'sensor_21']:
                        row[col] = 50.0 - (i * rate_mult) + np.random.normal(0, 0.5) # Degrading downwards
                    else:
                        row[col] = 20.0 + np.random.normal(0, 0.1) # Mostly flat/constant
                        
                history.append(row)
            df = pd.DataFrame(history)
            st.session_state['sim_df'] = df
            
else:
    # Manual Input Mode (Grouped Expanders)
    st.write("Enter the latest telemetry readings manually.")
    with st.form("manual_input_form"):
        st.markdown("##### 🎛️ Operating Conditions")
        col_s1, col_s2, col_s3 = st.columns(3)
        s1 = col_s1.number_input("Setting 1 (Altitude)", value=0.0)
        s2 = col_s2.number_input("Setting 2 (Mach)", value=0.0)
        s3 = col_s3.number_input("Setting 3 (TRA)", value=100.0)
        
        st.markdown("##### 🌡️ Core Sensors (Subset)")
        sc1, sc2, sc3, sc4 = st.columns(4)
        sens_inputs = {}
        # Only show a subset for manual entry to not overwhelm, fill rest with defaults
        sens_inputs['sensor_2'] = sc1.number_input("Sensor 2 (T24)", value=642.1)
        sens_inputs['sensor_3'] = sc2.number_input("Sensor 3 (T30)", value=1589.7)
        sens_inputs['sensor_4'] = sc3.number_input("Sensor 4 (T50)", value=1400.2)
        sens_inputs['sensor_7'] = sc4.number_input("Sensor 7 (P50)", value=553.3)
        sens_inputs['sensor_11'] = sc1.number_input("Sensor 11 (Ps30)", value=47.5)
        sens_inputs['sensor_12'] = sc2.number_input("Sensor 12 (phi)", value=521.3)
        sens_inputs['sensor_15'] = sc3.number_input("Sensor 15 (BPR)", value=8.4)
        sens_inputs['sensor_21'] = sc4.number_input("Sensor 21 (W32)", value=39.0)
        
        submitted = st.form_submit_button("🔮 Predict RUL", type="primary", use_container_width=True)
        if submitted:
            # We must construct a DataFrame that has enough history for rolling features.
            # If manual, we duplicate the row to satisfy min_periods=1, but warn the user.
            row = {"time_cycles": float(current_cycle), "unit_number": engine_id}
            row["setting_1"] = s1
            row["setting_2"] = s2
            row["setting_3"] = s3
            for col in SENSOR_COLUMNS:
                row[col] = sens_inputs.get(col, 20.0) # default fallback
                
            df = pd.DataFrame([row]*10) # Duplicate 10 times to satisfy rolling windows
            df['time_cycles'] = [current_cycle - 9 + i for i in range(10)]
            st.session_state['sim_df'] = df

# ==========================================
# PREDICTION & RESULTS DISPLAY
# ==========================================
if 'sim_df' in st.session_state:
    df_to_predict = st.session_state['sim_df']
    
    with st.spinner("Executing Inference Pipeline..."):
        risk_info = predictor.predict(df_to_predict)
        
    if "error" in risk_info:
        st.error(f"Prediction Pipeline Failed.")
        st.warning(f"Validation Error: {risk_info.get('error')}")
        if 'details' in risk_info:
            st.code(risk_info['details'])
    else:
        st.success("✅ Input validated. Features engineered. Inference complete.")
        st.markdown("---")
        
        # Results Header
        rul_val = risk_info.get('rul', 0)
        health_cat = risk_info.get('health_status', 'UNKNOWN')
        
        # Determine CSS class
        card_class = "metric-card"
        if health_cat == "WARNING": card_class += " warning"
        elif health_cat == "CRITICAL": card_class += " critical"
            
        col_res1, col_res2, col_res3 = st.columns(3)
        
        with col_res1:
            st.markdown(f"""
            <div class="{card_class}">
                <div class="metric-title">REMAINING USEFUL LIFE</div>
                <div class="metric-value">{rul_val:.1f} Cycles</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col_res2:
            st.markdown(f"""
            <div class="{card_class}" title="HEALTHY: RUL > 50 | WARNING: 25 < RUL <= 50 | CRITICAL: RUL <= 25">
                <div class="metric-title">ENGINE HEALTH ℹ️</div>
                <div class="metric-value">{health_cat}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col_res3:
            st.markdown(f"""
            <div class="{card_class}">
                <div class="metric-title">RISK LEVEL</div>
                <div class="metric-value">{risk_info.get('risk_level', 'UNKNOWN')}</div>
            </div>
            """, unsafe_allow_html=True)
            
        # Explanations & Advisor
        tab_shap, tab_ai, tab_chat, tab_trend = st.tabs(["🔍 Why this prediction?", "🤖 AI Maintenance Advisor", "💬 Interactive AI Assistant", "📈 Historical Trend"])
        
        with tab_shap:
            st.markdown("#### Top Factors Affecting RUL")
            st.write("SHAP (SHapley Additive exPlanations) values indicate the relative impact of each sensor reading on the final RUL prediction.")
            processed_df = add_rolling_features(df_to_predict)
            current_features = processed_df.iloc[-1:][predictor.features]
            top_features = explain_prediction(predictor.model, current_features)
            
            if top_features:
                df_exp = pd.DataFrame(top_features)
                if not df_exp.empty:
                    fig = px.bar(
                        df_exp, 
                        x='impact', 
                        y='name', 
                        orientation='h',
                        color='impact',
                        color_continuous_scale=px.colors.diverging.RdBu[::-1],
                        title='Feature Contribution (Positive = Increases RUL, Negative = Decreases RUL)'
                    )
                    fig.update_layout(yaxis={'categoryorder':'total ascending'}, height=400)
                    st.plotly_chart(fig, use_container_width=True)
            else:
                top_features = []
                st.info("Explainability not available for this model type.")
                
        with tab_ai:
            st.markdown("#### Engineering Recommendation")
            with st.spinner("Consulting Gemini AI..."):
                recommendation = advisor.get_recommendation(risk_info, top_features)
                if recommendation:
                    st.info(recommendation)
                else:
                    st.warning("AI Advisor unavailable. Core ML prediction remains operational.")
                    
        with tab_chat:
            st.markdown("#### Chat with AeroPredict AI")
            st.write("Ask questions about the current engine status, SHAP explainability, or general maintenance!")
            
            if "messages" not in st.session_state:
                st.session_state.messages = []
                
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    
            if prompt := st.chat_input("E.g., Why is the risk level high?"):
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)
                    
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        response = advisor.chat_with_context(prompt, st.session_state.messages[:-1], risk_info, top_features)
                        st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
                    
        with tab_trend:
            st.markdown("#### Scenario Telemetry Trend")
            if input_mode == "Simulation Mode" and len(df_to_predict) > 1:
                # Plot Sensor 2 and Sensor 7 as examples
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df_to_predict['time_cycles'], y=df_to_predict['sensor_2'], name='Sensor 2 (Temp)'))
                fig.add_trace(go.Scatter(x=df_to_predict['time_cycles'], y=df_to_predict['sensor_7'], name='Sensor 7 (Pressure)', yaxis="y2"))
                
                fig.update_layout(
                    title="Engine Degradation Profile",
                    xaxis_title="Cycles",
                    yaxis_title="Sensor 2",
                    yaxis2=dict(title="Sensor 7", overlaying="y", side="right"),
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.write("Trend visualization requires simulation mode with multiple cycles.")
