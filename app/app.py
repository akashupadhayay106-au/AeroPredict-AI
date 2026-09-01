import streamlit as st
import pandas as pd
import plotly.express as px
from src.config import SENSOR_COLUMNS
from src.inference import EnginePredictor
from src.explainability import explain_prediction
from llm.gemini_client import GeminiAdvisor
from src.feature_engineering import add_rolling_features

# Cache models to prevent reloading on every Streamlit interaction
@st.cache_resource
def load_backend():
    return EnginePredictor(), GeminiAdvisor()

predictor, advisor = load_backend()

st.set_page_config(page_title="AeroPredict AI", page_icon="✈️", layout="wide")

st.title("✈️ AeroPredict AI")
st.markdown("### Intelligent Aircraft Engine Health & Maintenance System")

# Sidebar for controls
with st.sidebar:
    st.header("Engine Selection")
    engine_id = st.number_input("Enter Engine ID", min_value=1, max_value=500, value=1)
    st.markdown("---")
    st.info("This dashboard runs locally with direct model integration, delivering real-time RUL predictions, SHAP explainability, and Gemini LLM maintenance recommendations.")

# Try to check backend health
if predictor.model is not None:
    st.success("Backend Status: HEALTHY (Model Loaded: True)")
else:
    st.error("Model could not be loaded. Please ensure models are trained.")
    st.stop()

st.markdown("---")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Data Input")
    st.write("Simulate engine history data. For a real scenario, this would stream from sensors.")
    
    # Simple form to simulate some cycles
    cycles = st.slider("Cycles to simulate", 1, 50, 10)
    
    # Create dummy data for the selected number of cycles
    if st.button("Generate & Predict"):
        with st.spinner("Generating prediction..."):
            # Mock some history data
            history = []
            for i in range(1, cycles + 1):
                row = {"time_cycles": float(i)}
                for col in SENSOR_COLUMNS:
                    row[col] = 10.0 + (i * 0.1) # dummy degradation trend
                history.append(row)
                
            df = pd.DataFrame(history)
            df['unit_number'] = engine_id
            
            # 1. Prediction
            try:
                risk_info = predictor.predict(df)
                if "error" in risk_info:
                    st.error(risk_info["error"])
                else:
                    st.subheader("Prediction Results")
                    
                    # Display metrics
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Predicted RUL", f"{risk_info.get('rul', 0):.2f} Cycles")
                    m2.metric("Risk Level", risk_info.get("risk_level", "UNKNOWN"))
                    m3.metric("Health Status", risk_info.get("health_status", "UNKNOWN"))
                    
                    st.session_state['risk_info'] = risk_info
                    
                # 2. Explanation
                processed_df = add_rolling_features(df)
                current_features = processed_df.iloc[-1:][predictor.features]
                top_features = explain_prediction(predictor.model, current_features)
                
                st.session_state['top_features'] = top_features
                
                if top_features:
                    st.subheader("Feature Impact (SHAP)")
                    df_exp = pd.DataFrame(top_features)
                    if not df_exp.empty:
                        fig = px.bar(df_exp, x='impact', y='name', orientation='h', title='Top Degradation Drivers')
                        fig.update_layout(yaxis={'categoryorder':'total ascending'})
                        st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Prediction failed: {e}")

with col2:
    st.subheader("AI Maintenance Advisor")
    if 'risk_info' in st.session_state and 'top_features' in st.session_state:
        with st.spinner("Consulting Gemini AI for maintenance strategy..."):
            recommendation = advisor.get_recommendation(st.session_state['risk_info'], st.session_state['top_features'])
            if recommendation:
                st.markdown("### 🤖 Gemini Recommendation")
                st.info(recommendation)
            else:
                st.warning("No recommendation available.")
    else:
        st.write("Run a prediction first to get AI maintenance recommendations.")
