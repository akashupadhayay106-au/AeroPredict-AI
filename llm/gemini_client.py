import os
from google import genai
from src.config import GEMINI_API_KEY
from llm.prompts import MAINTENANCE_ADVISOR_PROMPT

class GeminiAdvisor:
    def __init__(self):
        self.api_key = GEMINI_API_KEY
        
        # Fallback to Streamlit Secrets if running in Streamlit Cloud
        if not self.api_key:
            try:
                import streamlit as st
                if "GEMINI_API_KEY" in st.secrets:
                    self.api_key = st.secrets["GEMINI_API_KEY"]
            except ImportError:
                pass
                
        self.client = None
        if self.api_key:
            self.client = genai.Client(api_key=self.api_key)
            
    def get_recommendation(self, risk_info: dict, top_features: list) -> str:
        if not self.client:
            return "AI Maintenance Advisor unavailable. (API Key missing or invalid)\n\nModel prediction and SHAP analysis are still available."
            
        try:
            # Format top features nicely
            features_str = "\n".join([f"- {f['name']}: {f['impact']}" for f in top_features])
            
            prompt = MAINTENANCE_ADVISOR_PROMPT.format(
                engine_id=risk_info.get('engine_id', 'Unknown'),
                current_cycle=risk_info.get('current_cycle', 'Unknown'),
                predicted_rul=risk_info.get('rul', 'Unknown'),
                risk_level=risk_info.get('risk_level', 'Unknown'),
                health_status=risk_info.get('health_status', 'Unknown'),
                top_features=features_str
            )
            
            response = self.client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt,
            )
            
            return response.text
        except Exception as e:
            return f"AI Maintenance Advisor unavailable due to error: {str(e)}\n\nModel prediction and SHAP analysis are still available."
