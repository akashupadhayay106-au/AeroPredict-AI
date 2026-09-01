import os
import time
from google import genai
from src.config import GEMINI_API_KEY
from llm.prompts import MAINTENANCE_ADVISOR_PROMPT

MAX_RETRIES = 2
RETRY_DELAY_SECONDS = 2

class GeminiAdvisor:
    def __init__(self):
        self.api_key = self._resolve_api_key()
        self.client = None
        if self.api_key:
            try:
                self.client = genai.Client(api_key=self.api_key)
            except Exception:
                self.client = None

    @staticmethod
    def _resolve_api_key() -> str:
        """Resolve API key: env var first, then Streamlit secrets."""
        key = GEMINI_API_KEY
        if not key:
            try:
                import streamlit as st
                if "GEMINI_API_KEY" in st.secrets:
                    key = st.secrets["GEMINI_API_KEY"]
            except (ImportError, Exception):
                pass
        return key or ""

    def _call_gemini(self, prompt: str) -> str:
        """Call Gemini with limited retry and exponential backoff."""
        last_error = None
        for attempt in range(MAX_RETRIES + 1):
            try:
                response = self.client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=prompt,
                )
                return response.text
            except Exception as e:
                last_error = e
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY_SECONDS * (attempt + 1))
        raise last_error

    def get_recommendation(self, risk_info: dict, top_features: list) -> str:
        if not self.client:
            return ""

        features_str = "\n".join([f"- {f['name']}: {f['impact']}" for f in top_features]) if top_features else "Not available"

        prompt = MAINTENANCE_ADVISOR_PROMPT.format(
            engine_id=risk_info.get('engine_id', 'Unknown'),
            current_cycle=risk_info.get('current_cycle', 'Unknown'),
            predicted_rul=risk_info.get('rul', 'Unknown'),
            risk_level=risk_info.get('risk_level', 'Unknown'),
            health_status=risk_info.get('health_status', 'Unknown'),
            top_features=features_str
        )

        return self._call_gemini(prompt)

    def chat_with_context(self, user_message: str, chat_history: list, risk_info: dict, top_features: list) -> str:
        if not self.client:
            return "AI Chatbot unavailable. (API Key not configured)"

        features_str = "\n".join([f"- {f['name']}: {f['impact']}" for f in top_features]) if top_features else "Not available"

        context_prompt = f"""You are the AeroPredict AI Maintenance Assistant.
You are helping an engineer diagnose an aircraft engine with the following current status:
Engine ID: {risk_info.get('engine_id', 'Unknown')}
Current Cycle: {risk_info.get('current_cycle', 'Unknown')}
Predicted Remaining Useful Life (RUL): {risk_info.get('rul', 'Unknown')} cycles
Risk Level: {risk_info.get('risk_level', 'Unknown')}
Health Status: {risk_info.get('health_status', 'Unknown')}

Top Factors influencing this prediction (SHAP values):
{features_str}

Important rules:
- Clearly distinguish between model predictions and your own recommendations.
- Do not claim certainty about engine failure.
- Do not prescribe certified aviation procedures.
- State that recommendations are informational and should be validated.

Please answer the user's question directly, clearly, and concisely.
"""
        full_prompt = context_prompt + "\n\nConversation History:\n"
        for msg in chat_history[-10:]:  # Limit history to last 10 messages
            role = "User" if msg["role"] == "user" else "Assistant"
            full_prompt += f"{role}: {msg['content']}\n"
        full_prompt += f"\nUser: {user_message}\nAssistant:"

        return self._call_gemini(full_prompt)
