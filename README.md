# ✈️ AeroPredict AI

**Intelligent Aircraft Engine Health & Predictive Maintenance System**

## 🚀 Live Demo

👉 **[Open AeroPredict AI Live App](https://akash-aeropredict-ai-2.streamlit.app/)**

---

## 🎯 What Does It Predict?

AeroPredict AI estimates the **Remaining Useful Life (RUL)** of aircraft engines — the number of operating cycles an engine may have remaining before it requires maintenance attention.

> RUL is a predictive estimate, not a certified aviation maintenance decision.

**Input:** Engine operating conditions + sensor measurements + engine history / scenario data.

**Output:**
- Predicted Remaining Useful Life (RUL) in cycles
- Engine Health Status (Healthy / Monitor / Warning / Critical)
- Risk Level (Low / Medium / High / Critical)
- SHAP-based explainability (which factors influenced the prediction)
- AI-powered maintenance recommendations (via Google Gemini)

---

## 📊 Dataset

Built on the **NASA C-MAPSS** (Commercial Modular Aero-Propulsion System Simulation) turbofan engine degradation dataset:
- Run-to-failure trajectories for ~100 engines
- 3 operational settings + 21 sensor measurements per cycle
- Subset: FD001 (single operating condition, single fault mode)

---

## 🧠 Model

| Component | Detail |
|---|---|
| Production Inference Model | LightGBM (best ML performer) |
| Compared Models | Linear Regression, Random Forest, XGBoost, LightGBM, GRU |
| Feature Engineering | Rolling mean/std (windows 5, 10), constant column removal |
| Total Features | 101 (3 settings + 14 sensors + 84 rolling features) |
| Target | RUL capped at 125 cycles |

### Actual Metrics

| Metric | LightGBM (ML) | GRU (DL) |
|---|---|---|
| MAE | 43.90 | 36.17 |
| RMSE | 59.10 | 41.85 |
| R² | −0.004 | −0.014 |

> The negative R² indicates the model has not outperformed a naïve mean predictor on the test set. This is common in C-MAPSS due to early-cycle sensor noise before degradation manifests. Further hyperparameter tuning is recommended for production aviation use.

---

## 🔍 Explainability

**SHAP** (SHapley Additive exPlanations) provides per-prediction feature importance, showing which sensors and operating conditions contributed most to each individual RUL estimate.

---

## 🤖 AI Maintenance Advisor

A **Google Gemini**-powered conversational assistant provides:
- Engineering-oriented maintenance recommendations
- Interactive Q&A about the current engine status
- Context-aware guidance based on the actual prediction and SHAP factors

The AI Advisor is **optional** — if the API key is missing or the service is unavailable, all core ML predictions continue to work normally.

---

## 🏗️ Architecture

```
NASA C-MAPSS Data
      ↓
Data Preparation & Feature Engineering
      ↓
ML Model (LightGBM)
      ↓
RUL Prediction
      ↓
┌────────────────┬────────────────────┐
│ SHAP Explain.  │ Gemini AI Advisor  │
└────────────────┴────────────────────┘
      ↓
Streamlit Application (Public)
```

Single-tier monolithic architecture optimised for Streamlit Community Cloud deployment.

---

## 🏃 Local Setup

```bash
# 1. Clone the repository
git clone https://github.com/akashupadhayay106-au/AeroPredict-AI.git
cd AeroPredict-AI

# 2. Install dependencies
pip install -r requirements.txt

# 3. Create .env file
echo "GEMINI_API_KEY=your_key_here" > .env

# 4. Run the Streamlit app
streamlit run app/app.py
```

---

## ☁️ Streamlit Cloud Deployment

1. Navigate to [share.streamlit.io](https://share.streamlit.io) and log in with GitHub.
2. Click **New app**.
3. Select `akashupadhayay106-au/AeroPredict-AI`, branch `master`, main file `app/app.py`.
4. Click **Advanced settings** → add `GEMINI_API_KEY = "your_key"` in Secrets.
5. Click **Deploy**.

---

## 🧪 Testing

```bash
pytest -q
```

Tests cover: RUL calculation, feature engineering, schema validation, inference pipeline, health thresholds, and Gemini graceful degradation.

---

## ⚠️ Limitations

- The model uses simulated (C-MAPSS) data and should not be directly applied to real-world maintenance without retraining.
- Predictions are decision-support estimates, not certified aviation guidance.
- Health status categories are rule-based heuristics, not model-generated classifications.
- The Gemini AI Advisor depends on an external API and may experience temporary outages.
- Simulation mode generates synthetic data that approximates but does not replicate real sensor behaviour.

---

## 📜 License

This project is for educational and demonstration purposes.
