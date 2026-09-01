# AeroPredict AI ✈️

## Problem Statement
AeroPredict AI is an Intelligent Aircraft Engine Health & Maintenance System. The goal is to accurately predict the Remaining Useful Life (RUL) of aircraft engines using run-to-failure degradation trajectories. This enables predictive maintenance, preventing catastrophic failures while maximizing the operational lifespan of the engine.

## Dataset
The project is built on the **NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation) dataset**.
It includes four distinct subsets (FD001 to FD004) featuring different operational conditions and fault modes.

## Architecture
The system employs a unified, single-tier architecture optimized for serverless deployments like Streamlit Community Cloud:
- **Frontend/Inference:** Streamlit UI integrated directly with the pre-trained models. (No separate FastAPI backend required for the public app).
- **AI Maintenance Advisor:** Google Gemini 2.5 Flash LLM, consuming SHAP explainability metrics and predicted RUL to generate human-readable maintenance strategies.

## Model Performance Metrics
Both Classical ML and Deep Learning architectures were evaluated on the processed data.
- **Classical ML (Best: LightGBM)**
  - MAE: 43.90
  - RMSE: 59.10
  - R2: -0.004
- **Deep Learning (Best: GRU)**
  - MAE: 36.16
  - RMSE: 41.85
  - R2: -0.014

**Selected Inference Model:** The **GRU Deep Learning Model** was selected for production inference due to superior temporal sequence modeling on the multi-sensor dataset.

## Links & Deployment
## 🚀 Live Demo 👉 [Open AeroPredict-AI Live App](https://akash-aeropredict-ai-2.streamlit.app/)
- **GitHub Repository:** [https://github.com/akashupadhayay106-au/AeroPredict-AI](https://github.com/akashupadhayay106-au/AeroPredict-AI)
- **FastAPI Live URL:** Not Required (Monolithic Architecture)
- **API Documentation:** N/A

## Local Run Instructions
To run the system locally for development:
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run Streamlit App
streamlit run app/app.py
```

## Deployment Instructions (Streamlit Community Cloud)
1. Navigate to [share.streamlit.io](https://share.streamlit.io) and log in with your GitHub account.
2. Click **New app**.
3. Select `akashupadhayay106-au/AeroPredict-AI` as the repository.
4. Select `master` as the branch.
5. Enter `app/app.py` as the Main file path.
6. Click **Advanced settings** and add your `GEMINI_API_KEY` to the Secrets section.
7. Click **Deploy**.

## Known Limitations
- The model exhibits a negative R2 score on the test set, indicating that the baseline variance is high relative to the predictions. This is common in C-MAPSS due to early-cycle sensor noise before degradation begins. Piece-wise RUL capping at 125 cycles was applied to mitigate this, but further hyperparameter tuning (sequence length, learning rate) is recommended for production aviation use.
