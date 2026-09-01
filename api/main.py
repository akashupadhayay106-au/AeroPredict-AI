import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

from src.inference import EnginePredictor
from src.explainability import explain_prediction
from llm.gemini_client import GeminiAdvisor
import pandas as pd

app = FastAPI(
    title="AeroPredict AI API",
    description="API for Aircraft Engine Health, RUL Prediction & Intelligent Maintenance System",
    version="1.0.0"
)

# Load singletons
predictor = EnginePredictor()
advisor = GeminiAdvisor()

class EngineDataRequest(BaseModel):
    engine_id: int
    history: List[Dict[str, float]] # List of rows representing history of this engine

class PredictResponse(BaseModel):
    engine_id: int
    current_cycle: int
    predicted_rul: float
    risk_level: str
    health_status: str
    
class ExplainResponse(BaseModel):
    top_features: List[Dict[str, Any]]
    
class RecommendationResponse(BaseModel):
    recommendation: str

@app.get("/")
def root():
    return {"message": "Welcome to AeroPredict AI API. Use /docs for documentation."}

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": predictor.model is not None}

@app.get("/model-info")
def model_info():
    info_path = 'reports/metadata/best_model_info.txt'
    if os.path.exists(info_path):
        with open(info_path, 'r') as f:
            return {"metadata": f.read()}
    return {"error": "Model metadata not found."}

@app.post("/predict", response_model=PredictResponse)
def predict(request: EngineDataRequest):
    if predictor.model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
        
    df = pd.DataFrame(request.history)
    if 'unit_number' not in df.columns:
        df['unit_number'] = request.engine_id
        
    risk_info = predictor.predict(df)
    if "error" in risk_info:
        raise HTTPException(status_code=400, detail=risk_info["error"])
        
    return PredictResponse(
        engine_id=risk_info['engine_id'],
        current_cycle=risk_info['current_cycle'],
        predicted_rul=risk_info['rul'],
        risk_level=risk_info['risk_level'],
        health_status=risk_info['health_status']
    )

@app.post("/explain", response_model=ExplainResponse)
def explain(request: EngineDataRequest):
    if predictor.model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
        
    df = pd.DataFrame(request.history)
    if 'unit_number' not in df.columns:
        df['unit_number'] = request.engine_id
        
    from src.feature_engineering import add_rolling_features
    processed_df = add_rolling_features(df)
    current_features = processed_df.iloc[-1:][predictor.features]
    
    top_features = explain_prediction(predictor.model, current_features)
    return ExplainResponse(top_features=top_features)

@app.post("/recommendation", response_model=RecommendationResponse)
def get_recommendation(request: EngineDataRequest):
    # First get prediction
    pred_response = predict(request)
    risk_info = pred_response.dict()
    
    # Then get explanation
    exp_response = explain(request)
    top_features = exp_response.top_features
    
    # Send to Gemini
    recommendation = advisor.get_recommendation(risk_info, top_features)
    return RecommendationResponse(recommendation=recommendation)
