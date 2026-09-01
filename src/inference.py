import os
import joblib
import pandas as pd
from src.config import MODEL_DIR
from src.feature_engineering import add_rolling_features
from src.risk_engine import assess_risk

class EnginePredictor:
    def __init__(self):
        self.model = None
        self.features = None
        self._load_model()
        
    def _load_model(self):
        model_path = os.path.join(MODEL_DIR, "best_ml_model.pkl")
        features_path = os.path.join(MODEL_DIR, "feature_config.pkl")
        
        if os.path.exists(model_path) and os.path.exists(features_path):
            self.model = joblib.load(model_path)
            self.features = joblib.load(features_path)
        else:
            print("Warning: Model or features not found. Train models first.")
            
    def predict(self, df: pd.DataFrame) -> dict:
        """
        Predict RUL for a single engine's history.
        The dataframe should contain the full history to calculate rolling features.
        """
        if self.model is None:
            return {"error": "Model not loaded."}
            
        # Feature Engineering on the fly
        # (Assuming df has unit_number and sensors)
        processed_df = add_rolling_features(df)
        
        # Take the last row (current cycle)
        current_features = processed_df.iloc[-1:][self.features]
        current_cycle = int(df.iloc[-1]['time_cycles'])
        engine_id = int(df.iloc[-1]['unit_number'])
        
        # Predict RUL
        predicted_rul = self.model.predict(current_features)[0]
        
        # Risk Assessment
        risk_info = assess_risk(predicted_rul)
        risk_info['current_cycle'] = current_cycle
        risk_info['engine_id'] = engine_id
        
        return risk_info
