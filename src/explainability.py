import shap
import pandas as pd
import numpy as np

def explain_prediction(model, features_df: pd.DataFrame):
    """
    Generate SHAP values for a specific prediction (local explanation).
    We use TreeExplainer for Tree models like XGBoost/LightGBM/RF.
    """
    try:
        # Assuming the model is tree-based for this prototype.
        # If it's a linear model, we'd use LinearExplainer.
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(features_df)
        
        # Get feature impacts for the specific row
        impacts = shap_values[0] if isinstance(shap_values, list) else shap_values
        if len(impacts.shape) > 1:
            impacts = impacts[0] # Single prediction
            
        feature_importance = pd.DataFrame({
            'name': features_df.columns,
            'impact': impacts
        })
        
        # Sort by absolute impact
        feature_importance['abs_impact'] = feature_importance['impact'].abs()
        feature_importance = feature_importance.sort_values(by='abs_impact', ascending=False)
        
        # Return top 5 drivers
        top_features = feature_importance.head(5).to_dict('records')
        
        # Clean up output
        cleaned_top = []
        for f in top_features:
            cleaned_top.append({
                "name": f["name"],
                "impact": round(float(f["impact"]), 4)
            })
            
        return cleaned_top
    except Exception as e:
        print(f"Error generating SHAP explanation: {e}")
        return []
