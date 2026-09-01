from src.config import RISK_THRESHOLDS

def assess_risk(rul: float) -> dict:
    """
    Assess engine risk based on RUL prediction.
    """
    if rul > RISK_THRESHOLDS['LOW']:
        risk_level = "LOW"
        health_status = "HEALTHY"
    elif rul > RISK_THRESHOLDS['MEDIUM']:
        risk_level = "MEDIUM"
        health_status = "MONITORING_SUGGESTED"
    elif rul > RISK_THRESHOLDS['HIGH']:
        risk_level = "HIGH"
        health_status = "ATTENTION_REQUIRED"
    else:
        risk_level = "CRITICAL"
        health_status = "IMMEDIATE_MAINTENANCE_REQUIRED"
        
    return {
        "rul": round(float(rul), 2),
        "risk_level": risk_level,
        "health_status": health_status
    }
