import pandas as pd
import numpy as np
from src.rul import calculate_rul
from src.feature_engineering import add_rolling_features
from src.config import SENSOR_COLUMNS, SETTING_COLUMNS, RISK_THRESHOLDS


# ── RUL Tests ──

def test_calculate_rul():
    df = pd.DataFrame({
        'unit_number': [1, 1, 1, 1],
        'time_cycles': [1, 2, 3, 4]
    })
    result_df = calculate_rul(df)
    expected_rul = [3, 2, 1, 0]
    np.testing.assert_array_equal(result_df['RUL'].values, expected_rul)


def test_calculate_rul_with_cap():
    df = pd.DataFrame({
        'unit_number': [1, 1, 1, 1],
        'time_cycles': [1, 2, 3, 4]
    })
    result_df = calculate_rul(df, cap=2)
    expected_rul = [2, 2, 1, 0]
    np.testing.assert_array_equal(result_df['RUL'].values, expected_rul)


# ── Feature Engineering Tests ──

def test_feature_engineering_no_leakage():
    df = pd.DataFrame({
        'unit_number': [1, 1, 2, 2],
        'time_cycles': [1, 2, 1, 2],
        'sensor_1': [10, 20, 100, 200]
    })
    import src.config
    original = src.config.SENSOR_COLUMNS
    src.config.SENSOR_COLUMNS = ['sensor_1']
    result = add_rolling_features(df, window_sizes=[2])
    src.config.SENSOR_COLUMNS = original  # restore

    assert result.loc[1, 'sensor_1_roll_mean_2'] == 15.0
    assert result.loc[2, 'sensor_1_roll_mean_2'] == 100.0


# ── Inference Schema Validation Tests ──

def test_inference_missing_base_columns():
    from src.inference import EnginePredictor
    predictor = EnginePredictor()
    if predictor.model is not None:
        df_invalid = pd.DataFrame({'sensor_2': [10]})
        res = predictor.predict(df_invalid)
        assert "error" in res
        assert "Missing required base columns" in res["error"]


def test_inference_missing_model_features():
    from src.inference import EnginePredictor
    predictor = EnginePredictor()
    if predictor.model is not None:
        df_invalid = pd.DataFrame({'unit_number': [1], 'time_cycles': [1], 'sensor_2': [10]})
        res = predictor.predict(df_invalid)
        assert "error" in res
        assert "Schema mismatch" in res["error"]
        assert "Missing expected features" in res["details"]


def test_inference_valid_prediction():
    from src.inference import EnginePredictor
    predictor = EnginePredictor()
    if predictor.model is not None:
        rows = []
        for i in range(1, 11):
            row = {"time_cycles": float(i), "unit_number": 1}
            for col in SETTING_COLUMNS:
                row[col] = 0.0
            for col in SENSOR_COLUMNS:
                row[col] = 20.0 + np.random.normal(0, 0.1)
            rows.append(row)
        df = pd.DataFrame(rows)
        res = predictor.predict(df)
        assert "error" not in res
        assert "rul" in res
        assert isinstance(res["rul"], float)
        assert not np.isnan(res["rul"])
        assert not np.isinf(res["rul"])


# ── Risk Engine / Health Tests ──

def test_health_thresholds():
    from src.risk_engine import assess_risk
    healthy = assess_risk(60.0)
    assert healthy["health_status"] == "HEALTHY"
    assert healthy["risk_level"] == "LOW"

    monitoring = assess_risk(40.0)
    assert monitoring["risk_level"] == "MEDIUM"

    attention = assess_risk(15.0)
    assert attention["risk_level"] == "HIGH"

    critical = assess_risk(5.0)
    assert critical["health_status"] == "IMMEDIATE_MAINTENANCE_REQUIRED"
    assert critical["risk_level"] == "CRITICAL"


# ── Gemini Graceful Degradation Tests ──

def test_gemini_missing_key():
    from unittest.mock import patch
    with patch('llm.gemini_client.GEMINI_API_KEY', ''):
        from llm.gemini_client import GeminiAdvisor
        advisor = GeminiAdvisor()
        # With no key and no st.secrets, client should be None
        if advisor.client is None:
            result = advisor.get_recommendation({"rul": 42}, [])
            assert result == ""


def test_gemini_chat_no_client():
    from unittest.mock import patch
    with patch('llm.gemini_client.GEMINI_API_KEY', ''):
        from llm.gemini_client import GeminiAdvisor
        advisor = GeminiAdvisor()
        if advisor.client is None:
            result = advisor.chat_with_context("Hello", [], {"rul": 42}, [])
            assert "unavailable" in result.lower()
