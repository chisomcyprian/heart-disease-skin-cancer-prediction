from src.heart.predict import predict_heart

HIGH_RISK = {"age": 67, "sex": 1, "cp": 4, "trestbps": 160, "chol": 286,
             "fbs": 0, "restecg": 2, "thalach": 108, "exang": 1,
             "oldpeak": 1.5, "slope": 2, "ca": 3, "thal": 7}

LOW_RISK = {"age": 41, "sex": 0, "cp": 2, "trestbps": 130, "chol": 204,
            "fbs": 0, "restecg": 2, "thalach": 172, "exang": 0,
            "oldpeak": 1.4, "slope": 1, "ca": 0, "thal": 3}


def test_output_format():
    result = predict_heart(HIGH_RISK)
    assert 0 <= result["probability"] <= 1
    assert result["prediction"] in (0, 1)


def test_high_risk_scores_higher():
    high = predict_heart(HIGH_RISK)["probability"]
    low = predict_heart(LOW_RISK)["probability"]
    assert high > low