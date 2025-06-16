import joblib
import numpy as np


def predict_certification(member_data):
    model = joblib.load('ai_models/trained_models/certification_predictor.pkl')
    scaler = joblib.load('ai_models/trained_models/certification_scaler.pkl')

    # Preprocess and predict
    scaled_data = scaler.transform([member_data])
    prediction = model.predict_proba(scaled_data)[0][1]

    return {
        'approved': prediction > 0.5,
        'confidence': prediction,
        'recommendation': 'Eligible' if prediction > 0.5 else 'Needs improvement'
    }