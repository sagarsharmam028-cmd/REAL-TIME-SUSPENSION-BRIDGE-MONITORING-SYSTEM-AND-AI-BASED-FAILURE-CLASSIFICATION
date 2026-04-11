from sklearn.ensemble import RandomForestClassifier
import numpy as np

def train_model():
    # Just create model (no dummy training)
    return RandomForestClassifier()


def predict(model, features):
    X = np.array([[ 
        features["mean_deflection"],
        features["max_deflection"],
        features["vibration_std"],
        features["vibration_max"],
        features["deflection_rate"]
    ]])

    prediction = model.predict(X)[0]
    confidence = model.predict_proba(X).max()

    return prediction, confidence