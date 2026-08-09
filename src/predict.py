import pickle
import pandas as pd


MODEL_PATH = "models/xgb_model.pkl"
PREPROCESSOR_PATH = "models/preprocessor.pkl"


def load_artifacts():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    with open(PREPROCESSOR_PATH, "rb") as f:
        preprocessor = pickle.load(f)

    return model, preprocessor


def predict_failure(input_data):
    model, preprocessor = load_artifacts()

    processed_data = preprocessor.transform(input_data)

    prediction = model.predict(processed_data)[0]

    probability = model.predict_proba(
        processed_data
    )[0, 1]

    return prediction, probability