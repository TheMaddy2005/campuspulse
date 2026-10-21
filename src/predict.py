"""Load the trained models and run predictions for one student record.

Used by the Streamlit app, but works standalone too:

    python -m src.predict
"""
from pathlib import Path

import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

# A realistic example student, used by the demo and the smoke test.
SAMPLE_STUDENT = {
    "school": "GP", "sex": "F", "age": 17, "address": "U", "famsize": "GT3",
    "Pstatus": "T", "Medu": 4, "Fedu": 4, "Mjob": "at_home", "Fjob": "teacher",
    "reason": "course", "guardian": "mother", "traveltime": 2, "studytime": 2,
    "failures": 0, "schoolsup": "yes", "famsup": "no", "paid": "no",
    "activities": "no", "nursery": "yes", "higher": "yes", "internet": "no",
    "romantic": "no", "famrel": 4, "freetime": 3, "goout": 4, "Dalc": 1,
    "Walc": 1, "health": 3, "absences": 6, "G1": 5, "G2": 6,
}


def load_models():
    regressor = joblib.load(MODELS_DIR / "grade_regressor.joblib")
    classifier = joblib.load(MODELS_DIR / "atrisk_classifier.joblib")
    return regressor, classifier


def predict_student(student: dict) -> dict:
    """Predict final grade and pass probability for one student."""
    regressor, classifier = load_models()
    X = pd.DataFrame([student])
    grade = float(regressor.predict(X)[0])
    pass_prob = float(classifier.predict_proba(X)[0][1])
    return {
        "predicted_grade": round(grade, 1),
        "pass_probability": round(pass_prob, 3),
        "fail_probability": round(1 - pass_prob, 3),
        "at_risk": pass_prob < 0.5,
    }


if __name__ == "__main__":
    print(predict_student(SAMPLE_STUDENT))
