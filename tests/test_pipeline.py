"""Smoke tests: the pipeline loads, trains tiny, predicts and recommends.

Run with:  python -m pytest tests/ -q
(or plain `python tests/test_pipeline.py`)
"""
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

from data_preprocessing import build_preprocessor, load_data, split_features_target
from predict import SAMPLE_STUDENT, predict_student
from recommender import generate_recommendations


def test_dataset_loads():
    df = load_data()
    assert len(df) == 395
    assert "G3" in df.columns


def test_preprocessor_transforms():
    df = load_data()
    X, _ = split_features_target(df)
    transformed = build_preprocessor(X).fit_transform(X.head(10))
    assert transformed.shape[0] == 10


def test_models_predict():
    result = predict_student(SAMPLE_STUDENT)
    assert 0 <= result["predicted_grade"] <= 20
    assert 0 <= result["pass_probability"] <= 1


def test_recommender_flags_weak_student():
    weak = dict(SAMPLE_STUDENT, studytime=1, failures=2, absences=15, G1=6, G2=4)
    recs = generate_recommendations(weak, predicted_grade=6.0, risk_probability=0.9)
    assert recs, "expected at least one recommendation"
    assert recs[0].priority == "high"


def test_recommender_positive_case():
    strong = dict(SAMPLE_STUDENT, studytime=4, failures=0, absences=0,
                  schoolsup="yes", famsup="yes", G1=15, G2=16, goout=2)
    recs = generate_recommendations(strong, predicted_grade=16.0, risk_probability=0.05)
    assert any(r.category == "On track" for r in recs)


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
    print("All tests passed.")
