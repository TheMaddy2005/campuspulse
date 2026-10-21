"""CampusPulse - Streamlit demo app.

Run from the repo root:

    streamlit run app/streamlit_app.py

Tabs:
  1. Predict & Recommend - enter a student profile, get the predicted final
     grade, the at-risk probability and personalized recommendations.
  2. Model Insights - how the models perform and what drives them.
  3. About - dataset and project notes.
"""
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

from predict import predict_student  # noqa: E402
from recommender import generate_recommendations  # noqa: E402

st.set_page_config(page_title="CampusPulse", page_icon="🎓", layout="wide")

st.title("🎓 CampusPulse")
st.caption("AI-powered student performance prediction & personalized learning")

tab_predict, tab_insights, tab_about = st.tabs(
    ["Predict & Recommend", "Model Insights", "About"]
)

# --------------------------------------------------------------------------
# Tab 1: prediction
# --------------------------------------------------------------------------
with tab_predict:
    st.subheader("Student profile")
    st.write("Fill in the details below. Defaults describe an average student.")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Background**")
        sex = st.selectbox("Sex", ["F", "M"])
        age = st.slider("Age", 15, 22, 17)
        address = st.selectbox("Home address", ["U", "R"],
                               format_func=lambda x: "Urban" if x == "U" else "Rural")
        famsize = st.selectbox("Family size", ["LE3", "GT3"],
                               format_func=lambda x: "3 or less" if x == "LE3" else "More than 3")
        Medu = st.select_slider("Mother's education (0 none - 4 higher)", [0, 1, 2, 3, 4], 2)
        Fedu = st.select_slider("Father's education (0 none - 4 higher)", [0, 1, 2, 3, 4], 2)
        internet = st.selectbox("Internet at home", ["yes", "no"])
        higher = st.selectbox("Wants higher education", ["yes", "no"])

    with col2:
        st.markdown("**Study habits**")
        studytime = st.select_slider(
            "Weekly study time", [1, 2, 3, 4],
            format_func=lambda x: {1: "<2 hrs", 2: "2-5 hrs", 3: "5-10 hrs", 4: ">10 hrs"}[x],
            value=2,
        )
        traveltime = st.select_slider(
            "Travel time to school", [1, 2, 3, 4],
            format_func=lambda x: {1: "<15 min", 2: "15-30 min", 3: "30-60 min", 4: ">1 hr"}[x],
        )
        failures = st.selectbox("Past class failures", [0, 1, 2, 3])
        absences = st.slider("Absences", 0, 32, 4)
        schoolsup = st.selectbox("Extra school support", ["no", "yes"])
        famsup = st.selectbox("Family support", ["yes", "no"])
        paid = st.selectbox("Extra paid classes", ["no", "yes"])

    with col3:
        st.markdown("**Lifestyle & current grades**")
        goout = st.select_slider("Going out with friends (1 low - 5 high)", [1, 2, 3, 4, 5], 3)
        freetime = st.select_slider("Free time (1 low - 5 high)", [1, 2, 3, 4, 5], 3)
        Dalc = st.select_slider("Workday alcohol (1 low - 5 high)", [1, 2, 3, 4, 5], 1)
        Walc = st.select_slider("Weekend alcohol (1 low - 5 high)", [1, 2, 3, 4, 5], 1)
        health = st.select_slider("Health (1 bad - 5 good)", [1, 2, 3, 4, 5], 4)
        famrel = st.select_slider("Family relationship (1 bad - 5 great)", [1, 2, 3, 4, 5], 4)
        G1 = st.slider("Period 1 grade (0-20)", 0, 20, 10)
        G2 = st.slider("Period 2 grade (0-20)", 0, 20, 10)

    if st.button("Predict performance", type="primary"):
        student = {
            "school": "GP", "sex": sex, "age": age, "address": address,
            "famsize": famsize, "Pstatus": "T", "Medu": Medu, "Fedu": Fedu,
            "Mjob": "other", "Fjob": "other", "reason": "course",
            "guardian": "mother", "traveltime": traveltime, "studytime": studytime,
            "failures": failures, "schoolsup": schoolsup, "famsup": famsup,
            "paid": paid, "activities": "no", "nursery": "yes", "higher": higher,
            "internet": internet, "romantic": "no", "famrel": famrel,
            "freetime": freetime, "goout": goout, "Dalc": Dalc, "Walc": Walc,
            "health": health, "absences": absences, "G1": G1, "G2": G2,
        }
        result = predict_student(student)

        st.divider()
        m1, m2, m3 = st.columns(3)
        m1.metric("Predicted final grade", f"{result['predicted_grade']} / 20")
        m2.metric("Pass probability", f"{result['pass_probability'] * 100:.0f}%")
        if result["at_risk"]:
            m3.error("⚠️ At risk of failing")
        else:
            m3.success("✅ On track to pass")

        st.subheader("Personalized learning plan")
        recs = generate_recommendations(
            student, result["predicted_grade"], result["fail_probability"]
        )
        for rec in recs:
            badge = {"high": "🔴", "medium": "🟡", "low": "🟢"}[rec.priority]
            with st.expander(f"{badge} {rec.title} ({rec.category})", expanded=rec.priority == "high"):
                st.write(rec.message)
                for name, url in rec.resources:
                    st.markdown(f"- [{name}]({url})")

# --------------------------------------------------------------------------
# Tab 2: model insights
# --------------------------------------------------------------------------
with tab_insights:
    st.subheader("How the models perform")
    metrics_path = BASE_DIR / "reports" / "metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())
        c1, c2, c3 = st.columns(3)
        reg = metrics["grade_regressor"]
        c1.metric("Grade regressor R²", reg["r2"])
        c2.metric("Mean absolute error", f"{reg['mae']} grades")
        c3.metric("Classifier F1", metrics["atrisk_classifier"]["f1"])
        st.caption(
            "Trained on 80% of 395 students, evaluated on a held-out 20%. "
            "The 'no midterms' variant (see reports/metrics.json) shows early-warning "
            "accuracy before any exam grades exist."
        )

    figures = BASE_DIR / "reports" / "figures"
    cols = st.columns(2)
    for i, fig in enumerate(["feature_importance.png", "actual_vs_predicted.png",
                             "confusion_matrix.png", "grade_distribution.png"]):
        path = figures / fig
        if path.exists():
            cols[i % 2].image(str(path), use_container_width=True)

# --------------------------------------------------------------------------
# Tab 3: about
# --------------------------------------------------------------------------
with tab_about:
    st.markdown(
        """
### What is CampusPulse?
CampusPulse predicts a student's final grade (0-20) and their risk of failing,
then turns that prediction into a concrete, personalized learning plan.

### How it works
1. **Data** - UCI Student Performance dataset: 395 students, 33 features
   covering demographics, study habits, lifestyle and period grades.
2. **Models** - a Gradient Boosting regressor predicts the final grade; a
   Gradient Boosting classifier flags at-risk students (predicted fail).
3. **Recommendations** - transparent rules map the profile + prediction to
   specific actions and free learning resources.

### Dataset
[UCI ML Repository - Student Performance](https://archive.ics.uci.edu/dataset/320/student+performance)
(CC BY 4.0). Cortez & Silva, 2008.

*Educational project - predictions are statistical estimates, not judgments.*
        """
    )
