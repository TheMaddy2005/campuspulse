"""Rule-based personalized learning recommendations.

The ML models tell us WHAT is likely to happen (predicted grade, at-risk
probability). This module turns that prediction into WHAT TO DO NEXT by
looking at the student's profile: study time, past failures, absences,
support systems and lifestyle. Each rule adds one concrete recommendation
with a category so the app can group them.

Rules are deliberately transparent - a teacher or student can read this file
and understand exactly why each recommendation appeared.
"""
from dataclasses import dataclass, field

# Curated free learning resources, mapped to the gaps we detect.
RESOURCES = {
    "math_fundamentals": [
        ("Khan Academy - Math", "https://www.khanacademy.org/math"),
        ("NPTEL - Mathematics courses", "https://nptel.ac.in/courses"),
    ],
    "study_skills": [
        ("Coursera - Learning How to Learn",
         "https://www.coursera.org/learn/learning-how-to-learn"),
        ("Spaced repetition with Anki", "https://apps.ankiweb.net/"),
    ],
    "exam_practice": [
        ("Khan Academy - Practice & quizzes", "https://www.khanacademy.org/"),
    ],
}


@dataclass
class Recommendation:
    category: str
    title: str
    priority: str  # "high", "medium" or "low"
    message: str
    resources: list = field(default_factory=list)


def _studytime_label(value: int) -> str:
    return {1: "<2 hrs/week", 2: "2-5 hrs/week", 3: "5-10 hrs/week",
            4: ">10 hrs/week"}.get(value, str(value))


def generate_recommendations(student: dict, predicted_grade: float,
                             risk_probability: float) -> list:
    """Return a priority-ordered list of Recommendation objects.

    `student` holds the raw feature values (same columns as the dataset),
    `predicted_grade` comes from the grade regressor and `risk_probability`
    is the classifier's P(fail).
    """
    recs = []

    # --- Academic history -------------------------------------------------
    failures = int(student.get("failures", 0))
    if failures >= 2:
        recs.append(Recommendation(
            "Academic recovery", "Multiple past failures", "high",
            f"You have failed {failures} past classes. Meet your teacher to "
            "identify the exact topics you are missing, and re-learn those "
            "before new material. Retaking fundamentals beats re-reading notes.",
            RESOURCES["math_fundamentals"],
        ))
    elif failures == 1:
        recs.append(Recommendation(
            "Academic recovery", "One past failure", "medium",
            "One past failure usually means a specific weak topic, not a "
            "general problem. Find that topic and drill it with practice "
            "questions before the next exam.",
        ))

    # --- Study habits ------------------------------------------------------
    studytime = int(student.get("studytime", 2))
    if studytime <= 1:
        recs.append(Recommendation(
            "Study habits", "Very low study time", "high",
            f"Current study time: {_studytime_label(studytime)}. Raising it "
            "to a steady 2-5 hrs/week (roughly 30-45 min a day) is the "
            "single biggest change most students can make. Fix a daily slot "
            "rather than relying on motivation.",
            RESOURCES["study_skills"],
        ))
    elif studytime == 2:
        recs.append(Recommendation(
            "Study habits", "Moderate study time", "low",
            "2-5 hrs/week is a decent base. Try adding one focused 45-minute "
            "session on your weakest topic each week.",
        ))

    # --- Attendance --------------------------------------------------------
    absences = int(student.get("absences", 0))
    if absences >= 10:
        recs.append(Recommendation(
            "Attendance", "Many absences", "high",
            f"{absences} absences is a lot - missed classes compound quickly. "
            "Ask classmates for notes from missed days and tell your teacher "
            "you are catching up; most will help if you show up and ask.",
        ))
    elif absences >= 5:
        recs.append(Recommendation(
            "Attendance", "Some absences", "medium",
            f"{absences} absences so far. Try to keep it under 5 for the rest "
            "of the term; attendance is one of the strongest quiet signals "
            "in the data.",
        ))

    # --- Support systems ---------------------------------------------------
    if student.get("schoolsup") == "no" and student.get("famsup") == "no":
        recs.append(Recommendation(
            "Support", "Little support around you", "medium",
            "Neither school nor family support is marked. Look for one "
            "accountability partner - a classmate you study with weekly - "
            "or ask the school about tutoring options.",
        ))
    if student.get("paid") == "no" and risk_probability > 0.5:
        recs.append(Recommendation(
            "Support", "Consider extra help", "medium",
            "Given the at-risk prediction, a few weeks of paid tutoring or a "
            "free peer-study group could pay off more than solo study.",
        ))

    # --- Lifestyle ---------------------------------------------------------
    if int(student.get("goout", 3)) >= 4 and studytime <= 2:
        recs.append(Recommendation(
            "Balance", "Social time vs study time", "medium",
            "Going out is high while study time is low. You do not need to "
            "cut your social life - just protect 2-3 fixed study slots "
            "first and plan social time around them.",
        ))
    if int(student.get("Walc", 1)) >= 4:
        recs.append(Recommendation(
            "Wellbeing", "Weekend alcohol consumption", "medium",
            "High weekend alcohol consumption correlates with lower grades "
            "in this dataset. Worth an honest look during exam periods.",
        ))

    # --- Exam technique ----------------------------------------------------
    g1, g2 = student.get("G1"), student.get("G2")
    if g1 is not None and g2 is not None and int(g2) < int(g1):
        recs.append(Recommendation(
            "Momentum", "Grades trending down", "high",
            f"Your grade moved from {g1} to {g2} between periods. That "
            "downturn is the strongest early-warning signal in the model. "
            "Act now: past papers under timed conditions, then review every "
            "mistake the same day.",
            RESOURCES["exam_practice"],
        ))

    # --- Positive case ------------------------------------------------------
    if not recs:
        recs.append(Recommendation(
            "On track", "Keep it up", "low",
            "No red flags in the profile. Keep the routine that is working, "
            "and consider helping a classmate - teaching is the best review.",
        ))

    priority_order = {"high": 0, "medium": 1, "low": 2}
    recs.sort(key=lambda r: priority_order.get(r.priority, 3))
    return recs
