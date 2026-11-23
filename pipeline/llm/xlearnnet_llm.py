
import ollama

MODEL = "llama3.2:3b"

SYSTEM_PROMPT = """
You are the X-LearnNet academic risk explanation assistant.

X-LearnNet uses Logistic Regression, GraphSAGE, SHAP,
LIME and GNNExplainer to predict academic risk.

STRICT RULES:

1. Use only the evidence explicitly provided.
2. Never invent student characteristics or performance.
3. A predicted risk is not a confirmed academic outcome.
4. GNNExplainer scores represent feature or edge mask
   importance, NOT positive or negative effects.
5. SHAP global importance represents overall feature
   importance, NOT the direction of an individual prediction.
6. Do not interpret LIME weights unless their actual
   values and directions are supplied.
7. Never use age, gender or demographic attributes
   to recommend interventions.
8. If records conflict, explicitly report the conflict.
9. Do not silently resolve missing or inconsistent data.
10. Recommend supportive, non-punitive academic actions.
11. Clearly distinguish model evidence from recommendations.
12. State that model predictions are uncertain.

Use these sections:
- Prediction Summary
- Model Evidence
- Data Quality and Limitations
- Recommended Support
- Important Disclaimer

If evidence is insufficient, say so instead of guessing.
"""

def generate_student_explanation(student_data):

    response = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Generate an evidence-based X-LearnNet "
                    "student risk explanation using this data:\n\n"
                    f"{student_data}"
                )
            }
        ],
        options={
            "temperature": 0.1
        }
    )

    return response["message"]["content"]


if __name__ == "__main__":

    sample_data = """
    Student ID: STU012586

    PREDICTIONS:
    GraphSAGE failure probability: 88.92%
    Logistic Regression failure probability: 67.96%
    Actual test label: 0 (no failure)

    LOCAL GNExplainer:
    - forum_participation_count: 0.910953
    - age: 0.891460
    - prior_gpa: 0.860259
    - average_activity_duration: 0.856378
    - assignment_submission_rate: 0.830806

    SHAP GLOBAL FEATURE IMPORTANCE:
    - attendance_rate: 0.032423
    - assignment_submission_rate: 0.027700
    - lms_logins_past_month: 0.027194
    - video_completion_rate: 0.016959

    DATA QUALITY:
    Profile attendance_rate: 45%
    Calculated attendance from records: 79.17%

    The attendance values conflict and require verification.
    Assessment dates also contain chronological inconsistencies.
    No matching intervention outcome records exist for this student.
    """

    explanation = generate_student_explanation(sample_data)

    print("\n========== X-LEARNNET LLM EXPLANATION ==========\n")
    print(explanation)

