import json
import sys
from pathlib import Path

import pandas as pd
import ollama

ROOT = Path(__file__).resolve().parents[2]
MODEL = "llama3.2:3b"

COMPARISON = ROOT / "outputs/graphsage_vs_logistic_comparison.csv"
GNN_FEATURES = ROOT / "outputs/graphsage/local_gnn_feature_explanations.csv"
GNN_EDGES = ROOT / "outputs/graphsage/local_gnn_edge_explanations.csv"
STUDENTS = ROOT / "data/output/unified_student_repository.csv"

# Current GNN explanation files were generated for this student only.
EXPLAINED_STUDENT = "STU012586"


def load_student_data(student_id):

    comparison = pd.read_csv(COMPARISON)
    students = pd.read_csv(STUDENTS)

    prediction = comparison[
        comparison["student_id"] == student_id
    ]

    profile = students[
        students["student_id"] == student_id
    ]

    if prediction.empty:
        raise ValueError(f"No prediction found for {student_id}")

    if profile.empty:
        raise ValueError(f"No student profile found for {student_id}")

    prediction = prediction.iloc[0]
    profile = profile.iloc[0]

    actual = int(prediction["actual"])
    graphsage_label = int(prediction["predicted"])
    graphsage_probability = float(prediction["failure_probability"])
    lr_probability = float(prediction["lr_probability"])

    # Logistic Regression threshold selected using validation data.
    lr_threshold = 0.20
    lr_label = int(lr_probability >= lr_threshold)

    def classify(actual_label, predicted_label):
        if actual_label == 0 and predicted_label == 1:
            return "False Positive"
        if actual_label == 1 and predicted_label == 0:
            return "False Negative"
        if actual_label == predicted_label == 1:
            return "True Positive"
        return "True Negative"

    result = {
        "student_id": student_id,

        "target_definition": {
            "0": "No course failure",
            "1": "Course failure"
        },

        "actual_test_label": actual,

        "graphsage": {
            "predicted_label": graphsage_label,
            "failure_probability": round(graphsage_probability, 4),
            "classification_result": classify(actual, graphsage_label)
        },

        "logistic_regression": {
            "predicted_label": lr_label,
            "failure_probability": round(lr_probability, 4),
            "decision_threshold": lr_threshold,
            "classification_result": classify(actual, lr_label)
        },

        "academic_profile": {
            "gpa": float(profile["gpa"]),
            "prior_gpa": float(profile["prior_gpa"]),
            "attendance_rate": float(profile["attendance_rate"]),
            "calculated_attendance_rate": float(
                profile["calculated_attendance_rate"]
            ),
            "assignment_submission_rate": float(
                profile["assignment_submission_rate"]
            ),
            "avg_course_grade": float(profile["avg_course_grade"]),
            "lms_logins_past_month": int(
                profile["lms_logins_past_month"]
            ),
            "video_completion_rate": float(
                profile["video_completion_rate"]
            ),
            "total_classes": int(profile["total_classes"]),
            "classes_attended": int(profile["classes_attended"])
        }
    }

    # Only use local GNN explanations for the student they explain.
    if student_id == EXPLAINED_STUDENT:

        features = pd.read_csv(GNN_FEATURES)
        edges = pd.read_csv(GNN_EDGES)

        demographic_terms = ["age", "gender", "cohort_year"]

        features = features[
            ~features["feature"].str.contains(
                "|".join(demographic_terms),
                case=False,
                regex=True
            )
        ]

        result["gnn_explainer_feature_masks"] = (
            features.nlargest(10, "importance").to_dict("records")
        )

        student_edges = edges[
            edges["source_node"] == student_id
        ]

        result["gnn_explainer_top_edges"] = (
            student_edges.nlargest(10, "importance")[
                ["source_node", "target_node", "importance"]
            ].to_dict("records")
        )

    else:
        result["gnn_explainer_note"] = (
            "Student-specific GNNExplainer results are unavailable. "
            "Do not reuse another student's explanation."
        )

    attendance_a = float(profile["attendance_rate"])
    attendance_b = float(profile["calculated_attendance_rate"])

    result["data_quality"] = {
        "attendance_values_match": abs(
            attendance_a - attendance_b
        ) < 0.001,
        "attendance_discrepancy_percentage_points": round(
            abs(attendance_a - attendance_b) * 100, 2
        )
    }

    return result


SYSTEM_PROMPT = """
You are the X-LearnNet student-risk explanation assistant.

Use only the supplied student records and model outputs.

TARGET LABEL DEFINITIONS:
- course_failure_flag = 1 means course failure.
- course_failure_flag = 0 means NO course failure.
- Predicted value 1 means predicted failure.
- Predicted value 0 means predicted no failure.
- Never reverse these definitions.

PREDICTION INTERPRETATION:
- GraphSAGE failure probability is the GraphSAGE model's probability.
- Logistic Regression probability is the Logistic Regression model's probability.
- Report each model and its probability separately.
- Never interchange model names or probabilities.
- Compare predicted labels with the actual target.
- Actual 0 and predicted 1 means FALSE POSITIVE.
- Actual 1 and predicted 0 means FALSE NEGATIVE.
- Never describe a prediction as a confirmed outcome.

NUMERICAL ACCURACY:
- Compare only measurements with compatible meanings and units.
- GPA 2.96 is higher than prior GPA 2.90.
- Never compare assignment submission rate directly with course grade.
- Preserve supplied values and units.
- Do not invent thresholds or categories.

GNN EXPLANATIONS:
- GNNExplainer feature and edge values are mask importance scores.
- They are not percentages, probabilities, signed effects, or causal effects.
- Never say that an important feature or edge is a significant
  factor, cause, driver, or reason for the prediction.
- Describe assessment edges only as graph connections.
- Do not infer assessment performance from graph connections.
- Do not infer student participation or engagement from edge masks.
- Explicitly state that feature and edge mask scores show
  explanation importance, not positive or negative effects.

DATA QUALITY:
- Report recorded attendance as 0.45 and calculated attendance
  as 0.7917 (79.17%).
- State that these values differ by 34.17 percentage points.
- Do not describe either attendance value as lower or higher
  performance because their definitions may differ.
- Never compare assignment submission rate with course grade.
- Do not infer trends from inconsistent assessment dates.
- Do not claim data conflicts caused model predictions.

FAIRNESS AND SUPPORT:
- Do not use demographic attributes to recommend interventions.
- Recommend supportive, non-punitive academic assistance.
- Do not recommend improving attendance until the attendance
  discrepancy has been investigated and resolved.
- Do not recommend verifying LMS logins or video completion
  unless the supplied records show a specific data-quality issue.
- Recommend checking attendance source records before making
  attendance-based decisions.
- Do not invent expected performance benchmarks.
- Make recommendations conditional when evidence is incomplete.

STRICT RULES:
- Never invent student information or model results.
- Never confuse predictions with actual outcomes.
- Never invent SHAP or LIME results.
- Never substitute another student's explanation.
- Clearly state uncertainty and data limitations.
- Do not make unsupported comparisons with expected performance.
- Do not treat GNN mask scores as signed feature effects.
- Do not claim that an assessment connection proves poor performance.
- Never compare values with different units or meanings.
- Do not invent expected performance benchmarks.
- If evidence is insufficient, explicitly say so.
- Keep recommendations conditional on verification of the data.
Use these exact headings:
- Prediction Summary
- Academic Evidence
- Graph Explanation
- Data Quality
- Recommended Support
- Important Disclaimer
"""


def generate_explanation(student_data):

    response = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Generate an evidence-based X-LearnNet report "
                    "from the following verified records:\n\n"
                    + json.dumps(student_data, indent=2, default=str)
                )
            }
        ],
        options={"temperature": 0.1}
    )

    return response["message"]["content"]


if __name__ == "__main__":

    student_id = (
        sys.argv[1] if len(sys.argv) > 1
        else EXPLAINED_STUDENT
    )

    print(f"\nLoading X-LearnNet records for {student_id}...")

    data = load_student_data(student_id)

    print("Generating explanation using Ollama...\n")

    explanation = generate_explanation(data)

    print("=" * 65)
    print("X-LEARNNET AUTOMATIC STUDENT EXPLANATION")
    print("=" * 65)
    print(explanation)

    output_dir = ROOT / "outputs/llm"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"{student_id}_explanation.txt"
    output_file.write_text(explanation, encoding="utf-8")

    print("\nReport saved to:", output_file)
