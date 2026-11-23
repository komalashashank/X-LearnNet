
from pathlib import Path
import pandas as pd

OUT = Path("outputs")
GNN = OUT / "graphsage"
REPORT_DIR = OUT / "unified_explainability"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

STUDENT = "STU012586"

# Load verified explanation results
comparison = pd.read_csv(OUT / "graphsage_vs_logistic_comparison.csv")
shap_global = pd.read_csv(OUT / "xlearnnet_lr_shap_importance.csv")
shap_local = pd.read_csv(OUT / "xlearnnet_lr_shap_STU012586.csv")
lime_local = pd.read_csv(OUT / "xlearnnet_lr_lime_STU012586.csv")
gnn_features = pd.read_csv(GNN / "local_gnn_feature_explanations.csv")
gnn_edges = pd.read_csv(GNN / "local_gnn_edge_explanations.csv")

profile = pd.read_csv("data/output/unified_student_repository.csv")
student = profile[profile["student_id"] == STUDENT]

if len(student) != 1:
    raise ValueError(
        f"Expected one profile for {STUDENT}; found {len(student)}"
    )

student = student.iloc[0]

llm_path = OUT / "llm" / f"{STUDENT}_explanation.txt"
if not llm_path.exists():
    raise FileNotFoundError(f"LLM explanation not found: {llm_path}")
llm_explanation = llm_path.read_text(encoding="utf-8").strip()

prediction = comparison[comparison["student_id"] == STUDENT]

if len(prediction) != 1:
    raise ValueError(f"Expected one prediction row for {STUDENT}")

prediction = prediction.iloc[0]

actual = int(prediction["actual"])
lr_probability = float(prediction["lr_probability"])
gnn_probability = float(prediction["failure_probability"])
lr_predicted = int(lr_probability >= 0.20)
gnn_predicted = int(prediction["predicted"])


def result_name(predicted):
    if actual == 0 and predicted == 0:
        return "True Negative"
    if actual == 0 and predicted == 1:
        return "False Positive"
    if actual == 1 and predicted == 0:
        return "False Negative"
    return "True Positive"


# SHAP: global magnitude and local signed contributions
global_top = shap_global.sort_values(
    "mean_abs_shap", ascending=False
).head(10)

local_positive = shap_local[
    shap_local["shap_value"] > 0
].sort_values(
    "shap_value", ascending=False
).head(10)

local_negative = shap_local[
    shap_local["shap_value"] < 0
].sort_values(
    "shap_value", ascending=True
).head(10)


# LIME: local positive and negative weights
lime_positive = lime_local[
    lime_local["lime_weight"] > 0
].sort_values(
    "lime_weight", ascending=False
).head(10)

lime_negative = lime_local[
    lime_local["lime_weight"] < 0
].sort_values(
    "lime_weight", ascending=True
).head(10)


# GNN feature masks: exclude demographic terms from
# student-facing support recommendations
demographic_terms = ["gender", "age", "cohort"]

gnn_features["is_demographic"] = (
    gnn_features["feature"]
    .str.lower()
    .apply(lambda name: any(term in name for term in demographic_terms))
)

gnn_demographic = gnn_features[
    gnn_features["is_demographic"]
]

gnn_support_features = gnn_features[
    ~gnn_features["is_demographic"]
].sort_values(
    "importance", ascending=False
).head(10)


# GNN edge mask scores
top_edges = gnn_edges.sort_values(
    "importance", ascending=False
).head(10)


# Academic data quality
recorded_attendance = float(student["attendance_rate"])

calculated_attendance = (
    float(student["classes_attended"]) / float(student["total_classes"])
    if float(student["total_classes"]) else None
)

attendance_difference = (
    abs(recorded_attendance - calculated_attendance) * 100
    if calculated_attendance is not None else None
)


def table_rows(df, columns, formatters=None):
    lines = []

    for _, row in df.iterrows():
        values = []

        for col in columns:
            value = row[col]

            if formatters and col in formatters:
                value = formatters[col](value)

            values.append(str(value))

        lines.append("| " + " | ".join(values) + " |")

    return "\n".join(lines)

report = f"""# X-LearnNet Unified Explainability Report

## 1. Student and prediction summary

- **Student ID:** {STUDENT}
- **Actual test label:** {actual} ({'Course failure' if actual == 1 else 'No course failure'})
- **Logistic Regression probability:** {lr_probability:.4f} ({lr_probability*100:.2f}%)
- **Logistic Regression predicted class:** {lr_predicted} ({'Failure' if lr_predicted else 'No failure'})
- **Logistic Regression result:** {result_name(lr_predicted)}
- **GraphSAGE probability:** {gnn_probability:.4f} ({gnn_probability*100:.2f}%)
- **GraphSAGE predicted class:** {gnn_predicted} ({'Failure' if gnn_predicted else 'No failure'})
- **GraphSAGE result:** {result_name(gnn_predicted)}

Both models produce false-positive predictions for this test record. Their probabilities represent model estimates, not certainty.

## 2. Academic profile

- GPA: {student['gpa']}
- Prior GPA: {student['prior_gpa']}
- Average course grade: {student['avg_course_grade']}
- Recorded attendance rate: {recorded_attendance:.4f}
- Calculated attendance rate: {f'{calculated_attendance:.4f}' if calculated_attendance is not None else 'Unavailable'}
- Assignment submission rate: {student['assignment_submission_rate']}
- LMS logins in past month: {student['lms_logins_past_month']}
- Video completion rate: {student['video_completion_rate']}
- Classes attended: {student['classes_attended']} / {student['total_classes']}

### Data-quality warning

The recorded attendance rate and the rate calculated from attendance records
{'differ by ' + f'{attendance_difference:.2f} percentage points' if attendance_difference is not None else 'could not be compared'}.
This discrepancy requires investigation. Neither value should silently replace the other.

## 3. Logistic Regression — SHAP

### Global feature importance

Mean absolute SHAP values across the 50 explained test samples.
These values indicate global contribution magnitude, not direction.

| Feature | Mean absolute SHAP |
|---|---:|
{table_rows(global_top, ['feature', 'mean_abs_shap'], {'mean_abs_shap': lambda x: f'{x:.6f}'})}

### Individual SHAP contributions for {STUDENT}

Positive contributions push the model output toward the Failure class; negative
contributions push it away from that class.

**Largest positive contributions**

| Feature | SHAP value |
|---|---:|
{table_rows(local_positive, ['feature', 'shap_value'], {'shap_value': lambda x: f'{x:.6f}'})}

**Largest negative contributions**

| Feature | SHAP value |
|---|---:|
{table_rows(local_negative, ['feature', 'shap_value'], {'shap_value': lambda x: f'{x:.6f}'})}

## 4. Logistic Regression — LIME

Local surrogate-model weights for the Failure class.
Positive weights support the Failure prediction; negative weights oppose it.
These weights are local approximations, not causal effects.

**Largest positive weights**

| Feature condition | LIME weight |
|---|---:|
{table_rows(lime_positive, ['feature_condition', 'lime_weight'], {'lime_weight': lambda x: f'{x:.6f}'})}

**Largest negative weights**

| Feature condition | LIME weight |
|---|---:|
{table_rows(lime_negative, ['feature_condition', 'lime_weight'], {'lime_weight': lambda x: f'{x:.6f}'})}

## 5. GraphSAGE — GNNExplainer

### Local feature mask scores

The following are the strongest non-demographic feature mask scores for this
student. They measure the importance of features to the explained prediction;
they are not probabilities, signed effects, or causal relationships.

| Feature | Mask importance |
|---|---:|
{table_rows(gnn_support_features, ['feature', 'importance'], {'importance': lambda x: f'{x:.6f}'})}

Demographic feature masks were also present in the explanation. They are
excluded from student-facing support recommendations and should be examined
in a separate fairness audit.

### Important graph edges

| Source node | Connected node | Edge mask score |
|---|---|---:|
{table_rows(top_edges, ['source_node', 'target_node', 'importance'], {'importance': lambda x: f'{x:.6f}'})}

Edge scores are mask importance values, not probabilities or signed effects.

## 6. Integrated interpretation

Both models predicted course failure for {STUDENT}, while the actual test label
is no course failure. Therefore, this case is a false positive for both models.

SHAP and LIME provide complementary explanations of the Logistic Regression
output. GNNExplainer provides feature and graph-edge mask information for
GraphSAGE. These methods explain model behavior; they do not establish that
the highlighted features caused an academic outcome.

The attendance discrepancy must be resolved before drawing conclusions based
on attendance. No unsupported benchmark is used to describe the student's
attendance, assignment submission, or grades as below expectations.

## 7. Suggested follow-up

- Verify the attendance records and reconcile the two attendance values.
- Review assignment and LMS records with the appropriate academic team.
- Treat the model outputs as prompts for human review, not as definitive
  judgments about the student.
- Evaluate false-positive rates and fairness across relevant student groups.
- Validate the framework on real, appropriately governed institutional data
  before operational use.

## 8. Model performance comparison

The following metrics were measured on the same held-out test set.

| Metric | Logistic Regression | GraphSAGE |
|---|---:|---:|
| Accuracy | 66.29% | 58.57% |
| Precision | 29.80% | 27.39% |
| Recall | 58.29% | 72.68% |
| F1-score | 39.43% | 39.78% |
| ROC-AUC | 0.6939 | 0.6827 |
| PR-AUC | 0.3340 | 0.3234 |

GraphSAGE achieved higher recall, identifying a greater proportion of
students who experienced course failure. Logistic Regression achieved
higher accuracy, precision, ROC-AUC and PR-AUC. Their F1-scores were
similar. Neither model is superior across every metric.

## 9. LLM-generated student explanation

The following narrative was generated using the Llama 3.2 model through
Ollama, using the supplied student profile, model predictions and GNN
explanation outputs.

> **Review note:** This is an AI-generated narrative. Numerical results
> and evidence-based sections elsewhere in this report remain authoritative.
> Recommendations require human review, particularly where data conflicts
> or unsupported academic benchmarks may exist.

{llm_explanation}

## 10. Limitations

- The current dataset is synthetic.
- The predictions are not proof of future student outcomes.
- SHAP and LIME explain the Logistic Regression model; GNNExplainer explains
  GraphSAGE. Their scores are not directly comparable.
- GNNExplainer mask scores do not provide signed feature effects.
- The current graph setup uses transductive message passing.
- The train/validation/test split has not been established as a temporal split.
- The attendance data contains a discrepancy for this student.

---
Generated by the X-LearnNet unified explainability pipeline.
"""

report_path = REPORT_DIR / f"{STUDENT}_unified_explainability_report.md"
report_path.write_text(report, encoding="utf-8")

# Export a machine-readable summary
summary = {
    "model_performance": {
        "logistic_regression": {
            "accuracy": 0.6629,
            "precision": 0.2980,
            "recall": 0.5829,
            "f1": 0.3943,
            "roc_auc": 0.6939,
            "pr_auc": 0.3340
        },
        "graphsage": {
            "accuracy": 0.5857,
            "precision": 0.2739,
            "recall": 0.7268,
            "f1": 0.3978,
            "roc_auc": 0.6827,
            "pr_auc": 0.3234
        }
    },
    "llm_explanation": llm_explanation,
    "student_id": STUDENT,
    "actual_label": actual,
    "lr_probability": lr_probability,
    "lr_prediction": lr_predicted,
    "lr_result": result_name(lr_predicted),
    "graphsage_probability": gnn_probability,
    "graphsage_prediction": gnn_predicted,
    "graphsage_result": result_name(gnn_predicted),
    "recorded_attendance": recorded_attendance,
    "calculated_attendance": calculated_attendance,
    "attendance_discrepancy_percentage_points": attendance_difference,
    "global_shap_top_features": global_top.to_dict("records"),
    "local_shap_top_positive": local_positive.to_dict("records"),
    "lime_top_positive": lime_positive.to_dict("records"),
    "gnn_top_non_demographic_features": gnn_support_features.to_dict("records"),
    "gnn_top_edges": top_edges.to_dict("records"),
}

import json
(REPORT_DIR / f"{STUDENT}_unified_explainability_summary.json").write_text(
    json.dumps(summary, indent=2, default=float),
    encoding="utf-8"
)

print("Unified report generated:")
print(report_path)
print(REPORT_DIR / f"{STUDENT}_unified_explainability_summary.json")
print("\nPrediction results:")
print("Logistic Regression:", result_name(lr_predicted))
print("GraphSAGE:", result_name(gnn_predicted))
print("Attendance discrepancy:", f"{attendance_difference:.2f} percentage points")
