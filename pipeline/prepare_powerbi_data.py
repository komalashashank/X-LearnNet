from pathlib import Path
import pandas as pd

ROOT = Path(".")
OUT = ROOT / "powerbi_data"
OUT.mkdir(exist_ok=True)

# 1. Load source datasets
students = pd.read_csv("data/output/unified_student_repository.csv")
comparison = pd.read_csv("outputs/graphsage_vs_logistic_comparison.csv")
shap = pd.read_csv("outputs/xlearnnet_lr_shap_importance.csv")
lime = pd.read_csv("outputs/xlearnnet_lr_lime_STU012586.csv")
gnn = pd.read_csv("outputs/graphsage/local_gnn_feature_explanations.csv")

# 2. Student overview: all 20,000 students
overview_cols = [
    "student_id", "gpa", "prior_gpa", "avg_course_grade",
    "course_load", "attendance_rate", "calculated_attendance_rate",
    "assignment_submission_rate", "lms_logins_past_month",
    "video_completion_rate", "forum_participation_count",
    "avg_learning_hours", "risk_level", "enrollment_status",
    "program_level", "major", "semester", "data_split",
    "support_needs_flag"
]

overview = students[overview_cols].copy()
overview = overview.rename(columns={
    "attendance_rate": "recorded_attendance_rate"
})
overview.to_csv(OUT / "student_risk_overview.csv", index=False)

# 3. Model predictions: test-set students only
if not (comparison["actual"].astype(int) ==
        comparison["label"].astype(int)).all():
    raise ValueError("The actual and label columns do not match.")

predictions = comparison.drop(columns=["label"]).copy()

predictions["lr_prediction"] = (
    predictions["lr_probability"] >= 0.20
).astype(int)

predictions = predictions.rename(columns={
    "actual": "actual_course_failure",
    "predicted": "graphsage_prediction",
    "failure_probability": "graphsage_failure_probability",
    "lr_probability": "lr_failure_probability"
})

predictions["graphsage_result"] = predictions.apply(
    lambda r: (
        "True Positive" if r.actual_course_failure == 1 and r.graphsage_prediction == 1
        else "False Positive" if r.actual_course_failure == 0 and r.graphsage_prediction == 1
        else "False Negative" if r.actual_course_failure == 1 and r.graphsage_prediction == 0
        else "True Negative"
    ), axis=1
)

predictions["lr_result"] = predictions.apply(
    lambda r: (
        "True Positive" if r.actual_course_failure == 1 and r.lr_prediction == 1
        else "False Positive" if r.actual_course_failure == 0 and r.lr_prediction == 1
        else "False Negative" if r.actual_course_failure == 1 and r.lr_prediction == 0
        else "True Negative"
    ), axis=1
)

student_details = students[
    ["student_id", "gpa", "attendance_rate",
     "calculated_attendance_rate", "assignment_submission_rate",
     "risk_level", "enrollment_status"]
]

predictions = predictions.merge(
    student_details, on="student_id", how="left",
    validate="one_to_one"
)

predictions.to_csv(OUT / "model_predictions.csv", index=False)

# 4. Model performance comparison
metrics = pd.DataFrame([
    {
        "model": "Logistic Regression",
        "accuracy": 0.6629,
        "precision": 0.2980,
        "recall": 0.5829,
        "f1_score": 0.3943,
        "roc_auc": 0.6939,
        "pr_auc": 0.3340
    },
    {
        "model": "GraphSAGE",
        "accuracy": 0.5857,
        "precision": 0.2739,
        "recall": 0.7268,
        "f1_score": 0.3978,
        "roc_auc": 0.6827,
        "pr_auc": 0.3234
    }
])
metrics.to_csv(OUT / "model_performance.csv", index=False)

# 5. Explainability datasets
shap.sort_values(
    "mean_abs_shap", ascending=False
).to_csv(OUT / "shap_global_importance.csv", index=False)

lime["impact_direction"] = lime["lime_weight"].apply(
    lambda x: "Pushes toward failure" if x > 0
    else "Pushes away from failure" if x < 0
    else "No contribution"
)
lime.to_csv(OUT / "lime_student_explanation.csv", index=False)

gnn.sort_values(
    "importance", ascending=False
).to_csv(OUT / "graphsage_feature_importance.csv", index=False)

# 6. Summary
print("\nPOWER BI DATA PREPARATION COMPLETE")
print("-" * 45)

for file in sorted(OUT.glob("*.csv")):
    df = pd.read_csv(file)
    print(f"{file.name}: {len(df):,} rows, {len(df.columns)} columns")

print("\nOriginal datasets were not modified.")
print("Output folder:", OUT.resolve())
