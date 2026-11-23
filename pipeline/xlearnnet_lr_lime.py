
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from lime.lime_tabular import LimeTabularExplainer
from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegressionModel
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.functions import col

MODEL_PATH = "hdfs://localhost:9000/xlearnnet/models/course_failure_logistic_regression"
DATA_PATH = "hdfs://localhost:9000/xlearnnet/feature_engineered/course_failure_flag/test"

spark = (
    SparkSession.builder
    .appName("XLearnNet-LIME")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

print("Loading GBT model and test data...")

model = LogisticRegressionModel.load(MODEL_PATH)
test_df = spark.read.parquet(DATA_PATH)

target_rows = (
    test_df.filter(col("student_id") == "STU012586")
    .select("student_id", "features", "label")
    .collect()
)

if not target_rows:
    raise ValueError("STU012586 not found in LIME test data")

sample_rows = (
    test_df.filter(col("student_id") != "STU012586")
    .select("student_id", "features", "label")
    .limit(199)
    .collect()
)

rows = target_rows + sample_rows

X = np.array([r.features.toArray() for r in rows])
y = np.array([r.label for r in rows])

feature_names = [
    "age", "GPA", "course_load", "avg_course_grade",
    "attendance_rate", "lms_logins_past_month",
    "avg_session_duration_minutes", "assignment_submission_rate",
    "forum_participation_count", "video_completion_rate",
    "cohort_year", "prior_gpa", "average_assessment_score",
    "assessment_count", "total_classes", "attendance_percentage",
    "total_enrollments", "total_lms_events",
    "average_activity_duration", "average_weekly_study_hours",
    "gender_Male", "gender_Female", "gender_Other",
    "gender_Not_reported", "gender_unknown",
    "major_Arts", "major_Computer_Science", "major_Engineering",
    "major_Business", "major_Undeclared", "major_unknown",
    "semester_2026_Spring", "semester_2026_Fall",
    "semester_2025_Fall", "semester_unknown",
    "program_level_Undergraduate", "program_level_Postgraduate", "program_level_unknown"
]

assert X.shape[1] == len(feature_names), (
    f"Feature mismatch: data={X.shape[1]}, names={len(feature_names)}"
)

print("Test sample shape:", X.shape)

# Build a prediction function compatible with LIME
schema = spark.createDataFrame(
    [(Vectors.dense([0.0] * X.shape[1]),)],
    ["features"]
).schema

def predict_fn(data):
    vectors = [
        (Vectors.dense(row.tolist()),)
        for row in data
    ]

    sdf = spark.createDataFrame(vectors, ["features"])
    predictions = model.transform(sdf)

    probabilities = predictions.select("probability").collect()

    return np.array([
        [float(row["probability"][0]), float(row["probability"][1])]
        for row in probabilities
    ])
explainer = LimeTabularExplainer(
    training_data=X,
    feature_names=feature_names,
    class_names=["No Failure", "Failure"],
    mode="classification",
    discretize_continuous=True,
    random_state=42
)

# Select one student for explanation
student_index = 0  # STU012586 is deliberately first

explanation = explainer.explain_instance(
    X[student_index],
    predict_fn,
    num_features=15,
    num_samples=1000
)

print("\nStudent ID:", rows[student_index].student_id)
print("Actual outcome:", y[student_index])
print("Model prediction:", predict_fn(X[student_index:student_index+1])[0])

lime_results = explanation.as_list(label=1)
pd.DataFrame(
    lime_results,
    columns=["feature_condition", "lime_weight"]
).to_csv(
    "/mnt/d/capstone/X-LearnNet/outputs/xlearnnet_lr_lime_STU012586.csv",
    index=False
)

explanation.save_to_file(
    "/mnt/d/capstone/X-LearnNet/outputs/xlearnnet_lr_lime_student.html"
)

fig = explanation.as_pyplot_figure()
fig.savefig(
    "/mnt/d/capstone/X-LearnNet/outputs/xlearnnet_lr_lime_student.png",
    bbox_inches="tight",
    dpi=300
)
plt.close(fig)

print("\nLIME explanation saved successfully!")

spark.stop()
