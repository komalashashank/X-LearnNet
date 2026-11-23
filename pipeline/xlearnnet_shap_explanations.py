import os
import shap
import numpy as np
import matplotlib.pyplot as plt

from pyspark.sql import SparkSession
from pyspark.ml.classification import GBTClassificationModel
from pyspark.ml.functions import vector_to_array
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.types import StructType, StructField

# --------------------------------------------------
# 1. Start Spark
# --------------------------------------------------
spark = (
    SparkSession.builder
    .appName("XLearnNet_SHAP_Explanations")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

MODEL_PATH = "hdfs://localhost:9000/xlearnnet/models/course_failure_gbt"
DATA_PATH = "hdfs://localhost:9000/xlearnnet/feature_engineered/course_failure_flag/test"

OUTPUT_DIR = "/mnt/d/capstone/X-LearnNet/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------------------------------------
# 2. Load trained GBT model and test data
# --------------------------------------------------
print("Loading GBT model...")
model = GBTClassificationModel.load(MODEL_PATH)

print("Loading test data...")
test_df = spark.read.parquet(DATA_PATH)

# Convert Spark feature vectors into NumPy arrays
test_df = test_df.withColumn(
    "feature_array", vector_to_array("features")
)

rows = test_df.select("feature_array", "label").limit(200).collect()

X = np.array([row["feature_array"] for row in rows], dtype=float)
y = np.array([row["label"] for row in rows], dtype=float)

print("Test sample shape:", X.shape)

# --------------------------------------------------
# 3. Feature names
# --------------------------------------------------
numeric_features = [
    "age", "GPA", "course_load", "avg_course_grade",
    "attendance_rate", "lms_logins_past_month",
    "avg_session_duration_minutes", "assignment_submission_rate",
    "forum_participation_count", "video_completion_rate",
    "cohort_year", "prior_gpa", "average_assessment_score",
    "assessment_count", "total_classes", "attendance_percentage",
    "total_enrollments", "total_lms_events",
    "average_activity_duration", "average_weekly_study_hours"
]

categorical_features = [
    "gender_Male",
    "gender_Female",
    "gender_Other",
    "gender_Not_reported",
    "gender_invalid",

    "major_Arts",
    "major_Computer_Science",
    "major_Engineering",
    "major_Business",
    "major_Undeclared",
    "major_invalid",

    "semester_2026_Spring",
    "semester_2026_Fall",
    "semester_2025_Fall",
    "semester_invalid",

    "program_Undergraduate",
    "program_Postgraduate",
    "program_invalid"
]
feature_names = numeric_features + categorical_features

# --------------------------------------------------
# 4. Spark prediction wrapper
# --------------------------------------------------
schema = StructType([
    StructField("features", VectorUDT(), nullable=False)
])

def predict_probability(data):
    data = np.asarray(data, dtype=float)

    if data.ndim == 1:
        data = data.reshape(1, -1)

    spark_rows = [
        (Vectors.dense(row.tolist()),)
        for row in data
    ]

    prediction_df = spark.createDataFrame(spark_rows, schema)

    predictions = (
        model.transform(prediction_df)
        .select(vector_to_array("probability").alias("prob"))
        .collect()
    )

    return np.array([row["prob"][1] for row in predictions])

# --------------------------------------------------
# 5. SHAP KernelExplainer
# --------------------------------------------------
print("Preparing SHAP background data...")

background = shap.sample(X, min(10, len(X)), random_state=42)

print("Creating SHAP explainer...")
explainer = shap.KernelExplainer(
    predict_probability,
    background
)

explain_data = X[:50]

print("Calculating SHAP values...")
shap_values = explainer.shap_values(
    explain_data,
    nsamples=100
)

# Handle SHAP output format
if isinstance(shap_values, list):
    shap_values = shap_values[0]

shap_values = np.asarray(shap_values)

if shap_values.ndim == 3:
    shap_values = shap_values[:, :, -1]

print("SHAP calculation completed.")
print("SHAP values shape:", shap_values.shape)

# --------------------------------------------------
# 6. SHAP summary plot
# --------------------------------------------------
plt.figure()
shap.summary_plot(
    shap_values,
    explain_data,
    feature_names=feature_names,
    show=False
)

plt.tight_layout()
summary_path = os.path.join(OUTPUT_DIR, "xlearnnet_shap_summary.png")
plt.savefig(summary_path, dpi=300, bbox_inches="tight")
plt.close()

# --------------------------------------------------
# 7. Individual student explanation
# --------------------------------------------------
plt.figure()
shap.waterfall_plot(
    shap.Explanation(
        values=shap_values[0],
        base_values=float(explainer.expected_value),
        data=explain_data[0],
        feature_names=feature_names
    ),
    show=False
)

plt.tight_layout()
individual_path = os.path.join(OUTPUT_DIR, "xlearnnet_shap_student.png")
plt.savefig(individual_path, dpi=300, bbox_inches="tight")
plt.close()

# --------------------------------------------------
# 8. Save numerical SHAP results
# --------------------------------------------------
importance = np.mean(np.abs(shap_values), axis=0)

ranking = sorted(
    zip(feature_names, importance),
    key=lambda item: item[1],
    reverse=True
)

print("\nTOP 15 SHAP FEATURES")
for name, score in ranking[:15]:
    print(f"{name:35s} {score:.6f}")

print("\nSaved files:")
print(summary_path)
print(individual_path)

spark.stop()
