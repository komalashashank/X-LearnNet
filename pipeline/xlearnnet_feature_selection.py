from pyspark.sql import SparkSession
from pyspark.sql.functions import col

spark = SparkSession.builder \
    .appName("XLearnNet_Feature_Selection") \
    .getOrCreate()

# Load training dataset from HDFS
df = spark.read.csv(
    "hdfs:///xlearnnet/processed/training_dataset",
    header=True,
    inferSchema=True
)

# Numerical features
numerical_features = [
    "age", "GPA", "course_load", "avg_course_grade",
    "attendance_rate", "lms_logins_past_month",
    "avg_session_duration_minutes",
    "assignment_submission_rate",
    "forum_participation_count",
    "video_completion_rate", "cohort_year",
    "prior_gpa", "support_needs_flag",
    "average_assessment_score", "assessment_count",
    "total_classes", "attendance_percentage",
    "total_enrollments", "total_lms_events",
    "average_activity_duration",
    "average_weekly_study_hours"
]

# Categorical features
categorical_features = [
    "gender", "major", "semester", "program_level"
]

# Keep identifiers, split and prediction targets
metadata = ["student_id", "data_split"]

targets = [
    "course_failure_flag",
    "dropout_flag",
    "next_semester_GPA"
]

selected_columns = (
    metadata + numerical_features +
    categorical_features + targets
)

# Select features and targets
selected_df = df.select(*selected_columns)

print("Total records:", selected_df.count())
print("Total selected columns:", len(selected_df.columns))

print("\nSelected feature columns:")
print(selected_df.columns)

print("\nMissing values:")
for column in selected_df.columns:
    missing = selected_df.filter(col(column).isNull()).count()
    if missing > 0:
        print(column, missing)

# Save selected dataset to HDFS
output_path = "hdfs:///xlearnnet/processed/selected_features"

selected_df.write \
    .mode("overwrite") \
    .option("header", True) \
    .csv(output_path)

print("\nFeature selection completed!")
print("Saved to:", output_path)

spark.stop()
