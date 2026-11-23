from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, avg, count, sum, when, round
)

spark = SparkSession.builder \
    .appName("XLearnNet-Educational-BigData-Pipeline") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

BASE = "hdfs://localhost:9000/xlearnnet/raw/"

# Load datasets
students = spark.read.csv(BASE + "students_master.csv", header=True, inferSchema=True)
assessments = spark.read.csv(BASE + "assessment_results.csv", header=True, inferSchema=True)
attendance = spark.read.csv(BASE + "attendance_records.csv", header=True, inferSchema=True)
enrollments = spark.read.csv(BASE + "course_enrollments.csv", header=True, inferSchema=True)
lms = spark.read.csv(BASE + "lms_activity_events.csv", header=True, inferSchema=True)
weekly = spark.read.csv(BASE + "weekly_learning_sequences.csv", header=True, inferSchema=True)

print("All datasets loaded successfully")

# Assessment features
assessment_features = assessments.groupBy("student_id").agg(
    round(avg("score"), 2).alias("average_assessment_score"),
    count("*").alias("assessment_count")
)

# Attendance features
attendance_features = attendance.groupBy("student_id").agg(
    count("*").alias("total_classes"),
    round(avg(col("attended_flag").cast("double")) * 100, 2)
    .alias("attendance_percentage")
)

# Enrollment features
enrollment_features = enrollments.groupBy("student_id").agg(
    count("*").alias("total_enrollments")
)

# LMS activity features
lms_features = lms.groupBy("student_id").agg(
    count("*").alias("total_lms_events"),
    round(avg("duration_seconds") / 60, 2).alias("average_activity_duration")
)

# Weekly learning features
weekly_features = weekly.groupBy("student_id").agg(
    round(avg("learning_hours"), 2).alias("average_weekly_study_hours")
)

# Combine features
final_data = students

for feature_df in [
    assessment_features,
    attendance_features,
    enrollment_features,
    lms_features,
    weekly_features
]:
    final_data = final_data.join(
        feature_df, on="student_id", how="left"
    )

# Handle missing aggregated values
final_data = final_data.fillna(0)

print("Total students:", final_data.count())
print("Total features:", len(final_data.columns))

final_data.show(10, truncate=False)

# Save processed data to HDFS
OUTPUT = "hdfs://localhost:9000/xlearnnet/processed/student_features"

final_data.write \
    .mode("overwrite") \
    .option("header", True) \
    .csv(OUTPUT)

print("X-LearnNet Spark pipeline completed!")
print("Saved to:", OUTPUT)

spark.stop()
