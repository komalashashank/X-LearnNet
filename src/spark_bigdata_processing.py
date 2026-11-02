
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    avg, count, sum as spark_sum, round as spark_round
)

# Initialize Spark
spark = SparkSession.builder \
    .appName("X-LearnNet-BigData-Processing") \
    .master("local[*]") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

print("X-LEARNNET BIG DATA PROCESSING")
print("=" * 50)

# Load datasets
students = spark.read.csv(
    "data/processed/students_master.csv",
    header=True,
    inferSchema=True
)

assessments = spark.read.csv(
    "data/processed/assessment_results.csv",
    header=True,
    inferSchema=True
)

attendance = spark.read.csv(
    "data/processed/attendance_records.csv",
    header=True,
    inferSchema=True
)

# Display record counts
print("Students:", students.count())
print("Assessments:", assessments.count())
print("Attendance:", attendance.count())

# Aggregate assessment scores
assessment_summary = assessments.groupBy(
    "student_id"
).agg(
    spark_round(avg("score"), 2).alias("average_assessment_score"),
    count("score").alias("total_assessments")
)

# Aggregate attendance
attendance_summary = attendance.groupBy(
    "student_id"
).agg(
    count("*").alias("total_classes"),
    spark_sum("attended_flag").alias("classes_attended")
)

attendance_summary = attendance_summary.withColumn(
    "attendance_percentage",
    spark_round(
        attendance_summary.classes_attended /
        attendance_summary.total_classes * 100,
        2
    )
)

# Join all datasets
final_data = students \
    .join(assessment_summary, "student_id", "left") \
    .join(attendance_summary, "student_id", "left")

# Display results
print("\nFINAL STUDENT ANALYTICS")

final_data.select(
    "student_id",
    "gpa",
    "average_assessment_score",
    "attendance_percentage"
).show(10, truncate=False)

print("Total integrated students:", final_data.count())

# Save output

# Save Spark results locally using Pandas
import os

output_path = "data/output/spark_student_analytics.csv"

os.makedirs("data/output", exist_ok=True)

final_data.toPandas().to_csv(
    output_path,
    index=False
)

print("\nSpark analytics saved successfully!")
print("Output:", output_path)

spark.stop()

