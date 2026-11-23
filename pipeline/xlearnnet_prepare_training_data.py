from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("XLearnNet_Training_Data") \
    .getOrCreate()

# Load student features
features = spark.read.csv(
    "hdfs:///xlearnnet/processed/student_features",
    header=True,
    inferSchema=True
)

# Load future outcome labels
targets = spark.read.csv(
    "hdfs:///xlearnnet/raw/student_outcomes_targets.csv",
    header=True,
    inferSchema=True
)

# Keep only required target columns
labels = targets.select(
    "student_id",
    "next_semester_GPA",
    "course_failure_flag",
    "dropout_flag"
)

# Join features and labels
training_data = features.join(
    labels,
    on="student_id",
    how="inner"
)

print("Total training records:", training_data.count())
print("Total columns:", len(training_data.columns))

print("\nTraining dataset preview:")
training_data.show(10, truncate=False)

# Save to HDFS
output_path = "hdfs:///xlearnnet/processed/training_dataset"

training_data.write \
    .mode("overwrite") \
    .option("header", True) \
    .csv(output_path)

print("Training dataset saved successfully!")

spark.stop()
