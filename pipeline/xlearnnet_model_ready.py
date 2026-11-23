
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

spark = SparkSession.builder \
    .appName("XLearnNet_Model_Ready_Datasets") \
    .getOrCreate()

# Load the existing training dataset
df = spark.read.option("header", True) \
    .option("inferSchema", True) \
    .csv("hdfs://localhost:9000/xlearnnet/processed/training_dataset")

# Prediction targets
targets = [
    "course_failure_flag",
    "dropout_flag",
    "next_semester_GPA"
]

# Exclude identifiers, split metadata, suspicious columns
# and every prediction target from model inputs
excluded = {
    "student_id",
    "data_split",
    "risk_level",
    "enrollment_status",
    "support_needs_flag",
    *targets
}

features = [c for c in df.columns if c not in excluded]

print("Total input features:", len(features))
print("Excluded columns:", sorted(excluded))

# Create one dataset for each prediction task
for target in targets:

    output_columns = [
        "student_id",
        "data_split"
    ] + features + [target]

    model_df = df.select(*output_columns)

    output_path = (
        "hdfs://localhost:9000/xlearnnet/model_ready/" + target
    )

    model_df.write.mode("overwrite") \
        .option("header", True) \
        .csv(output_path)

    print("\nCreated dataset:", target)
    print("Records:", model_df.count())
    print("Features:", len(features))
    print("Target:", target)

    print("\nTarget distribution by split:")
    model_df.groupBy("data_split", target) \
        .count().orderBy("data_split", target) \
        .show()

print("\nAll three model-ready datasets created successfully!")

spark.stop()

