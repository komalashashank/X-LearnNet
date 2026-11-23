
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, when

spark = SparkSession.builder \
    .appName("XLearnNet_Leakage_Audit") \
    .getOrCreate()

# Load training dataset from HDFS
df = spark.read.option("header", True).option(
    "inferSchema", True
).csv("hdfs://localhost:9000/xlearnnet/processed/training_dataset")

print("\n========== DATASET OVERVIEW ==========")
print("Total records:", df.count())
print("Total columns:", len(df.columns))

print("\n========== AVAILABLE COLUMNS ==========")
for column in df.columns:
    print(column)

# Columns requiring special investigation
suspicious_columns = [
    "risk_level",
    "enrollment_status",
    "support_needs_flag"
]

print("\n========== LEAKAGE INVESTIGATION ==========")

for column in suspicious_columns:
    if column in df.columns:
        print(f"\nDistribution of {column}:")
        df.groupBy(column).count().orderBy(
            col("count").desc()
        ).show(truncate=False)

# Check target distributions
targets = [
    "course_failure_flag",
    "dropout_flag"
]

for target in targets:
    if target in df.columns:
        print(f"\nDistribution of {target}:")
        df.groupBy(target).count().show()

print("\n========== MISSING VALUE CHECK ==========")

expressions = [
    count(when(col(c).isNull(), c)).alias(c)
    for c in df.columns
]

df.select(*expressions).show(truncate=False)

print("\nLeakage audit completed.")

spark.stop()
