from pyspark.sql import SparkSession
from pyspark.sql.functions import col

spark = SparkSession.builder \
    .appName("XLearnNet_Target_Split") \
    .getOrCreate()

df = spark.read.csv(
    "hdfs:///xlearnnet/processed/training_dataset",
    header=True,
    inferSchema=True
)

targets = [
    "course_failure_flag",
    "dropout_flag"
]

for target in targets:
    print(f"\nDISTRIBUTION: {target}")

    df.groupBy("data_split", target) \
      .count() \
      .orderBy("data_split", target) \
      .show()

spark.stop()
