from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("XLearnNet_Split_Analysis") \
    .getOrCreate()

df = spark.read.csv(
    "hdfs:///xlearnnet/processed/training_dataset",
    header=True,
    inferSchema=True
)

print("DATA SPLIT DISTRIBUTION")
df.groupBy("data_split").count().show()

print("RISK LEVEL DISTRIBUTION")
df.groupBy("risk_level").count().show()

print("ENROLLMENT STATUS DISTRIBUTION")
df.groupBy("enrollment_status").count().show()

print("TOTAL RECORDS:", df.count())

spark.stop()
