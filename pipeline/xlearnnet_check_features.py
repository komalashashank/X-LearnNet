from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("XLearnNet_Feature_Check") \
    .getOrCreate()

path = "hdfs://localhost:9000/xlearnnet/feature_engineered/course_failure_flag/train"

df = spark.read.parquet(path)

print("DATASET SCHEMA:")
df.printSchema()

print("TOTAL RECORDS:", df.count())

print("SAMPLE RECORDS:")
df.show(5, truncate=False)

spark.stop()
