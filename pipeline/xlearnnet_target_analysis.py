from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, avg, min, max

spark = SparkSession.builder \
    .appName("XLearnNet_Target_Analysis") \
    .getOrCreate()

targets = spark.read.csv(
    "hdfs:///xlearnnet/raw/student_outcomes_targets.csv",
    header=True,
    inferSchema=True
)

print("Total target records:", targets.count())

print("\nCOURSE FAILURE DISTRIBUTION")
targets.groupBy("course_failure_flag").count().show()

print("\nDROPOUT DISTRIBUTION")
targets.groupBy("dropout_flag").count().show()

print("\nNEXT SEMESTER GPA STATISTICS")
targets.select(
    avg("next_semester_GPA").alias("average_GPA"),
    min("next_semester_GPA").alias("minimum_GPA"),
    max("next_semester_GPA").alias("maximum_GPA")
).show()

spark.stop()
