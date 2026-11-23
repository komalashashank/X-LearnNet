from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, when, lit, sum as spark_sum
)
from pyspark.ml.functions import vector_to_array
from pyspark.ml.classification import RandomForestClassificationModel

spark = SparkSession.builder \
    .appName("XLearnNet_Final_Test") \
    .getOrCreate()

model_path = (
    "hdfs://localhost:9000/xlearnnet/models/"
    "course_failure_random_forest_weighted"
)

test_path = (
    "hdfs://localhost:9000/xlearnnet/feature_engineered/"
    "course_failure_flag/test"
)

threshold = 0.41

model = RandomForestClassificationModel.load(model_path)
test_data = spark.read.parquet(test_path)

predictions = model.transform(test_data)

results = predictions.select(
    col("label"),
    vector_to_array("probability")[1].alias("probability")
).withColumn(
    "prediction",
    when(col("probability") >= lit(threshold), 1.0)
    .otherwise(0.0)
).cache()

counts = results.agg(
    spark_sum(when(
        (col("label") == 1) & (col("prediction") == 1), 1
    ).otherwise(0)).alias("TP"),

    spark_sum(when(
        (col("label") == 0) & (col("prediction") == 1), 1
    ).otherwise(0)).alias("FP"),

    spark_sum(when(
        (col("label") == 1) & (col("prediction") == 0), 1
    ).otherwise(0)).alias("FN"),

    spark_sum(when(
        (col("label") == 0) & (col("prediction") == 0), 1
    ).otherwise(0)).alias("TN")
).first()

tp, fp = counts["TP"], counts["FP"]
fn, tn = counts["FN"], counts["TN"]

precision = tp / (tp + fp) if tp + fp else 0
recall = tp / (tp + fn) if tp + fn else 0
accuracy = (tp + tn) / (tp + tn + fp + fn)
f1 = (
    2 * precision * recall / (precision + recall)
    if precision + recall else 0
)

print("\nFINAL TEST RESULTS")
print("Threshold:", threshold)
print("Test records:", tp + fp + fn + tn)
print("Accuracy:", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1-score:", round(f1, 4))

print("\nCONFUSION MATRIX")
print("True Negatives:", tn)
print("False Positives:", fp)
print("False Negatives:", fn)
print("True Positives:", tp)

results.unpersist()
spark.stop()
