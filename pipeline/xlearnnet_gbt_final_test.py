from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, sum as spark_sum
from pyspark.ml.classification import GBTClassificationModel
from pyspark.ml.evaluation import BinaryClassificationEvaluator

spark = SparkSession.builder \
    .appName("XLearnNet_GBT_Final_Test") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

model_path = "hdfs://localhost:9000/xlearnnet/models/course_failure_gbt"
test_path = "hdfs://localhost:9000/xlearnnet/feature_engineered/course_failure_flag/test"

print("Loading saved GBT model...")
model = GBTClassificationModel.load(model_path)

print("Loading untouched test dataset...")
test_data = spark.read.parquet(test_path)

print("Test records:", test_data.count())

predictions = model.transform(test_data).cache()

counts = predictions.agg(
    spark_sum(when((col("label") == 1) & (col("prediction") == 1), 1).otherwise(0)).alias("TP"),
    spark_sum(when((col("label") == 0) & (col("prediction") == 1), 1).otherwise(0)).alias("FP"),
    spark_sum(when((col("label") == 1) & (col("prediction") == 0), 1).otherwise(0)).alias("FN"),
    spark_sum(when((col("label") == 0) & (col("prediction") == 0), 1).otherwise(0)).alias("TN")
).first()

tp, fp, fn, tn = [counts[x] for x in ["TP", "FP", "FN", "TN"]]

total = tp + fp + fn + tn
accuracy = (tp + tn) / total if total else 0
precision = tp / (tp + fp) if tp + fp else 0
recall = tp / (tp + fn) if tp + fn else 0
specificity = tn / (tn + fp) if tn + fp else 0
f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0

pr_auc = BinaryClassificationEvaluator(
    labelCol="label",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderPR"
).evaluate(predictions)

roc_auc = BinaryClassificationEvaluator(
    labelCol="label",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC"
).evaluate(predictions)

print("\n===== GBT FINAL TEST RESULTS =====")
print("Test records:", total)
print("Accuracy:", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("Specificity:", round(specificity, 4))
print("F1-score:", round(f1, 4))
print("PR-AUC:", round(pr_auc, 4))
print("ROC-AUC:", round(roc_auc, 4))

print("\n===== CONFUSION MATRIX =====")
print("True Negatives:", tn)
print("False Positives:", fp)
print("False Negatives:", fn)
print("True Positives:", tp)

predictions.unpersist()
spark.stop()
