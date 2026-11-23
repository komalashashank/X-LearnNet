from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum as spark_sum, when
from pyspark.ml.classification import (
    RandomForestClassificationModel,
    GBTClassificationModel
)
from pyspark.ml.evaluation import (
    MulticlassClassificationEvaluator,
    BinaryClassificationEvaluator
)
import csv

spark = SparkSession.builder \
    .appName("XLearnNet_Corrected_Evaluation") \
    .getOrCreate()

# Load common test dataset
test_path = (
    "hdfs://localhost:9000/xlearnnet/feature_engineered/"
    "course_failure_flag/test"
)

test_df = spark.read.parquet(test_path)

print("TEST RECORDS:", test_df.count())

# Load trained models
models = {
    "Random Forest": RandomForestClassificationModel.load(
        "hdfs://localhost:9000/xlearnnet/models/"
        "course_failure_random_forest"
    ),

    "Weighted Random Forest": RandomForestClassificationModel.load(
        "hdfs://localhost:9000/xlearnnet/models/"
        "course_failure_random_forest_weighted"
    ),

    "GBT": GBTClassificationModel.load(
        "hdfs://localhost:9000/xlearnnet/models/"
        "course_failure_gbt"
    )
}

# Positive-class evaluators
precision_eval = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="precisionByLabel",
    metricLabel=1.0
)

recall_eval = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="recallByLabel",
    metricLabel=1.0
)

f1_eval = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="fMeasureByLabel",
    metricLabel=1.0
)

pr_eval = BinaryClassificationEvaluator(
    labelCol="label",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderPR"
)

results = []

for name, model in models.items():

    print("\n==========", name, "==========")

    predictions = model.transform(test_df)

    # Calculate confusion matrix counts
    counts = predictions.agg(
        spark_sum(
            when((col("label") == 1) & (col("prediction") == 1), 1)
            .otherwise(0)
        ).alias("TP"),

        spark_sum(
            when((col("label") == 0) & (col("prediction") == 1), 1)
            .otherwise(0)
        ).alias("FP"),

        spark_sum(
            when((col("label") == 1) & (col("prediction") == 0), 1)
            .otherwise(0)
        ).alias("FN"),

        spark_sum(
            when((col("label") == 0) & (col("prediction") == 0), 1)
            .otherwise(0)
        ).alias("TN")
    ).first()

    tp, fp, fn, tn = (
        counts["TP"], counts["FP"],
        counts["FN"], counts["TN"]
    )

    precision = precision_eval.evaluate(predictions)
    recall = recall_eval.evaluate(predictions)
    f1 = f1_eval.evaluate(predictions)
    pr_auc = pr_eval.evaluate(predictions)

    specificity = tn / (tn + fp) if (tn + fp) else 0
    balanced_accuracy = (recall + specificity) / 2

    accuracy = (tp + tn) / (tp + tn + fp + fn)

    print("True Positives:", tp)
    print("False Positives:", fp)
    print("False Negatives:", fn)
    print("True Negatives:", tn)

    print("Accuracy:", round(accuracy, 4))
    print("Positive Precision:", round(precision, 4))
    print("Positive Recall:", round(recall, 4))
    print("Positive F1:", round(f1, 4))
    print("Specificity:", round(specificity, 4))
    print("Balanced Accuracy:", round(balanced_accuracy, 4))
    print("PR-AUC:", round(pr_auc, 4))

    results.append({
        "Model": name,
        "Accuracy": round(accuracy, 4),
        "Positive_Precision": round(precision, 4),
        "Positive_Recall": round(recall, 4),
        "Positive_F1": round(f1, 4),
        "Specificity": round(specificity, 4),
        "Balanced_Accuracy": round(balanced_accuracy, 4),
        "PR_AUC": round(pr_auc, 4)
    })

# Save report
output_file = "/home/shashank/xlearnnet_model_comparison_v2.csv"

with open(output_file, "w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

print("\n========== FINAL CORRECTED COMPARISON ==========")

for result in results:
    print(result)

print("\nReport saved:", output_file)

spark.stop()
