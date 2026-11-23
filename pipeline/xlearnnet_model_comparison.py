from pyspark.sql import SparkSession
from pyspark.ml.classification import (
    RandomForestClassificationModel,
    GBTClassificationModel
)
from pyspark.ml.evaluation import (
    MulticlassClassificationEvaluator,
    BinaryClassificationEvaluator
)

spark = SparkSession.builder \
    .appName("XLearnNet_Model_Comparison") \
    .getOrCreate()

# Load the common test dataset
test_path = (
    "hdfs://localhost:9000/xlearnnet/feature_engineered/"
    "course_failure_flag/test"
)

test_df = spark.read.parquet(test_path)

print("Test records:", test_df.count())

# Load the three trained models
models = {
    "Random Forest": (
        RandomForestClassificationModel.load(
            "hdfs://localhost:9000/xlearnnet/models/"
            "course_failure_random_forest"
        )
    ),

    "Weighted Random Forest": (
        RandomForestClassificationModel.load(
            "hdfs://localhost:9000/xlearnnet/models/"
            "course_failure_random_forest_weighted"
        )
    ),

    "Gradient Boosted Trees": (
        GBTClassificationModel.load(
            "hdfs://localhost:9000/xlearnnet/models/"
            "course_failure_gbt"
        )
    )
}

# Evaluation tools
metrics = {
    "Accuracy": MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="accuracy"
    ),

    "Precision": MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="weightedPrecision"
    ),

    "Recall": MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="weightedRecall"
    ),

    "F1": MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="f1"
    ),

    "PR-AUC": BinaryClassificationEvaluator(
        labelCol="label",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderPR"
    )
}

results = []

for model_name, model in models.items():

    print("\nEvaluating:", model_name)

    predictions = model.transform(test_df)

    row = {"Model": model_name}

    for metric_name, evaluator in metrics.items():
        score = evaluator.evaluate(predictions)
        row[metric_name] = round(score, 4)
        print(metric_name, ":", round(score, 4))

    print("\nConfusion Matrix:")

    confusion = predictions.groupBy(
        "label", "prediction"
    ).count().orderBy("label", "prediction")

    confusion.show()

    results.append(row)

# Display final comparison
print("\n========== FINAL MODEL COMPARISON ==========")

for row in results:
    print(row)

# Save comparison report locally
import csv

output_file = "/home/shashank/xlearnnet_model_comparison.csv"

with open(output_file, "w", newline="") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=results[0].keys()
    )
    writer.writeheader()
    writer.writerows(results)

print("\nComparison report saved:", output_file)

spark.stop()
