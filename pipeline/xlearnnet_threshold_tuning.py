from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, sum as spark_sum, lit
from pyspark.ml.functions import vector_to_array
from pyspark.ml.classification import (
    RandomForestClassificationModel,
    GBTClassificationModel
)
import pandas as pd

spark = SparkSession.builder \
    .appName("XLearnNet_Threshold_Comparison") \
    .getOrCreate()

data_path = (
    "hdfs://localhost:9000/xlearnnet/feature_engineered/"
    "course_failure_flag/validation"
)

validation = spark.read.parquet(data_path).cache()
print("Validation records:", validation.count())

models = {
    "Random Forest": RandomForestClassificationModel.load(
        "hdfs://localhost:9000/xlearnnet/models/course_failure_random_forest"
    ),
    "Weighted Random Forest": RandomForestClassificationModel.load(
        "hdfs://localhost:9000/xlearnnet/models/course_failure_random_forest_weighted"
    ),
    "GBT": GBTClassificationModel.load(
        "hdfs://localhost:9000/xlearnnet/models/course_failure_gbt"
    )
}

thresholds = [
    0.10, 0.15, 0.20, 0.25, 0.30,
    0.35, 0.40, 0.45, 0.50, 0.55, 0.60
]

all_results = []

for name, model in models.items():
    print(f"\n========== {name} ==========")

    predictions = model.transform(validation)

    results = predictions.select(
        col("label"),
        vector_to_array("probability")[1].alias("failure_probability")
    ).cache()

    model_results = []

    for threshold in thresholds:
        evaluated = results.withColumn(
            "prediction",
            when(col("failure_probability") >= lit(threshold), 1)
            .otherwise(0)
        )

        counts = evaluated.agg(
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
            ).alias("FN")
        ).first()

        tp, fp, fn = counts["TP"], counts["FP"], counts["FN"]

        precision = tp / (tp + fp) if tp + fp else 0
        recall = tp / (tp + fn) if tp + fn else 0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0
        )

        row = {
            "Model": name,
            "Threshold": threshold,
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "F1": round(f1, 4)
        }

        model_results.append(row)
        all_results.append(row)

        print(
            f"Threshold: {threshold:.2f} | "
            f"Precision: {precision:.4f} | "
            f"Recall: {recall:.4f} | F1: {f1:.4f}"
        )

    best = max(model_results, key=lambda x: x["F1"])

    print("\nBest validation threshold:", best["Threshold"])
    print("Best validation F1:", best["F1"])

    results.unpersist()

pd.DataFrame(all_results).to_csv(
    "outputs/threshold_comparison_validation.csv",
    index=False
)

print("\nResults saved to outputs/threshold_comparison_validation.csv")

validation.unpersist()
spark.stop()
