
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, lit, sum as spark_sum
from pyspark.ml.functions import vector_to_array
from pyspark.ml.classification import GBTClassificationModel

spark = SparkSession.builder \
    .appName("XLearnNet_GBT_Threshold") \
    .getOrCreate()

model_path = (
    "hdfs://localhost:9000/xlearnnet/models/course_failure_gbt"
)

data_path = (
    "hdfs://localhost:9000/xlearnnet/feature_engineered/"
    "course_failure_flag/validation"
)

model = GBTClassificationModel.load(model_path)
validation = spark.read.parquet(data_path)

predictions = model.transform(validation)

results = predictions.select(
    col("label"),
    vector_to_array("probability")[1].alias("probability")
).cache()

thresholds = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

print("\nGBT THRESHOLD COMPARISON")

for threshold in thresholds:

    evaluated = results.withColumn(
        "prediction",
        when(col("probability") >= lit(threshold), 1.0)
        .otherwise(0.0)
    )

    counts = evaluated.agg(
        spark_sum(when(
            (col("label") == 1) & (col("prediction") == 1), 1
        ).otherwise(0)).alias("TP"),

        spark_sum(when(
            (col("label") == 0) & (col("prediction") == 1), 1
        ).otherwise(0)).alias("FP"),

        spark_sum(when(
            (col("label") == 1) & (col("prediction") == 0), 1
        ).otherwise(0)).alias("FN")
    ).first()

    tp, fp, fn = counts["TP"], counts["FP"], counts["FN"]

    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0
    )

    print(
        f"Threshold: {threshold:.2f} | "
        f"Precision: {precision:.4f} | "
        f"Recall: {recall:.4f} | "
        f"F1: {f1:.4f}"
    )

results.unpersist()
spark.stop()
