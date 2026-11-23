from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, lit
from pyspark.sql.functions import when, lit
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import (
    MulticlassClassificationEvaluator,
    BinaryClassificationEvaluator
)

spark = SparkSession.builder \
    .appName("XLearnNet_Weighted_RF") \
    .getOrCreate()

base = "hdfs://localhost:9000/xlearnnet/feature_engineered/course_failure_flag/"

# Load datasets
train = spark.read.parquet(base + "train")
validation = spark.read.parquet(base + "validation")

# Calculate class weights from training data only
counts = {
    row["label"]: row["count"]
    for row in train.groupBy("label").count().collect()
}

total = sum(counts.values())

weight_0 = total / (2.0 * counts[0.0])
weight_1 = total / (2.0 * counts[1.0])

print("Class weights:")
print("No failure:", weight_0)
print("Failure:", weight_1)

# Add weights
train = train.withColumn(
    "class_weight",
    when(col("label") == 1.0, lit(weight_1))
    .otherwise(lit(weight_0))
)

# Train weighted Random Forest
rf = RandomForestClassifier(
    featuresCol="features",
    labelCol="label",
    weightCol="class_weight",
    numTrees=150,
    maxDepth=10,
    seed=42
)

model = rf.fit(train)

print("Weighted Random Forest training completed!")

# Evaluate on validation data
predictions = model.transform(validation)

metrics = [
    "accuracy",
    "precisionByLabel",
    "recallByLabel",
    "fMeasureByLabel"
]

for metric in metrics:
    evaluator = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName=metric,
        metricLabel=1.0
    ) if metric != "accuracy" else MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName=metric
    )

    print(metric, ":", evaluator.evaluate(predictions))

# PR-AUC
pr_evaluator = BinaryClassificationEvaluator(
    labelCol="label",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderPR"
)

print("Validation PR-AUC:", pr_evaluator.evaluate(predictions))

# Confusion matrix
print("VALIDATION CONFUSION MATRIX")
predictions.groupBy("label", "prediction") \
    .count().orderBy("label", "prediction").show()

# Save improved model separately
model.write().overwrite().save(
    "hdfs://localhost:9000/xlearnnet/models/course_failure_random_forest_weighted"
)

print("Weighted model saved successfully!")

spark.stop()
