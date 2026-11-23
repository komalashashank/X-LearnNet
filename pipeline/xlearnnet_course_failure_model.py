from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import (
    MulticlassClassificationEvaluator,
    BinaryClassificationEvaluator
)

spark = SparkSession.builder \
    .appName("XLearnNet_Course_Failure_Model") \
    .getOrCreate()

base = "hdfs://localhost:9000/xlearnnet/feature_engineered/course_failure_flag/"

# Load datasets
train = spark.read.parquet(base + "train")
validation = spark.read.parquet(base + "validation")
test = spark.read.parquet(base + "test")

print("Training records:", train.count())
print("Validation records:", validation.count())
print("Testing records:", test.count())

# Initialize Random Forest
rf = RandomForestClassifier(
    featuresCol="features",
    labelCol="label",
    numTrees=100,
    maxDepth=10,
    seed=42
)

# Train model
model = rf.fit(train)

print("Random Forest training completed!")

# Validation predictions
val_predictions = model.transform(validation)

# Evaluation
for metric in ["accuracy", "f1", "weightedPrecision", "weightedRecall"]:
    evaluator = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName=metric
    )
    print(metric, ":", evaluator.evaluate(val_predictions))

# ROC-AUC
auc_evaluator = BinaryClassificationEvaluator(
    labelCol="label",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC"
)

print("Validation ROC-AUC:",
      auc_evaluator.evaluate(val_predictions))

# Test evaluation
test_predictions = model.transform(test)

print("\nTEST CONFUSION MATRIX")
test_predictions.groupBy("label", "prediction") \
    .count().orderBy("label", "prediction").show()

# Save model
model.write().overwrite().save(
    "hdfs://localhost:9000/xlearnnet/models/course_failure_random_forest"
)

print("Model saved successfully!")

spark.stop()

