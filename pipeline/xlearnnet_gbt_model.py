
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, lit
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator
)

spark = SparkSession.builder \
    .appName("XLearnNet_GBT_Training") \
    .getOrCreate()

# HDFS paths
base_path = "hdfs://localhost:9000/xlearnnet/feature_engineered/course_failure_flag"

train_path = base_path + "/train"
validation_path = base_path + "/validation"

model_path = (
    "hdfs://localhost:9000/xlearnnet/models/"
    "course_failure_gbt"
)

# Load datasets
train_df = spark.read.parquet(train_path)
validation_df = spark.read.parquet(validation_path)

print("Training records:", train_df.count())
print("Validation records:", validation_df.count())

# Apply class weights
# Based on the training class distribution
train_df = train_df.withColumn(
    "class_weight",
    when(col("label") == 1.0, lit(2.631294293174189))
    .otherwise(lit(0.617299614980749))
)

validation_df = validation_df.withColumn(
    "class_weight",
    when(col("label") == 1.0, lit(2.631294293174189))
    .otherwise(lit(0.617299614980749))
)

# Configure GBT classifier
gbt = GBTClassifier(
    labelCol="label",
    featuresCol="features",
    weightCol="class_weight",
    maxIter=50,
    maxDepth=5,
    stepSize=0.05,
    seed=42
)

# Train model
print("\nStarting GBT training...")

gbt_model = gbt.fit(train_df)

print("GBT training completed!")

# Validation predictions
predictions = gbt_model.transform(validation_df)

# Evaluate using default threshold
accuracy_evaluator = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="accuracy"
)

f1_evaluator = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="f1"
)

auc_evaluator = BinaryClassificationEvaluator(
    labelCol="label",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderPR"
)

print("\nGBT VALIDATION RESULTS")
print("Accuracy:", accuracy_evaluator.evaluate(predictions))
print("F1-score:", f1_evaluator.evaluate(predictions))
print("PR-AUC:", auc_evaluator.evaluate(predictions))

# Confusion matrix
print("\nGBT CONFUSION MATRIX")

predictions.groupBy(
    "label", "prediction"
).count().orderBy(
    "label", "prediction"
).show()

# Save model
gbt_model.write().overwrite().save(model_path)

print("\nGBT model saved successfully!")
print("Model location:", model_path)

spark.stop()
