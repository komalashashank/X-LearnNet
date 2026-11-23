from pyspark.sql import SparkSession
from pyspark.ml.classification import GBTClassificationModel

spark = SparkSession.builder \
    .appName("XLearnNet_GBT_Feature_Importance") \
    .getOrCreate()

model_path = "hdfs://localhost:9000/xlearnnet/models/course_failure_gbt"
data_path = "hdfs://localhost:9000/xlearnnet/feature_engineered/course_failure_flag/train"

print("Loading saved GBT model...")
model = GBTClassificationModel.load(model_path)

print("Loading training data metadata...")
df = spark.read.parquet(data_path)

importances = model.featureImportances
metadata = df.schema["features"].metadata.get("ml_attr", {})
attributes = metadata.get("attrs", {})

feature_names = {}
for group in attributes.values():
    for item in group:
        feature_names[item["idx"]] = item["name"]

results = []
for index, score in enumerate(importances):
    name = feature_names.get(index, f"feature_{index}")
    results.append((name, float(score)))

results.sort(key=lambda x: x[1], reverse=True)

print("\n===== GBT FEATURE IMPORTANCE =====")
for rank, (name, score) in enumerate(results, 1):
    print(f"{rank:2}. {name:40} {score:.6f}")

spark.stop()
