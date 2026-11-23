from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.ml.evaluation import (
    MulticlassClassificationEvaluator,
    BinaryClassificationEvaluator
)

spark = SparkSession.builder \
    .appName("XLearnNet_Failure_Evaluation") \
    .getOrCreate()

base = "hdfs://localhost:9000/xlearnnet/feature_engineered/course_failure_flag/"

model = RandomForestClassificationModel.load(
    "hdfs://localhost:9000/xlearnnet/models/course_failure_random_forest"
)

for name in ["train", "validation", "test"]:

    df = spark.read.parquet(base + name)

    print("\n", name.upper(), "CLASS DISTRIBUTION")
    df.groupBy("label").count().show()

    if name == "test":
        predictions = model.transform(df)

        print("TEST POSITIVE CLASS METRICS")

        for metric in [
            "precisionByLabel",
            "recallByLabel",
            "fMeasureByLabel"
        ]:
            evaluator = MulticlassClassificationEvaluator(
                labelCol="label",
                predictionCol="prediction",
                metricName=metric,
                metricLabel=1.0
            )

            print(metric, ":", evaluator.evaluate(predictions))

        pr = BinaryClassificationEvaluator(
            labelCol="label",
            rawPredictionCol="rawPrediction",
            metricName="areaUnderPR"
        )

        print("PR-AUC:", pr.evaluate(predictions))

spark.stop()
