from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    StringIndexer, OneHotEncoder,
    VectorAssembler, StandardScaler
)

spark = (
    SparkSession.builder
    .appName("XLearnNet-Feature-Mapping")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("ERROR")

target = "course_failure_flag"

numeric_cols = [
    "age", "GPA", "course_load", "avg_course_grade",
    "attendance_rate", "lms_logins_past_month",
    "avg_session_duration_minutes",
    "assignment_submission_rate",
    "forum_participation_count", "video_completion_rate",
    "cohort_year", "prior_gpa",
    "average_assessment_score", "assessment_count",
    "total_classes", "attendance_percentage",
    "total_enrollments", "total_lms_events",
    "average_activity_duration", "average_weekly_study_hours"
]

categorical_cols = ["gender", "major", "semester", "program_level"]

path = f"hdfs://localhost:9000/xlearnnet/model_ready/{target}"

df = (
    spark.read.option("header", True)
    .option("inferSchema", True)
    .csv(path)
)

train_df = df.filter(col("data_split") == "train")

indexers = [
    StringIndexer(
        inputCol=c,
        outputCol=c + "_index",
        handleInvalid="keep"
    )
    for c in categorical_cols
]

encoder = OneHotEncoder(
    inputCols=[c + "_index" for c in categorical_cols],
    outputCols=[c + "_encoded" for c in categorical_cols],
    handleInvalid="keep"
)

assembler = VectorAssembler(
    inputCols=numeric_cols +
              [c + "_encoded" for c in categorical_cols],
    outputCol="raw_features",
    handleInvalid="keep"
)

scaler = StandardScaler(
    inputCol="raw_features",
    outputCol="features",
    withStd=True,
    withMean=True
)

pipeline = Pipeline(
    stages=indexers + [encoder, assembler, scaler]
)

print("Fitting the original preprocessing pipeline...")
model = pipeline.fit(train_df)

for stage in model.stages:
    if hasattr(stage, "labels"):
        print(f"\n{stage.getInputCol()} category order:")
        for i, label in enumerate(stage.labels):
            print(i, label)

    if hasattr(stage, "categorySizes"):
        print("\nOneHotEncoder category sizes:")
        print(stage.categorySizes)
        print("Drop last:", stage.getDropLast())

transformed = model.transform(train_df.limit(5))

print("\nRAW FEATURE VECTOR METADATA:")
print(transformed.schema["raw_features"].metadata)

print("\nSCALED FEATURE VECTOR METADATA:")
print(transformed.schema["features"].metadata)

print("\nFeature dimensions:")
print(len(transformed.first()["features"]))

spark.stop()
