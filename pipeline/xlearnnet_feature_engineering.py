
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    StringIndexer,
    OneHotEncoder,
    VectorAssembler,
    StandardScaler
)

spark = SparkSession.builder \
    .appName("XLearnNet_Feature_Engineering") \
    .getOrCreate()

# Input features
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

categorical_cols = [
    "gender", "major", "semester", "program_level"
]

targets = [
    "course_failure_flag",
    "dropout_flag",
    "next_semester_GPA"
]

# Process each prediction task separately
for target in targets:

    print("\nProcessing:", target)

    input_path = (
        "hdfs://localhost:9000/xlearnnet/model_ready/" + target
    )

    df = spark.read.option("header", True) \
        .option("inferSchema", True).csv(input_path)

    # Separate training, validation and test records
    train_df = df.filter(col("data_split") == "train")
    validation_df = df.filter(col("data_split") == "validation")
    test_df = df.filter(col("data_split") == "test")

    # Convert target to numeric label
    train_df = train_df.withColumn(target, col(target).cast("double"))
    validation_df = validation_df.withColumn(target, col(target).cast("double"))
    test_df = test_df.withColumn(target, col(target).cast("double"))

    # Fit preprocessing only on training data
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

    pipeline_model = pipeline.fit(train_df)

    # Transform all three splits using the training-fitted pipeline
    for split_name, split_df in [
        ("train", train_df),
        ("validation", validation_df),
        ("test", test_df)
    ]:

        transformed = pipeline_model.transform(split_df)

        final_df = transformed.select(
            "student_id",
            col(target).alias("label"),
            "features"
        )

        output_path = (
            "hdfs://localhost:9000/xlearnnet/feature_engineered/"
            + target + "/" + split_name
        )

        final_df.write.mode("overwrite").parquet(output_path)

        print(
            split_name,
            "records:",
            final_df.count(),
            "saved to:",
            output_path
        )

    print("Completed:", target)

print("\nX-LearnNet feature engineering completed!")

spark.stop()
