
from pyspark.sql import SparkSession

# Initialize Spark
spark = SparkSession.builder \
    .appName("X-LearnNet") \
    .master("local[*]") \
    .getOrCreate()

print("X-LearnNet Spark Started!")
print("Spark Version:", spark.version)

# Load student dataset
students = spark.read.csv(
    "hdfs://localhost:9000/xlearnnet/raw/students_master.csv",
    header=True,
    inferSchema=True
)

# Display dataset
print("Total Students:", students.count())

students.show(5)

# Display schema
students.printSchema()

# Calculate average GPA
students.createOrReplaceTempView("students")

spark.sql("""
    SELECT AVG(GPA) AS average_gpa
    FROM students
""").show()

spark.stop()

print("Spark Processing Completed!")
