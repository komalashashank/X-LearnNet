
import pandas as pd
import os

RAW = "data/processed/"
OUTPUT = "data/output/"

os.makedirs(OUTPUT, exist_ok=True)

print("X-LEARNNET DATA INTEGRATION")
print("=" * 55)

# 1. Student master
students = pd.read_csv(RAW + "students_master.csv")

print("Students:", len(students))

# 2. Course enrollment aggregation
enrollments = pd.read_csv(RAW + "course_enrollments.csv")

enrollment_summary = (
    enrollments.groupby("student_id")
    .agg(
        total_courses=("course_id", "nunique"),
        total_enrollments=("enrollment_id", "count")
    )
    .reset_index()
)

# 3. Assessment aggregation
assessments = pd.read_csv(RAW + "assessment_results.csv")

assessment_summary = (
    assessments.groupby(["student_id", "assessment_type"])
    ["score"].mean()
    .unstack()
    .reset_index()
)

assessment_summary.columns.name = None

# 4. Attendance aggregation
attendance = pd.read_csv(RAW + "attendance_records.csv")

attendance_summary = (
    attendance.groupby("student_id")
    .agg(
        total_classes=("attended_flag", "count"),
        classes_attended=("attended_flag", "sum")
    )
    .reset_index()
)

attendance_summary["calculated_attendance_rate"] = (
    attendance_summary["classes_attended"] /
    attendance_summary["total_classes"]
)

# 5. LMS activity aggregation
lms = pd.read_csv(RAW + "lms_activity_events.csv")

lms_summary = (
    lms.groupby("student_id")
    .agg(
        total_lms_events=("event_id", "count"),
        total_learning_time_seconds=("duration_seconds", "sum"),
        unique_resources=("resource_id", "nunique")
    )
    .reset_index()
)

# 6. Weekly learning behaviour
weekly = pd.read_csv(RAW + "weekly_learning_sequences.csv")

weekly_summary = (
    weekly.groupby("student_id")
    .agg(
        avg_weekly_logins=("lms_logins", "mean"),
        avg_resources_viewed=("resources_viewed", "mean"),
        avg_learning_hours=("learning_hours", "mean"),
        avg_weekly_attendance=("weekly_attendance_rate", "mean"),
        avg_quiz_attempts=("quiz_attempts", "mean"),
        avg_assignment_completion=(
            "assignment_completion_rate", "mean"
        )
    )
    .reset_index()
)

# 7. Integrate datasets
integrated = students.copy()

for dataset in [
    enrollment_summary,
    assessment_summary,
    attendance_summary,
    lms_summary,
    weekly_summary
]:
    integrated = integrated.merge(
        dataset,
        on="student_id",
        how="left",
        validate="one_to_one"
    )

# 8. Save unified repository
output_path = OUTPUT + "unified_student_repository.csv"

integrated.to_csv(output_path, index=False)

print("\nINTEGRATION RESULTS")
print("=" * 55)

print("Total students:", len(integrated))
print("Total columns:", len(integrated.columns))
print("Duplicate student IDs:",
      integrated["student_id"].duplicated().sum())

print("\nMissing values:")
print(integrated.isnull().sum().sum())

print("\nOutput saved:", output_path)

print("\nDATA INTEGRATION COMPLETED!")
