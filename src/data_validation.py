
import pandas as pd

PATH = "data/raw/"

# Load datasets
students = pd.read_csv(PATH + "students_master.csv")
enrollments = pd.read_csv(PATH + "course_enrollments.csv")
courses = pd.read_csv(PATH + "course_catalog.csv")
faculty = pd.read_csv(PATH + "faculty.csv")
assessments = pd.read_csv(PATH + "assessment_results.csv")
attendance = pd.read_csv(PATH + "attendance_records.csv")
edges = pd.read_csv(PATH + "knowledge_graph_edges.csv")

print("=" * 55)
print("X-LEARNNET DATA QUALITY VALIDATION")
print("=" * 55)

# 1. Student ID validation
print("\n1. Student ID Validation")

print("Unique students:", students["student_id"].nunique())
print("Duplicate student IDs:", students["student_id"].duplicated().sum())

# 2. Course relationship validation
print("\n2. Course Validation")

invalid_courses = ~enrollments["course_id"].isin(courses["course_id"])

print("Invalid course references:", invalid_courses.sum())

# 3. Faculty relationship validation
print("\n3. Faculty Validation")

invalid_faculty = ~enrollments["faculty_id"].isin(faculty["faculty_id"])

print("Invalid faculty references:", invalid_faculty.sum())

# 4. Student enrollment validation
print("\n4. Enrollment Validation")

invalid_students = ~enrollments["student_id"].isin(students["student_id"])

print("Invalid student references:", invalid_students.sum())

# 5. Assessment validation
print("\n5. Assessment Validation")

invalid_assessments = ~assessments["enrollment_id"].isin(
    enrollments["enrollment_id"]
)

print("Invalid assessment enrollment IDs:", invalid_assessments.sum())

# 6. Attendance validation
print("\n6. Attendance Validation")

invalid_attendance = ~attendance["enrollment_id"].isin(
    enrollments["enrollment_id"]
)

print("Invalid attendance enrollment IDs:", invalid_attendance.sum())

# 7. Graph duplicate validation
print("\n7. Knowledge Graph Validation")

print("Total edges:", len(edges))
print("Duplicate edges:", edges.duplicated().sum())

# 8. Assessment score validation
print("\n8. Score Validation")

print("Scores below zero:", (assessments["score"] < 0).sum())
print("Scores above 100:", (assessments["score"] > 100).sum())

print("\nVALIDATION COMPLETED")
