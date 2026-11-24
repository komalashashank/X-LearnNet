import pandas as pd
import os

RAW = "data/raw/"
PROCESSED = "data/processed/"

os.makedirs(PROCESSED, exist_ok=True)

def clean_dataset(filename, key=None):

    print("\nProcessing:", filename)

    df = pd.read_csv(RAW + filename)

    # Standardize column names
    df.columns = df.columns.str.strip().str.lower()

    # Remove whitespace from text columns
    for col in df.columns:
        if pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].str.strip()

    # Remove exact duplicate rows
    before = len(df)
    df = df.drop_duplicates()
    removed = before - len(df)

    print("Original rows:", before)
    print("Duplicates removed:", removed)
    print("Final rows:", len(df))

    # Save processed dataset
    output = PROCESSED + filename
    df.to_csv(output, index=False)

    return df


# Clean important datasets
students = clean_dataset("students_master.csv")
assessments = clean_dataset("assessment_results.csv")
attendance = clean_dataset("attendance_records.csv")
enrollments = clean_dataset("course_enrollments.csv")
lms = clean_dataset("lms_activity_events.csv")
weekly = clean_dataset("weekly_learning_sequences.csv")

print("\nDATA CLEANING COMPLETED")
