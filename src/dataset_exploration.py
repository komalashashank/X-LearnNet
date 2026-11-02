
import pandas as pd
import os

DATA_PATH = "data/raw"

files = [f for f in os.listdir(DATA_PATH) if f.endswith(".csv")]

print("X-LEARNNET DATASET EXPLORATION")
print("Total datasets:", len(files))

for file in sorted(files):

    path = os.path.join(DATA_PATH, file)
    df = pd.read_csv(path)

    print("\n" + "=" * 50)
    print("Dataset:", file)
    print("Rows:", df.shape[0])
    print("Columns:", df.shape[1])

    print("\nMissing values:")
    print(df.isnull().sum().sum())

    print("Duplicate rows:", df.duplicated().sum())

print("\nExploration completed!")
