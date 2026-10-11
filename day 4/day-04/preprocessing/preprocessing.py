
import pandas as pd
import os

# Get the dataset path
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
file_path = os.path.join(base_dir, "dataset", "student_data.csv")

# Load dataset
df = pd.read_csv(file_path)

print("Original Dataset:")
print(df)

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Remove duplicate rows
df = df.drop_duplicates()

# Remove rows with missing values
df = df.dropna()

print("\nCleaned Dataset:")
print(df)

# Save cleaned dataset
output_path = os.path.join(base_dir, "dataset", "cleaned_student_data.csv")
df.to_csv(output_path, index=False)

print("\nPreprocessing completed successfully!")