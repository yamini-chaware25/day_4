
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

# Get project path
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load cleaned dataset
file_path = os.path.join(
    base_dir, "dataset", "cleaned_student_data.csv"
)
df = pd.read_csv(file_path)

# Features and target
X = df[["Hours"]]
y = df["Marks"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Save model
model_path = os.path.join(base_dir, "models", "student_model.pkl")

with open(model_path, "wb") as file:
    pickle.dump(model, file)

print("Model trained successfully!")
print("Model saved at:", model_path)