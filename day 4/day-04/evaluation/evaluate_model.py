
import pandas as pd
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Get project path
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load dataset
data_path = os.path.join(base_dir, "dataset", "cleaned_student_data.csv")
df = pd.read_csv(data_path)

# Split features and target
X = df[["Hours"]]
y = df["Marks"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Load trained model
model_path = os.path.join(base_dir, "models", "student_model.pkl")

with open(model_path, "rb") as file:
    model = pickle.load(file)

# Predict test data
predictions = model.predict(X_test)

# Evaluate model
print("Actual Marks:", list(y_test))
print("Predicted Marks:", predictions.round(2))
print("Mean Absolute Error:", round(mean_absolute_error(y_test, predictions), 2))
print("Mean Squared Error:", round(mean_squared_error(y_test, predictions), 2))
print("R2 Score:", round(r2_score(y_test, predictions), 2))