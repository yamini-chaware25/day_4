
import os
import pickle
import pandas as pd

# Get project path
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load trained model
model_path = os.path.join(base_dir, "models", "student_model.pkl")

with open(model_path, "rb") as file:
    model = pickle.load(file)

# Take input from user
hours = float(input("Enter study hours: "))

# Make prediction
data = pd.DataFrame({"Hours": [hours]})
predicted_marks = model.predict(data)

print("Study Hours:", hours)
print("Predicted Marks:", round(predicted_marks[0], 2))