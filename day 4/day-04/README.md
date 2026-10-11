# Day-04: Machine Learning Prediction Model

## Objective

Understand core Machine Learning concepts and build a basic prediction model using Python.

## Project Description

This project predicts student marks based on the number of hours studied using Linear Regression.

## Technologies Used

* Python
* Pandas
* Scikit-learn
* Pickle

## Folder Structure

```text
day-04/
├── dataset/
├── preprocessing/
├── models/
├── evaluation/
├── predictions/
└── README.md
```

## Tasks Completed

* Created and loaded a student dataset.
* Checked missing values and removed duplicates.
* Preprocessed the dataset.
* Trained a Linear Regression model.
* Evaluated model performance using MAE, MSE, and R² score.
* Predicted marks for new study hours.

## How to Run

Install the required libraries:

```bash
pip install pandas scikit-learn
```

Run the scripts from the `day-04` directory:

```bash
python preprocessing/preprocessing.py
python models/train_model.py
python evaluation/evaluate_model.py
python predictions/predict.py
```

## Conclusion

Successfully built a basic Machine Learning model to predict student marks using Linear Regression.
