#END TO END MACHINE LEARNING PROJECT

📊 Student Performance Prediction (ML + Flask)

This project is an end-to-end Machine Learning pipeline deployed with a Flask web application.
It predicts student exam performance (math scores) based on demographic and educational factors such as gender, parental education, lunch type, test preparation, and reading/writing scores.

🚀 Features

Data Ingestion → Load raw CSV data and split into train/test sets.

Data Transformation → Apply preprocessing and feature engineering.

Model Training → Train multiple regression models with hyperparameter tuning.

Model Selection → Automatically selects and saves the best-performing model (model.pkl).

Web App → Flask UI where users can input data and get predictions instantly.

🛠️ Tech Stack

Python 3.8+

Pandas, NumPy, Scikit-learn

XGBoost, CatBoost, LightGBM

Flask (web framework)

Pickle (model persistence)
