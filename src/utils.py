import os
import sys
import pickle
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException


def save_object(file_path, obj):
    """
    Save a Python object to a pickle file.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    """
    Train and evaluate multiple models with hyperparameter tuning.
    Returns a dict: model_name -> best test R² score.
    """
    try:
        report = {}

        for model_name, model in models.items():
            para = param.get(model_name, {})

            gs = GridSearchCV(model, para, cv=3, scoring="r2", n_jobs=-1, verbose=0)
            gs.fit(X_train, y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            y_test_pred = model.predict(X_test)
            test_score = r2_score(y_test, y_test_pred)

            report[model_name] = test_score

        return report

    except Exception as e:
        raise CustomException(e, sys)
