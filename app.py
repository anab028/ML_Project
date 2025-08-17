from flask import Flask, request, render_template
import os
import pandas as pd

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

# -------- CSV-driven dropdown choices --------
CSV_PATH = os.path.join(os.getcwd(), "stud.csv")  # put stud.csv in your project root

CHOICES = {
    "gender": [],
    "race_ethnicity": [],
    "parental_level_of_education": [],
    "lunch": [],
    "test_preparation_course": [],
}

def load_choices():
    if not os.path.exists(CSV_PATH):
        print(f"[WARN] CSV not found at {CSV_PATH}. Dropdowns will be empty until you add it.")
        return
    df = pd.read_csv(CSV_PATH)

    # expected columns (based on your file)
    expected = [
        "gender",
        "race_ethnicity",
        "parental_level_of_education",
        "lunch",
        "test_preparation_course",
        "reading_score",
        "writing_score",
        "math_score",  # target (not part of the form)
    ]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        print(f"[WARN] Missing columns in CSV: {missing}")
    # fill choices safely
    for col in ["gender", "race_ethnicity", "parental_level_of_education", "lunch", "test_preparation_course"]:
        if col in df.columns:
            CHOICES[col] = sorted(df[col].dropna().astype(str).unique().tolist())

# Load once at startup
load_choices()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predictdata", methods=["GET", "POST"])
def predict_datapoint():
    if request.method == "GET":
        return render_template("home.html", results=None, error=None, choices=CHOICES)

    try:
        # Collect & validate form fields
        gender = request.form.get("gender")
        race_ethnicity = request.form.get("race_ethnicity")
        parental_level_of_education = request.form.get("parental_level_of_education")
        lunch = request.form.get("lunch")
        test_preparation_course = request.form.get("test_preparation_course")

        reading_score = float(request.form.get("reading_score"))
        writing_score = float(request.form.get("writing_score"))

        if not (0 <= reading_score <= 100 and 0 <= writing_score <= 100):
            raise ValueError("Scores must be between 0 and 100.")

        data = CustomData(
            gender=gender,
            race_ethnicity=race_ethnicity,
            parental_level_of_education=parental_level_of_education,
            lunch=lunch,
            test_preparation_course=test_preparation_course,
            reading_score=reading_score,
            writing_score=writing_score,
        )

        pred_df = data.get_data_as_data_frame()
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)

        return render_template("home.html", results=float(results[0]), error=None, choices=CHOICES)

    except Exception as e:
        return render_template("home.html", results=None, error=str(e), choices=CHOICES)

if __name__ == "__main__":
    # macOS often reserves 5000; use 8080
    app.run(host="0.0.0.0", port=8080, debug=True)
