import pickle
import sys
from flask import Flask, request, render_template
import os

# Add the correct path for the pipeline module
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from pipeline.predict_pipeline import CustomData, PredictPipeline
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app = application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=["POST"])
def predict_datapoint():
    # Assuming POST because GET just reloads index.html as per your previous design
    gender = request.form.get('gender')
    race_ethnicity = request.form.get('ethnicity')
    parental_level_of_education = request.form.get('parental_level_of_education')
    lunch = request.form.get('lunch')
    test_preparation_course = request.form.get('test_preparation_course')
    reading_score = float(request.form.get('reading_score')) # type: ignore
    writing_score = float(request.form.get('writing_score')) # type: ignore

    # Create an instance of CustomData
    data = CustomData(
        gender=gender, # type: ignore
        race_ethnicity=race_ethnicity, # type: ignore
        parental_level_of_education=parental_level_of_education, # type: ignore
        lunch=lunch, # type: ignore
        test_preparation_course=test_preparation_course, # type: ignore
        reading_score=reading_score, # type: ignore
        writing_score=writing_score # type: ignore
    )

    # Convert data to DataFrame suitable for prediction
    pred_df = data.get_data_as_data_frame()
    print(pred_df)  # This prints to console, good for debugging

    # Make prediction using a pipeline
    predict_pipeline = PredictPipeline()
    result = predict_pipeline.predict(pred_df)

    # Render result.html with the prediction result
    return render_template('result.html', result=result[0])

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)
