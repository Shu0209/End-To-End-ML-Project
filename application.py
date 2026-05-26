from flask import Flask, request, render_template, redirect, url_for
import numpy as np
import pandas as pd
import logging

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

# Add this — sends Flask errors to gunicorn log
gunicorn_logger = logging.getLogger('gunicorn.error')
application.logger.handlers = gunicorn_logger.handlers
application.logger.setLevel(gunicorn_logger.level)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        try:
            data = CustomData(
                gender=request.form.get('gender'),
                race_ethnicity=request.form.get('race_ethnicity'),
                parental_level_of_education=request.form.get('parental_level_of_education'),
                lunch=request.form.get('lunch'),
                test_preparation_course=request.form.get('test_preparation_course'),
                reading_score=int(request.form.get('reading_score')),
                writing_score=int(request.form.get('writing_score'))
            )
            pred_df = data.get_data_as_data_frame()
            predict_pipeline = PredictPipeline()
            results = predict_pipeline.predict(pred_df)
            return render_template('home.html', results=round(float(results[0]), 2))

        except Exception as e:
            application.logger.error(f"Prediction error: {str(e)}", exc_info=True)
            raise e

if __name__ == "__main__":
    application.run(host="0.0.0.0", debug=False)