# Student-Performance-Predictor-End-to-End-ML-Project-with-Flask-AWS

An end-to-end machine learning project that predicts a student's **Math Score** based on demographic and academic inputs. Built with a modular ML pipeline, Flask web app, and cloud-ready deployment.

---

# How to run?

### STEPS:

Clone the repository

```bash
git clone https://github.com/Shu0209/Student-Performance-Predictor
```

### STEP 01 - Create a conda environment after opening the repository

```bash
conda create -p venv python=3.11 -y
```

```bash
conda activate venv/
```

### STEP 02 - Install the requirements

```bash
pip install -r requirements.txt
```

### STEP 03 - Train the model

```bash
# Run the training pipeline to generate artifacts (model.pkl + preprocessor.pkl)
python src/components/data_ingestion.py
```

This will:
- Ingest the dataset from `notebook/data/stud.csv`
- Apply preprocessing (imputation, encoding, scaling)
- Train and evaluate 8 regression models with hyperparameter tuning via GridSearchCV
- Save the best model and preprocessor to the `artifacts/` directory

### STEP 04 - Run the Flask app

```bash
python application.py
```

Now open:

```bash
http://localhost:5000
```

---

## Input Features

| Feature | Type | Description |
|---|---|---|
| Gender | Categorical | `male` / `female` |
| Race / Ethnicity | Categorical | Group A to Group E |
| Parental Education | Categorical | High school to Master's degree |
| Lunch Type | Categorical | `standard` / `free/reduced` |
| Test Prep Course | Categorical | `none` / `completed` |
| Reading Score | Numeric | 0–100 |
| Writing Score | Numeric | 0–100 |

**Output:** Predicted Math Score (0–100)

---

## Models Evaluated

The pipeline automatically selects the best model based on R² score on the test set:

- Linear Regression
- Decision Tree Regressor
- K-Nearest Neighbors Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost Regressor
- CatBoost Regressor
- AdaBoost Regressor

---

## Project Structure

```
├── src/
│   ├── components/
│   │   ├── data_ingestion.py        # Reads & splits dataset
│   │   ├── data_transformation.py   # Preprocessing pipeline
│   │   └── model_trainer.py         # GridSearchCV + model selection
│   ├── pipeline/
│   │   ├── predict_pipeline.py      # Inference logic + CustomData class
│   │   └── train_pipeline.py        # Training entry point
│   ├── exception.py                 # Custom exception handler
│   ├── logger.py                    # Logging configuration
│   └── utils.py                     # save/load object, evaluate models
├── templates/
│   ├── index.html                   # Landing page
│   └── home.html                    # Prediction form + result
├── artifacts/                       # Auto-generated: model.pkl, preprocessor.pkl
├── logs/                            # Auto-generated log files
├── notebook/
│   └── data/
│       └── stud.csv                 # Source dataset
├── application.py                   # Flask app entry point
├── setup.py                         # Package setup
└── requirements.txt
```

---

## Tech Stack Used

- **Python**
- **Scikit-learn** — preprocessing, model training, GridSearchCV
- **XGBoost / CatBoost** — gradient boosting regressors
- **Pandas / NumPy** — data handling
- **Flask** — web application
- **Dill** — model serialization
- **AWS EC2 + ECR** — cloud deployment
- **Docker** — containerization
- **GitHub Actions** — CI/CD pipeline

Saved URI - https://studentperformance-env.eba-fejy62qi.eu-north-1.elasticbeanstalk.com/predictdata