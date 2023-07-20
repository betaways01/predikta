from flask import Flask, render_template, request
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import datetime
import re
import os
import numpy as np
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier #type: ignore
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)

# Initialize variables
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model.pkl')
scaler_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scaler.pkl')

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)


prediction = None
advice = ""
prev_predictions = []
model_performance = {}

@app.route('/train', methods=['GET', 'POST'])
def train():
    global model, scaler, advice, model_performance

    input_data = request.form.get('new_data') # type: ignore
    if input_data:
        if re.match("^[0-9., ]*$", input_data) is None:
            advice = "Please enter numbers only, separated by commas or spaces."
        else:
            try:
                input_data = [float(x) for x in input_data.replace(" ", "").split(',') if x]
            except ValueError:
                advice = "Invalid data format. Ensure your input is in the correct format."
            else:
                # Load the existing data
                data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'multipliers.csv')
                data = pd.read_csv(data_path)

                # Append new data
                new_data = pd.DataFrame(input_data, columns=['Multiplier'])
                data = pd.concat([data, new_data], ignore_index=True)

                # Preprocess the data
                data['x2_or_wait'] = (data['Multiplier'] >= 2).astype(int)
                for i in range(1, 4):
                    data[f'lag_{i}'] = data['Multiplier'].shift(i)
                data = data.dropna()

                # Split the data into features (X) and target variable (y)
                X = data.drop(['x2_or_wait', 'Multiplier'], axis=1)
                y = data['x2_or_wait']

                # Balance the classes
                smote = SMOTE(random_state=42)
                X_smote, y_smote = smote.fit_resample(X, y) # type: ignore

                # Standardize the features
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X_smote) # type: ignore

                # Split the data into training and test sets
                X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_smote, test_size=0.2, random_state=42)

                # Train a LightGBM model
                model = LGBMClassifier(n_jobs=1)
                model.fit(X_train, y_train)

                # Calculate model performance
                y_pred = model.predict(X_test)
                report = classification_report(y_test, y_pred, output_dict=True)
                model_performance = {
                    'Accuracy': report['accuracy'], # type: ignore
                    'Precision': report['1']['precision'], # type: ignore
                    'Recall': report['1']['recall'], # type: ignore
                    'F1-score': report['1']['f1-score'], # type: ignore
                    'Confusion Matrix': confusion_matrix(y_test, y_pred).tolist(),
                }

                # Save the model and the scaler to disk
                joblib.dump(model, 'model.pkl')
                joblib.dump(scaler, 'scaler.pkl')

                advice = "Model training successful!"

    return render_template('train.html', advice=advice, stats=model_performance)

@app.route('/', methods=['GET', 'POST'])
def index():
    global model, prediction, stats, prev_predictions, advice, model_performance

    advice=""
    prediction = None

    if request.method == 'POST': # type: ignore
        input_data = request.form.get('last_game_results') # type: ignore
        if input_data:
            # Validate the input to ensure it contains only numbers, comma, and space
            if re.match("^[0-9., ]*$", input_data) is None:
                advice = "Please enter numbers only, separated by commas or spaces."
            else:
                # Convert string input into a list of floats
                try:
                    input_data = [float(x) for x in input_data.replace(" ", "").split(',') if x]
                except ValueError:
                    advice = "Invalid data format. Ensure your input is in the correct format."
                else:
                    # If not enough data, return a warning message
                    if len(input_data) < 3:
                        advice = "Input at least 3 values for the prediction."
                    else:
                        # Scale the data and make a prediction
                        input_data_scaled = scaler.transform([input_data])
                        prediction = model.predict(input_data_scaled)[0]
                        current_prediction = (prediction, datetime.datetime.now())
                        prev_predictions.append(current_prediction)

                        # Generate advice
                        if prediction == 1:
                            advice = "Considering your recent game results, the next prediction is likely to be Greater than 2 (BET NOW)"
                        else:
                            advice = "Considering your recent game results, the next prediction is likely to be Less than 2 (WAIT)"

    return render_template('index.html', advice=advice, prediction=prediction, prev_predictions=prev_predictions, model_performance=model_performance)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
