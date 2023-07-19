from flask import Flask, render_template, request
#import cloudpickle as joblib
import joblib
import datetime
import re
import os
import numpy as np
import json
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)

# Initialize variables

model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model.pkl')
scaler_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scaler_params.json')

model = joblib.load(model_path)

# Load the mean and variance for the features
with open(scaler_path) as f:
    params = json.load(f)
mean_ = np.array([13.815946002446857, 13.655341472281862, 14.553975745300855])
var_ = np.array([1375793.8500529677, 1294901.4013289963, 1768316.874358202])


prediction = None
advice = ""
prev_predictions = []

def custom_scaler(input_data):
    return (input_data - mean_) / np.sqrt(var_)
    

@app.route('/', methods=['GET', 'POST'])
def index():
    global model, prediction, stats, prev_predictions, advice

    advice=""
    prediction = None

    if request.method == 'POST':
        input_data = request.form.get('last_game_results')
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
                        input_data_scaled = custom_scaler(input_data)
                        prediction = model.predict(input_data_scaled.reshape(1,-1))[0]
                        current_prediction = (prediction, datetime.datetime.now())
                        prev_predictions.append(current_prediction)

                        # Generate advice
                        if prediction == 1:
                            advice = "Considering your recent game results, the next prediction is likely to be Greater than 2 (BET NOW)"
                        else:
                            advice = "Considering your recent game results, the next prediction is likely to be Less than 2 (WAIT)"

    return render_template('index.html', advice=advice, prediction=prediction, prev_predictions=prev_predictions)

if __name__ == '__main__':
    app.run(debug=True)
