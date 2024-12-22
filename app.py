from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import os

app = Flask(__name__)

model_path = 'lung_c.pkl'

# Check if the model file exists and log details
if not os.path.isfile(model_path):
    print(f"Model file {model_path} not found.")
    model = None
else:
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        model = None

def preprocess_input(data):
    # Convert categorical inputs to binary and ensure all inputs are numeric
    binary_data = {
        'Gender': 1 if data['Gender'].lower() == 'male' else 0,
        'Age': int(data['Age']),
        'Smoking': 1 if data['Smoking'].lower() == 'yes' else 0,
        'YellowFingers': 1 if data['YellowFingers'].lower() == 'yes' else 0,
        'Anxiety': 1 if data['Anxiety'].lower() == 'yes' else 0,
        'PeerPressure': 1 if data['PeerPressure'].lower() == 'yes' else 0,
        'ChronicDisease': 1 if data['ChronicDisease'].lower() == 'yes' else 0,
        'Fatigue': 1 if data['Fatigue'].lower() == 'yes' else 0,
        'Allergy': 1 if data['Allergy'].lower() == 'yes' else 0,
        'Wheezing': 1 if data['Wheezing'].lower() == 'yes' else 0,
        'AlcoholConsuming': 1 if data['AlcoholConsuming'].lower() == 'yes' else 0,
        'Cough': 1 if data['Coughing'].lower() == 'yes' else 0,
        'ShortnessOfBreath': 1 if data['ShortnessOfBreath'].lower() == 'yes' else 0,
        'SwallowingDifficulty': 1 if data['SwallowingDifficulty'].lower() == 'yes' else 0,
        'ChestPain': 1 if data['ChestPain'].lower() == 'yes' else 0
    }
    return np.array([list(binary_data.values())], dtype=float)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded properly'}), 500

    try:
        data = request.form.to_dict()
        print("Received data:", data)
        processed_data = preprocess_input(data)
        print("Processed data:", processed_data)
        prediction = model.predict(processed_data)
        
        # Determine result and CSS class based on prediction
        if prediction[0] == 1:
            result = 'You have lung cancer'
            result_class = 'result-positive'
        else:
            result = 'You do not have lung cancer'
            result_class = 'result-negative'
        
        return render_template('result.html', result=result, result_class=result_class)
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
