from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from evaluate import elastic_net  # Import your model training function

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input values from the form
    features = [float(request.form['feature1']),
                float(request.form['feature2']),
                float(request.form['feature3']),
                float(request.form['feature4'])]  # Add more as needed

    # Convert features to a numpy array
    features_array = np.array(features).reshape(1, -1)

    # Load weights and bias (if saved in a file)
    weights = np.load('weights.npy')
    bias = np.load('bias.npy')

    # Make prediction
    prediction = np.dot(features_array, weights) + bias

    return render_template('index.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
