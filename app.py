from flask import Flask, render_template, request
import pickle
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load trained model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input values from form
    feature1 = float(request.form['feature1'])
    feature2 = float(request.form['feature2'])
    feature3 = float(request.form['feature3'])

    # Convert inputs to array
    input_data = np.array([[feature1, feature2, feature3]])

    # Predict
    prediction = model.predict(input_data)

    # Example range (±10%)
    lower = prediction[0] * 0.9
    upper = prediction[0] * 1.1

    return render_template(
        'index.html',
        prediction_text=f"Estimated Output Range: {lower:.2f} – {upper:.2f}"
    )

if __name__ == '__main__':
    app.run(debug=True)
