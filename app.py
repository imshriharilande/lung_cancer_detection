from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io

app = Flask(__name__)
CORS(app)

# Load the trained model
model = load_model(r'F:\GOOGLE_HEALTH\Advancing Breast Cancer Detection with AI - Google Health_files\capstone\lung_cancer_detection_model.h5')

# Preprocess the image
def preprocess_image(image):
    img = Image.open(io.BytesIO(image))
    img = img.resize((256, 256))
    img = np.array(img)
    img = img / 255.0  # Normalize the image
    img = np.expand_dims(img, axis=0)
    return img

# Home route to serve the HTML page
@app.route('/')
def home():
    return render_template('index.html')  # Update to render index.html

# Define a route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    try:
        img = file.read()
        img = preprocess_image(img)
        prediction = model.predict(img)
        predicted_class = np.argmax(prediction, axis=1)[0]
        response = {'prediction': int(predicted_class)}
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
