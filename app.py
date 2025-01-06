from flask import Flask, request, jsonify
from keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Load the pre-trained model
model = load_model('model.h5')


# Define a route to classify an uploaded image
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    # Load and preprocess the image
    img = image.load_img(file, target_size=(224, 224))  # Update size based on your model
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # Get the prediction
    prediction = model.predict(img_array)
    predicted_class = 'dog' if prediction[0] > 0.5 else 'cat'  # Adjust for your model's output

    return jsonify({'prediction': predicted_class})


if __name__ == '__main__':
    app.run(debug=True)
