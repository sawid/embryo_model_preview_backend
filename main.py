# app.py
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.xception import Xception, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from PIL import Image
import requests
import numpy as np
import keras.backend as K
from io import BytesIO

app = Flask(__name__)

def get_f1(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1_val

# Assuming you have a custom F1 metric named 'get_f1'
custom_objects = {'get_f1': get_f1}

# Load the pre-trained InceptionV3 model
inception_model = load_model('InceptionV3_Optimizer_RMSprop_40epoch.h5', custom_objects=custom_objects)

# Load the pre-trained Xception model
xception_model = load_model('Xception_40epoch.h5', custom_objects=custom_objects)  # Replace 'your_xception_model.h5' with the actual path to your Xception model file

def preprocess_input_image(img_path):
    response = requests.get(img_path)
    img = Image.open(BytesIO(response.content)).convert('RGB')
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

def decode_predictions_custom(predictions, labels):
    predictions = predictions.astype(float)
    decoded_predictions = [(label, score) for label, score in zip(labels, predictions[0])]
    return decoded_predictions

@app.route('/predict_inception', methods=['POST'])
def predict_inception():
    try:
        data = request.get_json()
        img_path = data['image_path']
        img_array = preprocess_input_image(img_path)
        predictions = inception_model.predict(img_array)
        decoded_predictions = decode_predictions_custom(predictions, labels=['2-1-1', '2-1-2', '2-1-3', '2-2-1', '2-2-2', '2-2-3', '2-3-3', '3-1-1', '3-1-2', '3-1-3', '3-2-1', '3-2-2', '3-2-3', '3-3-2', '3-3-3', '4-2-2', 'Arrested', 'Early', 'Morula'])
        return jsonify({'predictions': decoded_predictions})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/predict_xception', methods=['POST'])
def predict_xception():
    try:
        data = request.get_json()
        img_path = data['image_path']
        img_array = preprocess_input_image(img_path)
        predictions = xception_model.predict(img_array)
        decoded_predictions = decode_predictions_custom(predictions, labels=['2-1-1', '2-1-2', '2-1-3', '2-2-1', '2-2-2', '2-2-3', '2-3-3', '3-1-1', '3-1-2', '3-1-3', '3-2-1', '3-2-2', '3-2-3', '3-3-2', '3-3-3', '4-2-2', 'Arrested', 'Early', 'Morula'])
        return jsonify({'predictions': decoded_predictions})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
