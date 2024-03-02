# app.py
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.xception import Xception, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from tensorflow.keras import models
from PIL import Image
import requests
import numpy as np
import keras.backend as K
from io import BytesIO
import matplotlib.pyplot as plt
import firebase_admin
from firebase_admin import credentials, storage
from datetime import datetime

app = Flask(__name__)

# Initialize Firebase Admin SDK
cred = credentials.Certificate("firebase_credential.json")
firebase_admin.initialize_app(cred, {"storageBucket": "model-preview-backend.appspot.com"})
bucket = storage.bucket()

def upload_to_firebase(image_path, destination_path):
    # Upload the image to Firebase Storage
    blob = bucket.blob(destination_path)
    blob.upload_from_filename(image_path)

    # Get the public URL of the uploaded image
    image_url = blob.public_url

    return image_url

def generate_unique_filename(prefix="concatenated_activations"):
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    unique_filename = f"{prefix}_{timestamp}.png"
    return unique_filename

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

def visualize_intermediate_activations(base_model, layer_names, img_url, img_size=(224, 224), destination_prefix="concatenated_activations.png"):
    # Create the activation model
    layers_outputs = [base_model.get_layer(name).output for name in layer_names]
    activation_model = models.Model(inputs=base_model.input, outputs=layers_outputs)

    # Load and preprocess the image from URL
    response = requests.get(img_url)
    img = Image.open(BytesIO(response.content)).convert('RGB')
    img = img.resize(img_size)
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255.

    # Get intermediate layer activations
    activations = activation_model.predict(img_tensor)

    # Resize all visualizations to a common height
    common_height = min([activation.shape[1] for activation in activations])
    resized_activations = [np.array(Image.fromarray(activation[0, :, :, 0]).resize((common_height, common_height))) for activation in activations]

    # Concatenate the visualizations horizontally into a single image
    concatenated_img = np.concatenate(resized_activations, axis=1)

    # Save the concatenated image
    concatenated_img_path = 'concatenated_activations.png'
    Image.fromarray((concatenated_img * 255).astype(np.uint8)).save(concatenated_img_path)
    unique_filename = generate_unique_filename(prefix=destination_prefix)
    image_url = upload_to_firebase(concatenated_img_path, unique_filename)

    return image_url

@app.route('/predict_inception', methods=['POST'])
def predict_inception():
    try:
        data = request.get_json()
        img_path = data['image_path']
        img_array = preprocess_input_image(img_path)
        predictions = inception_model.predict(img_array)
        decoded_predictions = decode_predictions_custom(predictions, labels=['2-1-1', '2-1-2', '2-1-3', '2-2-1', '2-2-2', '2-2-3', '2-3-3', '3-1-1', '3-1-2', '3-1-3', '3-2-1', '3-2-2', '3-2-3', '3-3-2', '3-3-3', '4-2-2', 'Arrested', 'Early', 'Morula'])
        layer_names_cnn = ['conv2d', 'conv2d_1', 'conv2d_2', 'conv2d_3']
        result_path_cnn = visualize_intermediate_activations(xception_model, layer_names_cnn, img_path)
        print(f"Concatenated activations saved at: {result_path}")
        return jsonify({'predictions': decoded_predictions, 'image_url_cnn': result_path_cnn})
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
        layer_names_cnn = ['block1_conv1_act', 'block1_conv2_act', 'block2_sepconv2_act', 'block3_sepconv1_act']
        result_path_cnn = visualize_intermediate_activations(xception_model, layer_names_cnn, img_path)
        print(f"Concatenated activations saved at: {result_path}")
        return jsonify({'predictions': decoded_predictions, 'image_url_cnn': result_path_cnn})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(port=8000, debug=True)
