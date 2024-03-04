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
from datetime import datetime
import cloudinary
from cloudinary.uploader import upload
from dotenv import load_dotenv
import os
import cv2
from scipy.ndimage import zoom
from tensorflow.keras.applications import imagenet_utils
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3

load_dotenv()

app = Flask(__name__)

# Cloudinary configuration
cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET")
)

def generate_and_upload_heatmap(image_url):
    # Download image from URL
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content)).convert('RGB')
    img = img.resize((299, 299))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img = preprocess_input(img_array)

    # Load InceptionV3 model
    inception_model = InceptionV3()

    # Modify the model to get intermediate layer output
    conv_output = inception_model.get_layer("mixed10").output
    pred_output = inception_model.output
    model = models.Model(inception_model.input, outputs=[conv_output, pred_output])

    # Get intermediate layer output and predictions
    conv, pred = model.predict(img)

    # Get target class
    target = np.argmax(pred, axis=1).squeeze()
    w, b = model.get_layer("predictions").weights
    weights = w[:, target].numpy()
    heatmap = conv.squeeze() @ weights

    # Normalize the heatmap
    heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))

    # Resize the heatmap to match the original image size
    heatmap = cv2.resize(heatmap, (img_array.shape[2], img_array.shape[1]))

    # Convert heatmap to BGR format
    heatmap_bgr = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)

    # Convert heatmap to the same data type as the original image
    heatmap_bgr = heatmap_bgr.astype(img_array[0].dtype)

    # Blend the heatmap with the original image
    alpha = 0.5  # Adjust the alpha value for blending
    overlay = cv2.addWeighted(img_array[0], 1 - alpha, heatmap_bgr, alpha, 0)

    # Save the result
    overlay_path = 'heatmap_overlay.png'
    cv2.imwrite(overlay_path, overlay)

    # Upload the overlaid image to Cloudinary using the provided function
    overlay_cloud_url = upload_to_cloudinary(overlay_path)

    return overlay_cloud_url

def generate_and_upload_heatmap_xception(image_url):
    # Download image from URL
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content)).convert('RGB')
    img = img.resize((299, 299))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img = preprocess_input(img_array)

    # Load Xception model
    xception_model = Xception()

    # Modify the model to get intermediate layer output
    conv_output = xception_model.get_layer("block14_sepconv2_act").output  # Choose the appropriate layer for Xception
    pred_output = xception_model.output
    model = models.Model(xception_model.input, outputs=[conv_output, pred_output])

    # Get intermediate layer output and predictions
    conv, pred = model.predict(img)

    # Get target class
    target = np.argmax(pred, axis=1).squeeze()
    w, b = model.get_layer("predictions").weights
    weights = w[:, target].numpy()
    heatmap = conv.squeeze() @ weights

    # Normalize the heatmap
    heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))

    # Resize the heatmap to match the original image size
    heatmap = cv2.resize(heatmap, (img_array.shape[2], img_array.shape[1]))

    # Convert heatmap to BGR format
    heatmap_bgr = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)

    # Convert heatmap to the same data type as the original image
    heatmap_bgr = heatmap_bgr.astype(img_array[0].dtype)

    # Blend the heatmap with the original image
    alpha = 0.5  # Adjust the alpha value for blending
    overlay = cv2.addWeighted(img_array[0], 1 - alpha, heatmap_bgr, alpha, 0)

    # Save the result
    overlay_path = 'heatmap_overlay.png'
    cv2.imwrite(overlay_path, overlay)

    # Upload the overlaid image to Cloudinary using the provided function
    overlay_cloud_url = upload_to_cloudinary(overlay_path)

    return overlay_cloud_url

def upload_to_cloudinary(image_path):
    # Upload the image to Cloudinary
    result = upload(image_path)

    # Get the public URL of the uploaded image
    image_url = result['secure_url']

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
    image_url = upload_to_cloudinary(concatenated_img_path)

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
        layer_names_activation = ['activation', 'activation_1', 'activation_2', 'activation_3']
        result_path_cnn = visualize_intermediate_activations(inception_model, layer_names_cnn, img_path)
        result_path_act = visualize_intermediate_activations(inception_model, layer_names_activation, img_path)
        heatmap_url = generate_and_upload_heatmap(img_path)
        print(f"Concatenated activations saved at: {result_path_cnn}")
        return jsonify({'predictions': decoded_predictions, 'image_url_cnn': result_path_cnn, 'image_url_act': result_path_act, 'heatmap_url': heatmap_url})
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
        layer_names_cnn = ['block1_conv1', 'block1_conv2', 'conv2d', 'conv2d_1']
        layer_names_activation = ['block1_conv1_act', 'block1_conv2_act', 'block2_sepconv2_act', 'block3_sepconv1_act']
        result_path_cnn = visualize_intermediate_activations(xception_model, layer_names_cnn, img_path)
        result_path_act = visualize_intermediate_activations(xception_model, layer_names_activation, img_path)
        heatmap_url = generate_and_upload_heatmap_xception(img_path)
        print(f"Concatenated activations saved at: {result_path_cnn}")
        return jsonify({'predictions': decoded_predictions, 'image_url_cnn': result_path_cnn, 'image_url_act': result_path_act, 'heatmap_url': heatmap_url})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(port=8000, debug=True)
