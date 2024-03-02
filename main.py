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

load_dotenv()

app = Flask(__name__)

# Cloudinary configuration
cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET")
)

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

def generate_heatmap_and_upload(model, image_path):
    # Load InceptionV3 model
    inception_model = model

    # Read and preprocess the input image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (299, 299))
    X = np.expand_dims(img, axis=0).astype(np.float32)
    X = preprocess_input(X)

    # Create a model with intermediate layers
    conv_output = inception_model.get_layer("mixed10").output
    pred_output = inception_model.output
    model = Model(inception_model.input, outputs=[conv_output, pred_output])

    # Get intermediate layer activations and predictions
    conv, pred = model.predict(X)

    # Get the target class and weights
    target = np.argmax(pred, axis=1).squeeze()
    w, b = model.get_layer("predictions").weights
    weights = w[:, target].numpy()
    
    # Generate the heatmap
    heatmap = conv.squeeze() @ weights

    # Update scale for visualization
    scale = 299 / 8

    # Save the heatmap image
    heatmap_image_path = 'heatmap.png'
    plt.imsave(heatmap_image_path, zoom(heatmap, zoom=(scale, scale)), cmap='jet', format='png', vmin=0, vmax=1)

    # Upload the heatmap image to Cloudinary
    heatmap_url = upload_to_cloudinary(heatmap_image_path, cloudinary_config)

    # Remove the locally saved heatmap image
    os.remove(heatmap_image_path)

    return heatmap_url


@app.route('/predict_inception', methods=['POST'])
def predict_inception():
    try:
        data = request.get_json()
        img_path = data['image_path']
        img_array = preprocess_input_image(img_path)
        predictions = inception_model.predict(img_array)
        decoded_predictions = decode_predictions_custom(predictions, labels=['2-1-1', '2-1-2', '2-1-3', '2-2-1', '2-2-2', '2-2-3', '2-3-3', '3-1-1', '3-1-2', '3-1-3', '3-2-1', '3-2-2', '3-2-3', '3-3-2', '3-3-3', '4-2-2', 'Arrested', 'Early', 'Morula'])
        layer_names_cnn = ['conv2d', 'conv2d_1', 'conv2d_2', 'conv2d_3']
        layer_names_activation = ['block1_conv1_act', 'block1_conv2_act', 'block2_sepconv2_act', 'block3_sepconv1_act']
        result_path_cnn = visualize_intermediate_activations(xception_model, layer_names_cnn, img_path)
        result_path_act = visualize_intermediate_activations(xception_model, layer_names_activation, img_path)
        print(f"Concatenated activations saved at: {result_path_cnn}")
        return jsonify({'predictions': decoded_predictions, 'image_url_cnn': result_path_cnn, 'image_url_act': result_path_act})
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
        print(f"Concatenated activations saved at: {result_path_cnn}")
        return jsonify({'predictions': decoded_predictions, 'image_url_cnn': result_path_cnn, 'image_url_act': result_path_act})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(port=8000, debug=True)
