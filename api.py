import streamlit as st
import gdown
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Function to download the model from Google Drive
def download_model(file_id, destination_path):
    gdown.download(f"https://drive.google.com/uc?export=download&id={file_id}", destination_path, quiet=False)

# File ID of the model file on Google Drive
file_id = "1GP2IdE-mPdQ9D3ouDIqbIU-3gl26Kgf2"  # Replace with your actual file ID
model_path = "model_weights.h5"

# Download the model from Google Drive if it's not cached
@st.cache(allow_output_mutation=True)
def load_cached_model(file_id, model_path):
    download_model(file_id, model_path)
    return tf.keras.models.load_model(model_path)

# Load the trained model
model = load_cached_model(file_id, model_path)
base_model = VGG16(weights='imagenet', include_top=False)

# Function to preprocess the image
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)  # Add batch dimension as the model expects it
    return img

# Function to make predictions
def make_prediction(image_path):
    new_image = preprocess_image(image_path)

    # Extract features using the VGG16 base model
    new_image_features = base_model.predict(new_image)

    # Reshape the features
    new_image_features = new_image_features.reshape(1, 7 * 7 * 512)

    # Make predictions
    predictions = model.predict(new_image_features)

    # Since your model has 2 output neurons (softmax), you can use argmax to get the predicted class index
    predicted_class_index = np.argmax(predictions[0])

    # If your classes are labeled as 0 and 1, you can map the index back to class labels
    class_labels = {0: 'Healthy Workspace Environment :)', 1: '!! Sexual Harassment Detected !!'}
    predicted_class_label = class_labels[predicted_class_index]
    return predicted_class_label

# Streamlit app
def main():
    st.title("Sexual Harassment Detection")

    uploaded_file = st.file_uploader("Input an Image to detect any incident of Sexual harassment", type=["jpg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)

        # Make prediction on the uploaded image
        predicted_class = make_prediction(uploaded_file)

        # Show the prediction result
        st.write("Prediction:", predicted_class)

if __name__ == '__main__':
    main()
