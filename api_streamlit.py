import streamlit as st
import cv2
import numpy as np
import gdown  # To download the model from Google Drive
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

# Function to download the model from Google Drive
@st.cache(allow_output_mutation=True)
def load_cached_model(file_id, output_path):
    gdown.download(f"https://drive.google.com/uc?id={file_id}", output_path, quiet=False)
    model = load_model(output_path)
    return model

def predict_from_camera(model, base_model):
    cap = cv2.VideoCapture(0)

    st.write("Live Stream:")
    stream = st.empty()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        resized_frame = cv2.resize(frame, (224, 224))
        preprocessed_frame = img_to_array(resized_frame)
        preprocessed_frame = np.expand_dims(preprocessed_frame, axis=0)
        preprocessed_frame = preprocess_input(preprocessed_frame)

        features = base_model.predict(preprocessed_frame)
        features_flatten = features.reshape(1, -1)

        prediction = model.predict(features_flatten)[0]
        class_label = np.argmax(prediction)
        class_prob = prediction[class_label]

        label = "Harassment" if class_label == 1 else "Non-Harassment"
        prob_text = f"{label} ({class_prob:.2f})"
        cv2.putText(frame, prob_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        stream.image(frame, channels="BGR", caption="Live Prediction")

    cap.release()

def main():
    st.title("Live Harassment Detection")

    # Google Drive file ID
    file_id = "1GP2IdE-mPdQ9D3ouDIqbIU-3gl26Kgf2"

    # Path to where the model will be downloaded
    model_path = "weight.hdf5"

    # Download and cache the model
    model = load_cached_model(file_id, model_path)

    base_model = VGG16(weights='imagenet', include_top=False)

    st.write("Press the button to start prediction:")
    if st.button("Start Prediction"):
        predict_from_camera(model, base_model)

if __name__ == "__main__":
    main()
