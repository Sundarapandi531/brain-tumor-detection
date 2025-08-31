import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import os


MODEL_PATH = "brain_tumor_resnet101.h5"
model = load_model(MODEL_PATH)

# Class labels
classes = ["Glioma Tumor", "Meningioma Tumor", "Pituitary Tumor", "No Tumor"]

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("Brain Tumor MRI Classification")
st.write("Upload an MRI image and the model will predict the tumor type.")

uploaded_file = st.file_uploader("Choose an MRI image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption='Uploaded MRI Image', use_column_width=True)

    # Preprocess image
    img = img.resize((256, 256))  # resize to match training
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.resnet.preprocess_input(img_array)

    # Predict
    prediction = model.predict(img_array)
    predicted_class = classes[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    # Display result
    st.success(f"Predicted Tumor Type: {predicted_class}")
    st.info(f"Confidence: {confidence:.2f}%")
