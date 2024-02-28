from src.features.build_features import Preprocessor
from PIL import Image
import streamlit as st
import numpy as np
import keras
import cv2
import os

# Get model
my_model = keras.models.load_model("/home/irpro16/ml-projects/tumor-class/models/my_model9_76_4")

# Get Preprocessor module
preprocess = Preprocessor()

# Streamlit
st.title("Brain-mri Tumor Classification")
st.text(f"Predicts images with {76}% accuracy!")

# Get input 
img = st.file_uploader("Chosose a file", type=['png', 'jpg'])

if img is not None:
    open_image = Image.open(img)
    image_array = np.asarray(open_image)
    preprocessed_image = preprocess.preprocess_image(image_array)
    
if st.button("Predict"):
    prediction = my_model.predict(preprocessed_image)
    result = np.argmax(prediction)
    
    if result == 0:
        st.header("Glioma Tumor")
    elif result == 1:
        st.header('Meningioma Tumor')
    elif result == 2:
        st.header("No Tumor")
    else:
        st.header("Pituitary Tumor")    