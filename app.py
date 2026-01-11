import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from PIL import Image

# Model load karein (ensure karein file path sahi ho)
@st.cache_resource
def get_model():
    # Dono mein se jo behtar result de rahi hai wo likhein
    # Mashwara: v2 wali file use karein
    return load_model('v2_brain_tumor_detector_vgg16.h5')

model = get_model()
class_labels = ['Glioma Tumor', 'Meningioma Tumor', 'No Tumor', 'Pituitary Tumor']

# UI Setup
st.title("🧠 Brain Tumor Detector")
st.write("Upload an MRI image to check for tumors.")

uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display Image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded MRI Scan', use_container_width=True)
    
    # Preprocessing
    img = image.resize((128, 128))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Prediction
    if st.button('Predict'):
        with st.spinner('Analyzing...'):
            predictions = model.predict(img_array)
            predicted_class = np.argmax(predictions, axis=1)[0]
            confidence = np.max(predictions) * 100
            
            result = class_labels[predicted_class]
            
            # Result Display
            if result == 'No Tumor':
                st.success(f"Result: {result} (Confidence: {confidence:.2f}%)")
            else:
                st.error(f"Detection: {result} (Confidence: {confidence:.2f}%)")