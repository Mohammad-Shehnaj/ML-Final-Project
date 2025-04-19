# Importing Libraries
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image

# Load the trained CNN model
model = load_model("best_model.h5")
class_labels = ['compost', 'general', 'recycling']

# Set the title of the web/app
st.title("Smart Waste Classifier")
st.markdown("Upload an image of waste and this model will classify it into **Compost**, **General**, or **Recycling**.")

# Create a session state variable to track if prediction result has been shown
if 'result_displayed' not in st.session_state:
    st.session_state.result_displayed = False

# Show "Upload Another Image" button if a result has been displayed
if st.session_state.result_displayed:
    if st.button("Upload Another Image"):
        st.session_state.result_displayed = False
        st.rerun()

# Image upload section
if not st.session_state.result_displayed:
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
        
        # Preprocess image
        img = Image.open(uploaded_file).resize((224, 224)).convert("RGB")
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # Prediction using the trained model
        preds = model.predict(img_array)
        pred_class = class_labels[np.argmax(preds)]
        confidence = np.max(preds) * 100

        # Display the predicted category and confidence
        st.markdown(f"Prediction: **{pred_class.capitalize()}** ({confidence:.2f}%)")
        st.session_state.result_displayed = True
