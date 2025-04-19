
# Smart Waste Classifier
A Smart Image Recognition System to help users classify waste items into Recycling, Compost or General Waste using deep learning and traditional machine learning models.

## Project Highlights

- Built using MobileNetV2 (CNN) for end-to-end waste image classification.
- Traditional ML models (SVM, KNN, XGBoost) trained on deep features for comparison.
- Transfer Learning used to improve accuracy and reduce training time.
- Interactive web app deployed using Streamlit.
- Users can upload an image and instantly see which bin it belongs to.

## Demo
Try the live app here:  
Smart Waste Classifier Web App: https://ml-final-project-4nd43mkqnvpeyhtwxulfbb.streamlit.app/

## How to Run Locally

### 1. Clone the repo

git clone https://github.com/Mohammad-Shehnaj/ML-Final-Project.git
cd ML-Final-Project

### 2. Install dependencies

pip install -r requirements.txt

### 3. Run the Streamlit app

streamlit run streamlit_app.py
