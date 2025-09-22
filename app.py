import streamlit as st
from streamlit_option_menu import option_menu
import os
import numpy as np
import cv2
from PIL import Image
from io import BytesIO
import requests
import pandas as pd
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras import models

# Page configuration
st.set_page_config(page_title="Tribe Classification", layout="wide")

# Model and class setup
MODEL_PATH = r"C:\Users\vamsh\Downloads\SportsImageClassify.h5"
CLASS_NAMES = ["Chenchus", "Gonds", "Lambada", "Gadaba", "Koya"]

@st.cache_resource
def load_model():
    return models.load_model(MODEL_PATH)

model = load_model()

def cnn_preprocess_image(img):
    img = cv2.resize(img, (224, 224))
    if np.max(img) > 1:
        img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def predict_image(image):
    image = Image.open(image).convert('RGB')
    image = image.resize((224, 224))
    img_array = img_to_array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    preds = model.predict(img_array)
    top3_idx = np.argsort(preds[0])[-3:][::-1]
    top_preds = [(CLASS_NAMES[i], preds[0][i]) for i in top3_idx]
    df = pd.DataFrame(top_preds, columns=["Tribe", "Probability"])
    df["Probability"] = df["Probability"].apply(lambda x: f"{x*100:.2f}%")
    return df

# Streamlit UI
with st.sidebar:
    selected = option_menu(
        "Tribe Classifier",
        ["Home", "Upload & Predict"],
        icons=["house", "cloud-upload"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "#000"},
            "icon": {"color": "white", "font-size": "15px"},
            "nav-link": {"font-size": "15px", "text-align": "left", "margin": "0px", "--hover-color": "#000"},
            "nav-link-selected": {"background-color": "green"},
        }
    )

if selected == "Home":
    st.subheader(":green[Tribe Image Classification]")
    st.markdown("<hr style='margin-top: 1px;'>", unsafe_allow_html=True)
    st.write("Classifying tribal images into 5 categories: Chenchus, Gonds, Lambada, Gadaba, Koya.")
    st.image("https://cdn-icons-png.flaticon.com/512/4359/4359963.png", width=300)
else:
    st.subheader(":red[Upload Image or Enter the URL of Image]")
    st.markdown("<hr style='margin-top: 1px;'>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([0.5, 0.05, 0.5])
    with col1:
        pics = st.file_uploader("Select Images", type=["png", "jpeg", "jpg"], accept_multiple_files=True)
    with col2:
        st.write(":blue[Or]")
    with col3:
        url = st.text_input("Enter Image URL Here")

    if st.button("Analyze", type="primary"):
        col1, col2 = st.columns(2)
        with col1:
            st.write(":green[Given Image]")
            if url:
                response = requests.get(url)
                st.image(Image.open(BytesIO(response.content)))
            else:
                for pic in pics:
                    st.image(Image.open(pic))
        with col2:
            with st.spinner("Analyzing..."):
                if url:
                    response = requests.get(url)
                    df = predict_image(BytesIO(response.content))
                    st.write("Top-3 predictions:")
                    st.table(df)
                else:
                    for pic in pics:
                        df = predict_image(pic)
                        st.write("Top-3 predictions:")
                        st.table(df)
