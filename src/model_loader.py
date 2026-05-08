import tensorflow_hub as hub
import streamlit as st

@st.cache_resource
def get_style_model():
    # Magenta'nın hızlı Arbitrary Style Transfer modeli
    return hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
