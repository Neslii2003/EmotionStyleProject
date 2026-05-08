import os
import sys

try:
    import pkg_resources
except ImportError:
    import setuptools
    # Bu satır, pkg_resources'ı zorla sisteme tanıtır
    sys.modules['pkg_resources'] = setuptools.pkg_resources 

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image
from deepface import DeepFace

# --- 2. GÖRÜNTÜ İŞLEME FONKSİYONLARI ---

def load_img(path_to_img):
    """Resmi yükler ve modelin beklediği formatta normalize eder."""
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = img[tf.newaxis, :]
    return img

def preprocess_image(image_np):
    """Numpy dizisini model için Tensör formatına çevirir."""
    img = tf.convert_to_tensor(image_np, dtype=tf.float32)
    img = img / 255.0
    if len(img.shape) == 3:
        img = img[tf.newaxis, :]
    return img

@st.cache_resource
def load_style_model():
    """Magenta Style Transfer modelini yükler."""
    return hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

# --- 3. ARAYÜZ YAPILANDIRMASI ---

st.set_page_config(page_title="EmotionStyle AI", layout="wide")
st.title("🎨 Duygu Odaklı Dinamik Stil Transferi")
st.markdown("---")

with st.spinner("Yapay Zeka Modelleri Hazırlanıyor..."):
    hub_model = load_style_model()

# Yan Panel Ayarları
st.sidebar.header("⚙️ Kontrol Paneli")
alpha = st.sidebar.slider("Stil Yoğunluğu", 0.0, 1.0, 0.8)
source = st.sidebar.radio("Görsel Kaynağı", ["Kamera", "Dosya Yükle"])

if source == "Kamera":
    content_file = st.camera_input("Fotoğraf Çek")
else:
    content_file = st.file_uploader("Bir Resim Seçin", type=['jpg', 'jpeg', 'png'])

# --- 4. ANA İŞLEME DÖNGÜSÜ ---

if content_file:
    # Resmi aç ve RGB formatına zorla
    content_image = Image.open(content_file).convert('RGB')
    content_array = np.array(content_image)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🖼️ Orijinal Görüntü")
        st.image(content_image, use_container_width=True)

    with st.spinner("Analiz ve Sanat Sentezi Yapılıyor..."):
        try:
            # 1. Adım: DeepFace ile Duygu Analizi
            # enforce_detection=False: Yüz tam görünmese bile hata vermesini engeller
            analysis = DeepFace.analyze(content_array, actions=['emotion'], enforce_detection=False)
            emotion = analysis[0]['dominant_emotion'].lower()
            st.sidebar.success(f"Tespit Edilen Duygu: **{emotion.upper()}**")

            # 2. Adım: Duyguya Göre Stil Belirleme
            # Klasördeki dosyaları duygu ismine göre (uzantıdan bağımsız) tara
            current_dir = os.path.dirname(os.path.abspath(__file__))
            style_folder = os.path.join(current_dir, "assets", "styles")
            
            if os.path.exists(style_folder):
                available_files = os.listdir(style_folder)
                style_file_name = next((f for f in available_files if f.lower().startswith(emotion)), None)
                
                if style_file_name:
                    style_path = os.path.join(style_folder, style_file_name)
                    st.sidebar.image(style_path, caption=f"Eşleşen Stil: {emotion}")
                    
                    # 3. Adım: Stil Transferi Uygulama
                    content_tensor = preprocess_image(content_array)
                    style_tensor = load_img(style_path)

                    # Model tahmini
                    outputs = hub_model(tf.constant(content_tensor), tf.constant(style_tensor))
                    stylized_image = outputs[0]

                    # Orijinal resim ile stili harmanla (Alpha blending)
                    content_resized = tf.image.resize(content_tensor, stylized_image.shape[1:3])
                    final_image = alpha * stylized_image + (1 - alpha) * content_resized

                    with col2:
                        st.subheader("🎭 AI Sanat Çıktısı")
                        output_display = np.array(final_image[0])
                        st.image(output_display, use_container_width=True)
                        
                        # İndirme Butonu
                        final_pil = Image.fromarray((output_display * 255).astype(np.uint8))
                        st.download_button(
                            label="Sanat Eserini İndir",
                            data=final_pil.tobytes(),
                            file_name=f"emotion_style_{emotion}.png",
                            mime="image/png"
                        )
                else:
                    st.error(f"'{emotion}' duygusu için stil dosyası bulunamadı.")
            else:
                st.error("'assets/styles/' klasörü eksik!")

        except Exception as e:
            st.error(f"Hata oluştu: {str(e)}")

st.markdown("---")
st.caption("AI Engineering Project - Neslihan Gün")
