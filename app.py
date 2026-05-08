import sys
import os
import numpy as np
from PIL import Image

# 1. Klasör ve Kütüphane Yollarını Yapılandır
current_dir = os.path.dirname(os.path.abspath(__file__))

# libs klasörünü Python'a tanıt
libs_path = os.path.join(current_dir, "libs")
if libs_path not in sys.path:
    sys.path.insert(0, libs_path)

import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
from deepface import DeepFace

# --- 2. GÖRÜNTÜ İŞLEME FONKSİYONLARI ---

def load_img(path_to_img):
    """Görüntüyü yükler ve 0-1 arasına normalize eder."""
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = img[tf.newaxis, :]
    return img

def preprocess_image(image_np):
    """Numpy dizisini Tensör formatına sokar."""
    img = tf.convert_to_tensor(image_np, dtype=tf.float32)
    img = img / 255.0
    if len(img.shape) == 3:
        img = img[tf.newaxis, :]
    return img

@st.cache_resource
def load_style_model():
    """Magenta Arbitrary Style Transfer modelini yükler."""
    return hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

# --- 3. ARAYÜZ AYARLARI ---

st.set_page_config(page_title="EmotionStyle AI", layout="wide")
st.title("🎨 Duygu Odaklı Dinamik Stil Transferi")

with st.spinner("Yapay Zeka Modelleri Hazırlanıyor..."):
    hub_model = load_style_model()

# Yan Panel
st.sidebar.header("⚙️ Ayarlar")
alpha = st.sidebar.slider("Stil Yoğunluğu", 0.0, 1.0, 0.8)
source = st.sidebar.radio("Kaynak", ["Kamera", "Dosya Yükle"])

content_file = st.camera_input("Fotoğraf Çek") if source == "Kamera" else st.file_uploader("Dosya Seç")

# --- 4. ANA ÇALIŞMA DÖNGÜSÜ ---

if content_file:
    content_image = Image.open(content_file).convert('RGB')
    content_array = np.array(content_image)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🖼️ Orijinal")
        st.image(content_image, use_container_width=True)

    with st.spinner("Duygu Analizi ve Stil Transferi Yapılıyor..."):
        try:
            # 1. Duygu Analizi
            analysis = DeepFace.analyze(content_array, actions=['emotion'], enforce_detection=False)
            emotion = analysis[0]['dominant_emotion'].lower()
            st.sidebar.success(f"Tespit Edilen Duygu: {emotion.upper()}")

            # 2. ESNEK DOSYA BULUCU (Uzantıdan bağımsız)
            style_folder = os.path.join(current_dir, "assets", "styles")
            
            # Klasördeki dosyaları tara, duygu ismiyle başlayanı seç
            if os.path.exists(style_folder):
                available_files = os.listdir(style_folder)
                # Örn: 'happy' ile başlayan dosyayı bul (happy.jpg.jpeg olsa bile bulur)
                style_file_name = next((f for f in available_files if f.lower().startswith(emotion)), None)
                
                if style_file_name:
                    style_path = os.path.join(style_folder, style_file_name)
                    st.sidebar.image(style_path, caption=f"Kullanılan Stil: {style_file_name}")
                    
                    # 3. İŞLEMLER
                    content_tensor = preprocess_image(content_array)
                    style_tensor = load_img(style_path)

                    # Stil Transferi Uygula
                    outputs = hub_model(tf.constant(content_tensor), tf.constant(style_tensor))
                    stylized_image = outputs[0]

                    # Harmanlama
                    content_resized = tf.image.resize(content_tensor, stylized_image.shape[1:3])
                    final_image = alpha * stylized_image + (1 - alpha) * content_resized

                    with col2:
                        st.subheader(f"🎨 AI Sanat Çıktısı")
                        output_display = np.array(final_image[0])
                        st.image(output_display, use_container_width=True)
                        
                        # Kaydetme Butonu
                        final_pil = Image.fromarray((output_display * 255).astype(np.uint8))
                        st.download_button("Resmi İndir", data=final_pil.tobytes(), file_name=f"{emotion}_art.png")
                else:
                    st.error(f"Hata: 'assets/styles/' içinde '{emotion}' ile başlayan dosya yok.")
            else:
                st.error("Hata: 'assets/styles/' klasörü bulunamadı!")

        except Exception as e:
            st.error(f"Bir hata oluştu: {e}")

st.markdown("---")
st.caption("AI Engineering Project - Neslihan Gün")