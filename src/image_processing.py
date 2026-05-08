import tensorflow as tf
import numpy as np

def load_and_preprocess(img, max_dim=512):
    # Eğer input bir PIL Image ise numpy array'e çevir
    if not isinstance(img, np.ndarray):
        img = np.array(img)
    
    # BGR'den RGB'ye (OpenCV ise)
    img = tf.convert_to_tensor(img, dtype=tf.float32)
    img = img / 255.0  # Normalize
    
    # Boyutlandırma (Performans için kritik)
    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim
    new_shape = tf.cast(shape * scale, tf.int32)
    
    img = tf.image.resize(img, new_shape)
    return img[tf.newaxis, :]
