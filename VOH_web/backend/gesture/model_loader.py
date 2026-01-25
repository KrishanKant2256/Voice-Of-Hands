import numpy as np
import os
import tf_keras as keras
import tensorflow as tf

print("🧠 Loading ISL model with LEGACY TF-Keras...")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "isl_landmark_model.keras")
LABEL_PATH = os.path.join(BASE_DIR, "model", "label_classes.npy")

model = keras.models.load_model(MODEL_PATH, compile=False)
labels = np.load(LABEL_PATH, allow_pickle=True)

print("✅ ISL model loaded successfully")
print("✅ TensorFlow version:", tf.__version__)
