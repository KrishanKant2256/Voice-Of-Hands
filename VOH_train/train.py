import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# =========================
# LOAD DATASET
# =========================
CSV_PATH = "data/landmarks/landmarks.csv"

df = pd.read_csv(CSV_PATH)
print("✅ Dataset loaded:", df.shape)

# =========================
# FEATURES & LABELS
# =========================
X = df.iloc[:, :-1].values   # 126 features
y = df.iloc[:, -1].values   # labels

print("Feature shape:", X.shape)

# =========================
# LABEL ENCODING
# =========================
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

np.save("label_classes.npy", label_encoder.classes_)
print("✅ Labels saved")

# =========================
# TRAIN / TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

# =========================
# MODEL
# =========================
model = Sequential([
    Dense(256, activation="relu", input_shape=(126,)),
    Dropout(0.3),

    Dense(128, activation="relu"),
    Dropout(0.3),

    Dense(64, activation="relu"),

    Dense(len(label_encoder.classes_), activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# =========================
# TRAIN
# =========================
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=100,
    batch_size=32,
    callbacks=[early_stop]
)

# =========================
# EVALUATION
# =========================
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"✅ Test Accuracy: {acc * 100:.2f}%")

# =========================
# SAVE MODEL
# =========================
model.save("isl_landmark_model.keras")
print("✅ Model saved as isl_landmark_model.keras")
