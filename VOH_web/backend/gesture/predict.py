import time
import numpy as np
from collections import deque
from .model_loader import model, labels

BUFFER_SIZE = 7
prediction_buffer = deque(maxlen=BUFFER_SIZE)

PREDICT_EVERY = 0.15  # seconds
last_predict_time = 0

def predict_sign(landmarks):
    global last_predict_time

    now = time.time()
    if now - last_predict_time < PREDICT_EVERY:
        # reuse last stable prediction
        idx = max(set(prediction_buffer), key=prediction_buffer.count)
        return labels[idx], 0.0

    last_predict_time = now

    landmarks = landmarks.reshape(1, -1)
    preds = model.predict(landmarks, verbose=0)[0]
    idx = int(np.argmax(preds))
    conf = float(preds[idx])

    prediction_buffer.append(idx)
    final_idx = max(set(prediction_buffer), key=prediction_buffer.count)

    return labels[final_idx], conf
