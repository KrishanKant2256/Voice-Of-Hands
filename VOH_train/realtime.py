import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from collections import deque
import time

# =========================
# LOAD MODEL & LABELS
# =========================
model = tf.keras.models.load_model("isl_landmark_model.keras")
labels = np.load("label_classes.npy", allow_pickle=True)

print("✅ Model loaded")
print("✅ Classes:", labels)

# =========================
# MEDIAPIPE HANDS
# =========================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# =========================
# SETTINGS
# =========================
CONF_THRESHOLD = 0.6
STABLE_FRAMES_REQUIRED = 8   # how long to hold a sign
PREDICTION_WINDOW = 10

# =========================
# STATE VARIABLES
# =========================
pred_queue = deque(maxlen=PREDICTION_WINDOW)

sentence = []

current_word = "—"
current_conf = 0.0

stable_word = None
stable_count = 0
word_locked = False

# =========================
# HELPERS
# =========================
def extract_landmarks(results):
    features = []

    for hand in results.multi_hand_landmarks[:2]:
        for lm in hand.landmark:
            features.extend([lm.x, lm.y, lm.z])

    while len(features) < 126:
        features.extend([0.0, 0.0, 0.0])

    return np.array(features).reshape(1, 126)


def draw_hand_box(frame, hand_landmarks):
    h, w, _ = frame.shape
    xs = [int(lm.x * w) for lm in hand_landmarks.landmark]
    ys = [int(lm.y * h) for lm in hand_landmarks.landmark]

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    pad = 20
    min_x, min_y = max(0, min_x - pad), max(0, min_y - pad)
    max_x, max_y = min(w, max_x + pad), min(h, max_y + pad)

    cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)


def draw_ui(frame):
    # Top bar
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 90), (15, 15, 15), -1)

    cv2.putText(frame, "ISL TRANSLATOR", (20, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 255), 2)

    cv2.putText(frame, f"Word: {current_word}", (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.putText(frame, f"{int(current_conf * 100)}%",
                (frame.shape[1] - 100, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    # Stability bar
    bar_w = int((stable_count / STABLE_FRAMES_REQUIRED) * 200)
    bar_w = min(bar_w, 200)
    cv2.rectangle(frame, (20, 80), (220, 90), (80, 80, 80), -1)
    cv2.rectangle(frame, (20, 80), (20 + bar_w, 90), (0, 255, 0), -1)

    # Bottom sentence bar
    cv2.rectangle(frame, (0, frame.shape[0] - 80),
                  (frame.shape[1], frame.shape[0]), (15, 15, 15), -1)

    sentence_text = " ".join(sentence)
    cv2.putText(frame, sentence_text, (20, frame.shape[0] - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.putText(frame, "Hold sign to confirm | Remove hand for next",
                (frame.shape[1] - 480, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)


# =========================
# CAMERA
# =========================
cap = cv2.VideoCapture(0)
print("📸 Camera started | Press Q to quit")

# =========================
# MAIN LOOP
# =========================
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_lms in results.multi_hand_landmarks:
            draw_hand_box(frame, hand_lms)

        features = extract_landmarks(results)
        preds = model.predict(features, verbose=0)[0]

        idx = np.argmax(preds)
        conf = preds[idx]
        label = labels[idx]

        if conf >= CONF_THRESHOLD:
            current_word = label
            current_conf = conf

            if label == stable_word:
                stable_count += 1
            else:
                stable_word = label
                stable_count = 1

            if stable_count >= STABLE_FRAMES_REQUIRED and not word_locked:
                sentence.append(stable_word)
                word_locked = True

        else:
            current_word = "Detecting..."
            current_conf = conf
            stable_word = None
            stable_count = 0

    else:
        # HAND RELEASED → RESET LOCK
        current_word = "No Hand"
        current_conf = 0.0
        stable_word = None
        stable_count = 0
        word_locked = False

    draw_ui(frame)
    cv2.imshow("ISL Translator", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if key == ord("c"):
        sentence = []
        word_locked = False

# =========================
# CLEANUP
# =========================
cap.release()
cv2.destroyAllWindows()
hands.close()
print("👋 Closed safely")
