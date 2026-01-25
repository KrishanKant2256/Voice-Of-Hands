import cv2
import base64
import numpy as np
import mediapipe as mp

mp_hands = mp.solutions.hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def extract_landmarks_from_base64(data_url):
    if "," not in data_url:
        return None

    encoded = data_url.split(",")[1]
    img_bytes = base64.b64decode(encoded)
    img_array = np.frombuffer(img_bytes, np.uint8)

    image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = mp_hands.process(image_rgb)
    if not results.multi_hand_landmarks:
        return None

    landmarks = []
    for lm in results.multi_hand_landmarks[0].landmark:
        landmarks.extend([lm.x, lm.y, lm.z])

    landmarks = np.array(landmarks)

    # 🔥 REQUIRED: model expects 126 features
    if landmarks.shape[0] == 63:
        landmarks = np.concatenate([landmarks, np.zeros(63)])

    return landmarks
