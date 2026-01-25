import cv2
import mediapipe as mp
import csv
import os
import time

# ========================
# SETTINGS
# ========================

# 🔤 Alphabets (static)
ALPHABETS = [
    "A","B","C","D","E","F","G","H","I",
    "K","L","M","N","O","P","Q","R","S",
    "T","U","V","W","X","Y"
]

# 🔢 Numbers
NUMBERS = ["0","1","2","3","4","5","6","7","8","9"]

# 🧠 Basic ISL Words 
WORDS = [
    "YES", "NO", "HELLO", "STOP", "ME", "YOU",
    "THANKYOU", "WELCOME", "LOVE", "AGAIN",
    "TEACHER", "NAMASTE", "GOOD", "SMILE",
    "WRONG", "HELP"
]

GESTURES = ALPHABETS + NUMBERS + WORDS

SAMPLES_PER_GESTURE = 250

# ========================
# DIRECTORIES
# ========================
BASE_DIR = "data"
LANDMARK_DIR = os.path.join(BASE_DIR, "landmarks")
IMAGE_DIR = os.path.join(BASE_DIR, "images")

CSV_PATH = os.path.join(LANDMARK_DIR, "landmarks.csv")

os.makedirs(LANDMARK_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)

# ========================
# MEDIAPIPE
# ========================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)

# ========================
# CSV HEADER
# ========================
header = []
for i in range(42):
    header.extend([f"x{i}", f"y{i}", f"z{i}"])
header.append("label")

if not os.path.exists(CSV_PATH):
    with open(CSV_PATH, "w", newline="") as f:
        csv.writer(f).writerow(header)

print("📸 Press SPACE to capture | Q to quit")

# ========================
# DATA COLLECTION
# ========================
for gesture in GESTURES:
    print(f"\n➡️ Collecting: {gesture}")
    count = 0

    # Create image folder for gesture
    gesture_img_dir = os.path.join(IMAGE_DIR, gesture)
    os.makedirs(gesture_img_dir, exist_ok=True)

    while count < SAMPLES_PER_GESTURE:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        cv2.putText(frame, f"Gesture: {gesture}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.putText(frame, f"Samples: {count}/{SAMPLES_PER_GESTURE}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)

        if results.multi_hand_landmarks:
            landmarks = []

            for hand_lms in results.multi_hand_landmarks[:2]:
                for lm in hand_lms.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])

            # Pad if only one hand
            if len(results.multi_hand_landmarks) == 1:
                landmarks.extend([0.0] * (21 * 3))

            if len(landmarks) == 126:
                key = cv2.waitKey(1) & 0xFF
                if key == 32:  # SPACE
                    # Save landmark row
                    with open(CSV_PATH, "a", newline="") as f:
                        csv.writer(f).writerow(landmarks + [gesture])

                    # Save image
                    img_name = f"img_{count:04d}.jpg"
                    img_path = os.path.join(gesture_img_dir, img_name)
                    cv2.imwrite(img_path, frame)

                    count += 1
                    time.sleep(0.05)  # small delay for variation

        cv2.imshow("Collecting ISL Data", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            cap.release()
            cv2.destroyAllWindows()
            exit()

cap.release()
cv2.destroyAllWindows()
print("✅ Landmark + Image dataset collection complete")
