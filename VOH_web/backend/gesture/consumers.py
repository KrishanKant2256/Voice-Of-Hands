import json
import time
from channels.generic.websocket import AsyncWebsocketConsumer

from .hand_landmarks import extract_landmarks_from_base64
from .predict import predict_sign


class GestureConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.accept()
        print("✅ WebSocket connected")

        # ---- LATENCY + STABILITY ----
        self.last_send_time = 0
        self.SEND_EVERY = 0.12        # keep low latency
        self.no_hand_count = 0

        # ---- HOLD + CONFIDENCE ----
        self.last_sign = None
        self.sign_start_time = 0
        self.HOLD_TIME = 1.0          # 1 second hold
        self.CONFIDENCE_GATE = 0.7    # ignore weak preds

    async def disconnect(self, close_code):
        print("❌ WebSocket disconnected")

    async def receive(self, text_data):
        try:
            data = json.loads(text_data)

            if "image" not in data:
                return

            # ---- Throttle processing (latency control) ----
            now = time.time()
            if now - self.last_send_time < self.SEND_EVERY:
                return
            self.last_send_time = now

            # ---- Landmark extraction ----
            landmarks = extract_landmarks_from_base64(data["image"])

            # ---- NO HAND DETECTED ----
            if landmarks is None:
                self.no_hand_count += 1
                self.last_sign = None
                self.sign_start_time = 0

                if self.no_hand_count >= 2:
                    await self.send(text_data=json.dumps({
                        "sign": None,
                        "confidence": 0
                    }))
                return

            self.no_hand_count = 0

            # ---- Prediction ----
            sign, confidence = predict_sign(landmarks)

            # ---- Confidence gate ----
            if confidence < self.CONFIDENCE_GATE:
                self.last_sign = None
                self.sign_start_time = 0
                return

            # ---- HOLD LOGIC (ANTI‑SPAM) ----
            if sign == self.last_sign:
                if now - self.sign_start_time >= self.HOLD_TIME:
                    await self.send(text_data=json.dumps({
                        "sign": sign,
                        "confidence": round(confidence * 100, 2)
                    }))
                    self.sign_start_time = now  # reset after emit
            else:
                self.last_sign = sign
                self.sign_start_time = now

        except Exception as e:
            print("❌ Consumer error:", e)
