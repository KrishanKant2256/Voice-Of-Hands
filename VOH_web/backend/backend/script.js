const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

const signDiv = document.getElementById("sign");
const confidenceDiv = document.getElementById("confidence");

const socket = new WebSocket("ws://127.0.0.1:8000/ws/gesture/");

socket.onopen = () => {
    console.log("✅ WebSocket connected");
};

socket.onmessage = (event) => {
    const data = JSON.parse(event.data);

    if (data.sign) {
        signDiv.innerText = data.sign;
        confidenceDiv.innerText =
            data.confidence ? `Confidence: ${(data.confidence * 100).toFixed(1)}%` : "";
    }
};

navigator.mediaDevices.getUserMedia({ video: true })
    .then(stream => {
        video.srcObject = stream;
        video.onloadedmetadata = () => {
            video.play();
            sendFrames();
        };
    });

function sendFrames() {
    setInterval(() => {
        if (socket.readyState !== WebSocket.OPEN) return;

        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;

        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

        const imageBase64 = canvas.toDataURL("image/jpeg");

        socket.send(JSON.stringify({
            image: imageBase64
        }));
    }, 200);
}
