const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

const signEl = document.getElementById("sign");
const confEl = document.getElementById("conf");
const startBtn = document.getElementById("start");

const WS_URL = "ws://127.0.0.1:8000/ws/gesture/";

let socket = null;
let lastSpoken = "";


let frameCount = 0;

function sendFrame() {
    frameCount++;
    if (frameCount % 3 !== 0) return; // skip frames

    // send frame here
}

startBtn.onclick = async () => {
    // 🔓 unlock speech
    const unlock = new SpeechSynthesisUtterance(" ");
    window.speechSynthesis.speak(unlock);

    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = stream;
    await video.play();

    socket = new WebSocket(WS_URL);

    socket.onopen = () => sendFrame();

    socket.onmessage = (e) => {
        const data = JSON.parse(e.data);
        if (!data.sign) return;

        signEl.textContent = data.sign;
        confEl.textContent =
            `Confidence ${(data.confidence * 100).toFixed(1)}%`;

        if (data.confidence > 0.75 && data.sign !== lastSpoken) {
            speak(data.sign);
            lastSpoken = data.sign;
        }
    };
};

function sendFrame() {
    if (!socket || socket.readyState !== 1) return;

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0);

    socket.send(JSON.stringify({
        image: canvas.toDataURL("image/jpeg", 0.6)
    }));

    setTimeout(sendFrame, 120);
}

function speak(text) {
    window.speechSynthesis.cancel();
    const u = new SpeechSynthesisUtterance(text);
    u.rate = 0.9;
    window.speechSynthesis.speak(u);
}
