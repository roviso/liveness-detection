const video = document.getElementById('video');

// Start video stream from webcam
navigator.mediaDevices.getUserMedia({ video: true })
    .then(stream => { video.srcObject = stream });

const ws = new WebSocket("ws://localhost:8000/ws");

video.addEventListener('loadeddata', async () => {
    while (video.readyState >= 2) {
        // Capture the frame from the webcam
        const frame = captureFrame(video);
        
        // Send this frame to FastAPI server
        ws.send(frame);

        // Wait for the result from the server
        ws.onmessage = function(event) {
            // Process server response here, e.g., display results on the page
        };

        await new Promise(r => setTimeout(r, 100)); // Frame rate control
    }
});

function captureFrame(video) {
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext('2d').drawImage(video, 0, 0);
    return canvas.toDataURL('image/jpeg');
}
