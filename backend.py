from flask import Flask, Response
import cv2
from facenet_pytorch import MTCNN  # Install via pip install facenet-pytorch
import torch

RTSP = "rtsp://admin:admin@192.168.1.4:1935"
app = Flask(__name__)

# Initialize the MTCNN face detector
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=True, device=device)

def generate_frames():
    cap = cv2.VideoCapture(RTSP)  # Open the camera
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect faces using MTCNN
        boxes, _ = mtcnn.detect(frame)

        # Draw bounding boxes around detected faces
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Encode the frame as JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        # Yield the frame as part of an MJPEG stream
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)  # Listen on all network interfaces