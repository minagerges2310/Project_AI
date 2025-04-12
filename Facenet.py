"""
Facenet.py
This script implements a face recognition system using MediaPipe for face detection 
and FaceNet for face recognition. The system connects to a MongoDB database to load 
known face embeddings and uses OpenCV for video capture and display. The main features 
include real-time face detection, recognition, and zooming in on a target face.
Classes:
    FaceData: A data structure to store face embeddings and associated names.
    FaceRecognitionSystem: The main class that handles face detection, recognition, 
                           video processing, and database interaction.
Functions:
    main(): Entry point of the application. Initializes the FaceRecognitionSystem 
            with a target name and starts the processing loop.
Key Features:
    - Connects to MongoDB to load known face embeddings.
    - Uses MediaPipe for face detection and FaceNet for face recognition.
    - Processes video frames from an RTSP stream or a default webcam.
    - Identifies faces in real-time and zooms in on a specified target face.
    - Logs system events and errors to a file and console.
Dependencies:
    - OpenCV (cv2)
    - MediaPipe (mediapipe)
    - NumPy (numpy)
    - PyTorch (torch)
    - FaceNet-PyTorch (facenet_pytorch)
    - PyMongo (pymongo)
    - Dataclasses (dataclasses)
    - Logging (logging)
    - System utilities (sys, datetime, typing)
Usage:
    Run the script and provide a target name when prompted. The system will attempt 
    to recognize the target face in the video stream and zoom in on it.
Example:
    $ python Facenet.py
    Enter the name to zoom in on: John Doe"""
    
import cv2
import mediapipe as mp
import numpy as np
from pymongo import MongoClient, errors
from facenet_pytorch import InceptionResnetV1
import torch
from typing import List, Tuple, Optional
from dataclasses import dataclass
import logging
import sys
from datetime import datetime
from HLS import generate_hls_url

# Hardcoded configuration

#stream =generate_hls_url('Camera_121') #This will generate the HLS URL for the Kinesis Video Stream named 'Camera_121'
RTSP_URL = "rtsp://admin:admin@192.168.1.4:1935"
MONGO_URI = "mongodb+srv://mina23:01555758130@cluster0.22ogf.mongodb.net/?retryWrites=true&w=majority"
FRAME_WIDTH = 1080 # Width of the video frame
FRAME_HEIGHT = 720 # Height of the video frame
DETECTION_CONFIDENCE = 0.5 # Confidence threshold for face detection
ZOOM_SIZE = 150  # Size of the zoomed-in face area within the main window

# Face data structure
@dataclass
class FaceData:
    embedding: np.ndarray
    name: str

class FaceRecognitionSystem:
    def __init__(self, target_name: str):
        self.face_detection = mp.solutions.face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=DETECTION_CONFIDENCE
        )
        self.facenet = InceptionResnetV1(pretrained='vggface2').eval()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.facenet = self.facenet.to(self.device)
        logging.info(f"FaceNet running on {self.device}")
        
        self.cap = None
        self.collection = None
        self.known_faces: List[FaceData] = []
        self.target_name = target_name.lower()  # Case-insensitive matching
        self.setup_logging()

    def setup_logging(self):
        """Configure logging with immediate flush"""
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'face_recognition_{datetime.now().strftime("%Y%m%d")}.log', mode='w'),
                logging.StreamHandler(sys.stdout)
            ],
            force=True
        )
        for handler in logging.getLogger().handlers:
            handler.flush()

    def connect_to_mongodb(self) -> bool:
        """Connect to MongoDB synchronously with retry logic"""
        retries = 3
        for attempt in range(retries):
            try:
                client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
                client.server_info()  # Synchronous call to test connection
                self.collection = client["Face"]["Users"]
                logging.info("Connected to MongoDB")
                return True
            except Exception as e:
                logging.warning(f"MongoDB connection attempt {attempt + 1} failed: {e}")
                if attempt < retries - 1:
                    import time
                    time.sleep(2 ** attempt)  # Exponential backoff
        logging.error("Failed to connect to MongoDB after retries")
        return False

    def load_face_data(self) -> bool:
        """Load face embeddings from MongoDB synchronously"""
        if self.collection is None:
            logging.error("No MongoDB collection available")
            return False
            
        try:
            data = list(self.collection.find({}, {"embedding": 1, "name": 1, "_id": 0}))
            self.known_faces = [FaceData(np.array(d["embedding"]), d["name"]) for d in data]
            logging.info(f"Loaded {len(self.known_faces)} face embeddings")
            return bool(self.known_faces)
        except Exception as e:
            logging.error(f"Error loading face data: {e}")
            return False

    def initialize_video(self) -> bool:
        """Initialize video capture with error handling"""
        try:
            logging.debug(f"Attempting to open RTSP stream: {RTSP_URL}")
            self.cap = cv2.VideoCapture(RTSP_URL)
            if not self.cap.isOpened():
                logging.warning("RTSP stream failed - falling back to default webcam")
                self.cap = cv2.VideoCapture(0)
                if not self.cap.isOpened():
                    logging.error("Default webcam failed - exiting")
                    return False
                
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
            
            ret, frame = self.cap.read()
            if not ret:
                logging.error("Failed to read first frame from video source")
                return False
            logging.info("Video capture initialized successfully")
            return True
        except Exception as e:
            logging.error(f"Video initialization failed: {e}")
            return False

    def process_frame(self, frame: np.ndarray) -> Tuple[List, List[str]]:
        """Process frame synchronously with MediaPipe detection and FaceNet recognition"""
        try:
            logging.debug("Processing new frame")
            if frame is None or frame.size == 0:
                logging.error("Received invalid frame")
                return [], []
                
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_detection.process(rgb_frame)
            
            if not results.detections:
                logging.debug("No faces detected in frame")
                return [], []

            face_locations = []
            face_names = []
            THRESHOLD = 1.2
            
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                h, w = frame.shape[:2]
                left = max(0, int(bbox.xmin * w))
                top = max(0, int(bbox.ymin * h))
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                right = min(w, left + width)
                bottom = min(h, top + height)
                
                face_locations.append((top, right, bottom, left))
                
                face_crop = frame[top:bottom, left:right]
                if face_crop.size > 0:
                    face_resized = cv2.resize(face_crop, (160, 160))
                    face_tensor = torch.tensor(face_resized.transpose(2, 0, 1)).float() / 255.0
                    face_tensor = face_tensor.unsqueeze(0).to(self.device)
                    
                    with torch.no_grad():
                        embedding = self.facenet(face_tensor).cpu().numpy()[0]
                    
                    name = "Unknown"
                    min_distance = float('inf')
                    closest_name = None
                    
                    if self.known_faces:
                        for known_face in self.known_faces:
                            distance = np.linalg.norm(embedding - known_face.embedding)
                            logging.debug(f"Distance to {known_face.name}: {distance}")
                            if distance < min_distance:
                                min_distance = distance
                                closest_name = known_face.name
                        if min_distance < THRESHOLD:
                            name = closest_name
                    
                    face_names.append(name)
                    logging.debug(f"Assigned name: {name} (min_distance: {min_distance})")
                    
                    # Zoom in if the name matches the target_name
                    if name.lower() == self.target_name:
                        # Add padding to the crop for better visibility
                        padding = int((right - left) * 0.5)  # 50% padding around face
                        zoom_left = max(0, left - padding)
                        zoom_right = min(w, right + padding)
                        zoom_top = max(0, top - padding)
                        zoom_bottom = min(h, bottom + padding)
                        
                        zoomed_face = frame[zoom_top:zoom_bottom, zoom_left:zoom_right]
                        if zoomed_face.size > 0:
                            # Resize to fit the zoom area (top-right corner)
                            zoomed_face = cv2.resize(zoomed_face, (ZOOM_SIZE, ZOOM_SIZE))
                            # Place zoomed face in the top-right corner of the main frame
                            frame[0:ZOOM_SIZE, FRAME_WIDTH-ZOOM_SIZE:FRAME_WIDTH] = zoomed_face
                            # Add a label above the zoomed area
                            cv2.putText(frame, f"Zoom: {name}", 
                                        (FRAME_WIDTH-ZOOM_SIZE, 20), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            logging.debug(f"Detected {len(face_locations)} faces")
            return face_locations, face_names
        except Exception as e:
            logging.error(f"Frame processing error: {e}")
            return [], []

    def run(self):
        """Main processing loop - synchronous"""
        logging.info(f"Starting face recognition system with target name: {self.target_name}")
        
        try:
            if self.connect_to_mongodb():
                self.load_face_data()
            else:
                logging.warning("Proceeding without MongoDB data")
        except Exception as e:
            logging.error(f"MongoDB setup failed: {e}")
            logging.warning("Continuing without MongoDB data")

        if not self.initialize_video():
            logging.error("Failed to initialize video - exiting")
            return

        frame_count = 0
        try:
            logging.info("Entering main video processing loop")
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    logging.warning("Failed to capture frame, attempting to reconnect...")
                    self.cap.release()
                    if not self.initialize_video():
                        logging.error("Reconnection failed - exiting")
                        break
                    continue

                frame_count += 1
                if frame_count % 30 == 0:
                    logging.debug(f"Processing frame {frame_count}")

                face_locations, face_names = self.process_frame(frame)
                
                for (top, right, bottom, left), name in zip(face_locations, face_names):
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    cv2.putText(frame, name, (left, bottom + 20), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                cv2.imshow('Face Recognition', frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logging.info("User requested shutdown")
                    break

        except Exception as e:
            logging.error(f"Runtime error: {e}")
        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        self.face_detection.close()
        logging.info("System cleanup completed")
        for handler in logging.getLogger().handlers:
            handler.flush()

def main():
    try:
        # Get target name from user input
        target_name = input("Enter the name to zoom in on: ").strip()
        logging.info(f"Starting application with target name: {target_name}")
        system = FaceRecognitionSystem(target_name)
        system.run()
    except Exception as e:
        logging.error(f"Main function error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()