"""
This script populates a MongoDB database with face embeddings extracted from images in a specified folder.
It uses MediaPipe for face detection and FaceNet for generating face embeddings.
Classes:
    FaceDatabasePopulator: Handles the process of connecting to MongoDB, extracting face embeddings, 
                           and storing them in the database.
Functions:
    main(): Configures the MongoDB URI and folder path, initializes the FaceDatabasePopulator, 
            and triggers the database population process.
Class FaceDatabasePopulator:
    Methods:
        __init__(mongo_uri: str, folder_path: str):
            Initializes the populator with the MongoDB URI and the folder path containing images.
        connect_to_mongodb() -> bool:
            Connects to the MongoDB database and initializes the collection for storing face embeddings.
        get_face_embedding(image_path: str) -> Optional[np.ndarray]:
            Extracts a face embedding from a given image using MediaPipe for face detection 
            and FaceNet for embedding generation.
        populate_database():
            Scans the specified folder for images, processes each image to extract face embeddings, 
            and stores the embeddings in the MongoDB database.
        cleanup():
            Releases resources used by the MediaPipe face detection module.
Usage:
    1. Configure the `mongo_uri` with your MongoDB connection string.
    2. Set the `folder_path` to the directory containing the images.
    3. Run the script to populate the database with face embeddings.
"""
import cv2
import mediapipe as mp
import numpy as np
from facenet_pytorch import InceptionResnetV1
import torch
from pymongo import MongoClient
from pathlib import Path
import logging
from typing import Optional
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'populate_faces_{datetime.now().strftime("%Y%m%d")}.log'),
        logging.StreamHandler()
    ]
)

class FaceDatabasePopulator:
    def __init__(self, mongo_uri: str, folder_path: str):
        """Initialize the populator with MongoDB URI and folder path."""
        self.mongo_uri = mongo_uri
        self.folder_path = Path(folder_path)
        self.face_detection = mp.solutions.face_detection.FaceDetection(
            model_selection=1,  # Full-range model
            min_detection_confidence=0.5
        )
        self.facenet = InceptionResnetV1(pretrained='vggface2').eval()
        if torch.cuda.is_available():
            self.facenet = self.facenet.cuda()
            logging.info("Using GPU for FaceNet")
        else:
            logging.info("Using CPU for FaceNet")
        self.collection = None

    def connect_to_mongodb(self) -> bool:
        """Connect to MongoDB."""
        try:
            client = MongoClient(self.mongo_uri, serverSelectionTimeoutMS=5000)
            client.server_info()  # Test connection
            self.collection = client["Face"]["Users"]
            logging.info("Connected to MongoDB")
            return True
        except Exception as e:
            logging.error(f"Failed to connect to MongoDB: {e}")
            return False

    def get_face_embedding(self, image_path: str) -> Optional[np.ndarray]:
        """Extract face embedding from an image."""
        try:
            # Read and preprocess image
            img = cv2.imread(str(image_path))
            if img is None:
                logging.warning(f"Could not read image: {image_path}")
                return None
            
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Detect face with MediaPipe
            results = self.face_detection.process(rgb_img)
            if not results.detections or len(results.detections) == 0:
                logging.warning(f"No face detected in {image_path}")
                return None
            
            # Assume single face; take the first detection
            detection = results.detections[0]
            bbox = detection.location_data.relative_bounding_box
            h, w = img.shape[:2]
            left = int(bbox.xmin * w)
            top = int(bbox.ymin * h)
            width = int(bbox.width * w)
            height = int(bbox.height * h)
            right = left + width
            bottom = top + height
            
            # Crop face with padding
            face_crop = img[max(0, top-20):bottom+20, max(0, left-20):right+20]
            if face_crop.size == 0:
                logging.warning(f"Invalid face crop in {image_path}")
                return None
            
            # Resize to 160x160 for FaceNet
            face_resized = cv2.resize(face_crop, (160, 160))
            face_tensor = torch.tensor(face_resized.transpose(2, 0, 1)).float() / 255.0
            face_tensor = face_tensor.unsqueeze(0)
            if torch.cuda.is_available():
                face_tensor = face_tensor.cuda()
            
            # Generate embedding
            with torch.no_grad():
                embedding = self.facenet(face_tensor).cpu().numpy()[0]
            return embedding
        
        except Exception as e:
            logging.error(f"Error processing {image_path}: {e}")
            return None

    def populate_database(self):
        """Scan folder and populate MongoDB with face embeddings."""
        if not self.connect_to_mongodb():
            return
        
        # Supported image extensions
        image_extensions = ('.jpg', '.jpeg', '.png')
        image_files = [f for f in self.folder_path.iterdir() if f.suffix.lower() in image_extensions]
        
        if not image_files:
            logging.warning(f"No images found in {self.folder_path}")
            return
        
        for image_path in image_files:
            name = image_path.stem  # Use filename without extension as name
            logging.info(f"Processing {image_path}")
            
            embedding = self.get_face_embedding(image_path)
            if embedding is not None:
                # Store in MongoDB
                try:
                    self.collection.update_one(
                        {"name": name},
                        {"$set": {"embedding": embedding.tolist()}},
                        upsert=True
                    )
                    logging.info(f"Stored face for {name} in database")
                except Exception as e:
                    logging.error(f"Failed to store {name} in database: {e}")
            else:
                logging.warning(f"Skipping {name} due to processing failure")

    def cleanup(self):
        """Clean up resources."""
        self.face_detection.close()
        logging.info("Cleanup completed")

def main():
    # Configuration
    mongo_uri = "mongodb+srv://mina23:01555758130@cluster0.22ogf.mongodb.net/?retryWrites=true&w=majority"
    folder_path = r"C:\Users\User\Desktop\Python test\Faces"  # Replace with your folder path
    
    populator = FaceDatabasePopulator(mongo_uri, folder_path)
    populator.populate_database()
    populator.cleanup()

if __name__ == "__main__":
    main()