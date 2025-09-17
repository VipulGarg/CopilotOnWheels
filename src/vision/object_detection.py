"""
Object Detection module for Robot Vision
Uses YOLO for real-time object detection and identification
"""

import cv2
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional
import os
import urllib.request

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logging.warning("Ultralytics YOLO not available - using OpenCV DNN fallback")

class ObjectDetector:
    """Object detection using YOLO or OpenCV DNN"""
    
    def __init__(self, model_type: str = "yolo", confidence_threshold: float = 0.5):
        """
        Initialize object detector
        
        Args:
            model_type: "yolo" or "opencv_dnn"
            confidence_threshold: Minimum confidence for detection
        """
        self.confidence_threshold = confidence_threshold
        self.model_type = model_type
        self.model = None
        self.classes = []
        self.is_initialized = False
        
        # Initialize the appropriate model
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the detection model"""
        if self.model_type == "yolo" and YOLO_AVAILABLE:
            self._initialize_yolo()
        else:
            self._initialize_opencv_dnn()
    
    def _initialize_yolo(self):
        """Initialize YOLO model"""
        try:
            # Use YOLOv8 nano model for speed on Raspberry Pi
            self.model = YOLO('yolov8n.pt')
            
            # YOLO classes (COCO dataset)
            self.classes = [
                'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
                'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
                'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
                'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
                'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
                'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
                'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
                'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
                'toothbrush'
            ]
            
            self.model_type = "yolo"
            self.is_initialized = True
            logging.info("YOLO model initialized successfully")
            
        except Exception as e:
            logging.error(f"YOLO initialization failed: {e}")
            self._initialize_opencv_dnn()
    
    def _initialize_opencv_dnn(self):
        """Initialize OpenCV DNN as fallback"""
        try:
            # Download YOLOv4-tiny weights and config if not present
            weights_path = "yolov4-tiny.weights"
            config_path = "yolov4-tiny.cfg"
            classes_path = "coco.names"
            
            self._download_yolo_files(weights_path, config_path, classes_path)
            
            # Load the network
            self.model = cv2.dnn.readNet(weights_path, config_path)
            
            # Load class names
            with open(classes_path, 'r') as f:
                self.classes = [line.strip() for line in f.readlines()]
            
            self.model_type = "opencv_dnn"
            self.is_initialized = True
            logging.info("OpenCV DNN model initialized successfully")
            
        except Exception as e:
            logging.error(f"OpenCV DNN initialization failed: {e}")
            self.is_initialized = False
    
    def _download_yolo_files(self, weights_path: str, config_path: str, classes_path: str):
        """Download YOLO files if they don't exist"""
        files_to_download = [
            (weights_path, "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights"),
            (config_path, "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg"),
            (classes_path, "https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names")
        ]
        
        for file_path, url in files_to_download:
            if not os.path.exists(file_path):
                logging.info(f"Downloading {file_path}...")
                try:
                    urllib.request.urlretrieve(url, file_path)
                    logging.info(f"Downloaded {file_path}")
                except Exception as e:
                    logging.error(f"Failed to download {file_path}: {e}")
                    raise
    
    def detect_objects(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect objects in frame
        
        Args:
            frame: Input image frame
            
        Returns:
            List of detected objects with bounding boxes and confidence
        """
        if not self.is_initialized or self.model is None:
            logging.error("Object detector not initialized")
            return []
        
        try:
            if self.model_type == "yolo":
                return self._detect_with_yolo(frame)
            else:
                return self._detect_with_opencv_dnn(frame)
        
        except Exception as e:
            logging.error(f"Object detection error: {e}")
            return []
    
    def _detect_with_yolo(self, frame: np.ndarray) -> List[Dict]:
        """Detect objects using YOLO"""
        results = self.model(frame, verbose=False)
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    
                    if confidence >= self.confidence_threshold:
                        detections.append({
                            'class_id': class_id,
                            'class_name': self.classes[class_id],
                            'confidence': confidence,
                            'bbox': (x1, y1, x2, y2),
                            'center': ((x1 + x2) // 2, (y1 + y2) // 2),
                            'width': x2 - x1,
                            'height': y2 - y1,
                            'area': (x2 - x1) * (y2 - y1)
                        })
        
        return detections
    
    def _detect_with_opencv_dnn(self, frame: np.ndarray) -> List[Dict]:
        """Detect objects using OpenCV DNN"""
        height, width = frame.shape[:2]
        
        # Create blob from image
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        self.model.setInput(blob)
        
        # Run inference
        outputs = self.model.forward()
        
        boxes = []
        confidences = []
        class_ids = []
        
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence >= self.confidence_threshold:
                    # Get bounding box coordinates
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    x = int(center_x - w/2)
                    y = int(center_y - h/2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        # Apply non-maximum suppression
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, 0.4)
        
        detections = []
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                x1, y1, x2, y2 = x, y, x + w, y + h
                
                detections.append({
                    'class_id': class_ids[i],
                    'class_name': self.classes[class_ids[i]],
                    'confidence': confidences[i],
                    'bbox': (x1, y1, x2, y2),
                    'center': ((x1 + x2) // 2, (y1 + y2) // 2),
                    'width': w,
                    'height': h,
                    'area': w * h
                })
        
        return detections
    
    def find_object_by_name(self, frame: np.ndarray, object_name: str) -> Optional[Dict]:
        """
        Find specific object in frame
        
        Args:
            frame: Input image frame
            object_name: Name of object to find
            
        Returns:
            Best matching detection or None
        """
        detections = self.detect_objects(frame)
        
        # Find objects matching the name (case-insensitive, partial match)
        matching_objects = []
        object_name_lower = object_name.lower()
        
        for detection in detections:
            class_name = detection['class_name'].lower()
            
            # Direct match or partial match
            if object_name_lower in class_name or class_name in object_name_lower:
                matching_objects.append(detection)
        
        # Return the detection with highest confidence
        if matching_objects:
            return max(matching_objects, key=lambda x: x['confidence'])
        
        return None
    
    def get_object_position(self, detection: Dict, frame_width: int, frame_height: int) -> str:
        """
        Get relative position description of object
        
        Args:
            detection: Object detection result
            frame_width: Frame width
            frame_height: Frame height
            
        Returns:
            Position description (e.g., "center", "left", "top-right")
        """
        center_x, center_y = detection['center']
        
        # Determine horizontal position
        if center_x < frame_width * 0.33:
            h_pos = "left"
        elif center_x > frame_width * 0.67:
            h_pos = "right"
        else:
            h_pos = "center"
        
        # Determine vertical position
        if center_y < frame_height * 0.33:
            v_pos = "top"
        elif center_y > frame_height * 0.67:
            v_pos = "bottom"
        else:
            v_pos = "middle"
        
        # Combine positions
        if h_pos == "center" and v_pos == "middle":
            return "center"
        elif h_pos == "center":
            return v_pos
        elif v_pos == "middle":
            return h_pos
        else:
            return f"{v_pos}-{h_pos}"
    
    def draw_detections(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        Draw bounding boxes and labels on frame
        
        Args:
            frame: Input image frame
            detections: List of detections
            
        Returns:
            Frame with drawn detections
        """
        result_frame = frame.copy()
        
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            confidence = detection['confidence']
            class_name = detection['class_name']
            
            # Draw bounding box
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(result_frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), (0, 255, 0), -1)
            cv2.putText(result_frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        return result_frame
    
    def get_detection_summary(self, detections: List[Dict]) -> str:
        """
        Get a text summary of detections
        
        Args:
            detections: List of detections
            
        Returns:
            Human-readable summary
        """
        if not detections:
            return "No objects detected"
        
        # Count objects by class
        object_counts = {}
        for detection in detections:
            class_name = detection['class_name']
            object_counts[class_name] = object_counts.get(class_name, 0) + 1
        
        # Create summary
        if len(object_counts) == 1 and sum(object_counts.values()) == 1:
            # Single object
            obj_name = list(object_counts.keys())[0]
            return f"I can see a {obj_name}"
        
        # Multiple objects
        items = []
        for obj_name, count in object_counts.items():
            if count == 1:
                items.append(f"a {obj_name}")
            else:
                items.append(f"{count} {obj_name}s")
        
        if len(items) == 1:
            return f"I can see {items[0]}"
        elif len(items) == 2:
            return f"I can see {items[0]} and {items[1]}"
        else:
            return f"I can see {', '.join(items[:-1])}, and {items[-1]}"


def test_object_detection():
    """Test object detection functionality"""
    try:
        detector = ObjectDetector()
        print(f"Detector initialized: {detector.is_initialized}")
        print(f"Model type: {detector.model_type}")
        print(f"Available classes: {len(detector.classes)}")
        
        if detector.is_initialized:
            print("Object detector ready for use")
        else:
            print("Object detector initialization failed")
    
    except Exception as e:
        print(f"Object detection test failed: {e}")


if __name__ == "__main__":
    test_object_detection()