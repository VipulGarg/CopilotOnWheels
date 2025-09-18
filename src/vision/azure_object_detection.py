"""
Azure Computer Vision Object Detection module for Robot Vision
Uses Azure Cognitive Services for cloud-based object detection and analysis
"""

import cv2
import numpy as np
import logging
import json
import io
import base64
from typing import List, Dict, Optional, Tuple
import os
from dotenv import load_dotenv

try:
    from azure.cognitiveservices.vision.computervision import ComputerVisionClient
    from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
    from msrest.authentication import CognitiveServicesCredentials
    import requests
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False
    logging.warning("Azure Cognitive Services not available - install azure-cognitiveservices-vision-computervision")

# Load environment variables
load_dotenv()

class AzureObjectDetector:
    """Azure Computer Vision based object detection"""
    
    def __init__(self, confidence_threshold: float = 0.5):
        """
        Initialize Azure object detector
        
        Args:
            confidence_threshold: Minimum confidence for detection
        """
        self.confidence_threshold = confidence_threshold
        self.client = None
        self.endpoint = None
        self.key = None
        self.is_initialized = False
        
        # Initialize Azure client
        self._initialize_azure_client()
    
    def _initialize_azure_client(self):
        """Initialize Azure Computer Vision client"""
        if not AZURE_AVAILABLE:
            logging.error("Azure Cognitive Services SDK not available")
            return
        
        try:
            # Get Azure credentials from environment
            self.endpoint = os.getenv("AZURE_COMPUTER_VISION_ENDPOINT")
            self.key = os.getenv("AZURE_COMPUTER_VISION_KEY")
            
            if not self.endpoint or not self.key:
                logging.error("Azure Computer Vision credentials not found in environment variables")
                logging.error("Please set AZURE_COMPUTER_VISION_ENDPOINT and AZURE_COMPUTER_VISION_KEY")
                return
            
            # Create client
            credentials = CognitiveServicesCredentials(self.key)
            self.client = ComputerVisionClient(self.endpoint, credentials)
            
            self.is_initialized = True
            logging.info("Azure Computer Vision client initialized successfully")
            
        except Exception as e:
            logging.error(f"Azure Computer Vision initialization failed: {e}")
            self.is_initialized = False
    
    def detect_objects(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect objects in frame using Azure Computer Vision
        
        Args:
            frame: Input image frame
            
        Returns:
            List of detected objects with bounding boxes and confidence
        """
        if not self.is_initialized or self.client is None:
            logging.error("Azure object detector not initialized")
            return []
        
        try:
            # Convert frame to JPEG bytes
            image_bytes = self._frame_to_bytes(frame)
            if image_bytes is None:
                return []
            
            # Analyze image with Azure
            analysis = self.client.analyze_image_in_stream(
                image_bytes,
                visual_features=["Objects", "Categories", "Tags"],
                language="en"
            )
            
            # Log detailed analysis information
            logging.info("🔍 Azure Analysis Deep Details:")
            logging.info(f"  Frame shape: {frame.shape} (H×W×C: {frame.shape[0]}×{frame.shape[1]}×{frame.shape[2] if len(frame.shape) > 2 else 'N/A'})")
            logging.info(f"  Frame dtype: {frame.dtype}")
            logging.info(f"  Frame size: {frame.size} pixels")
            logging.info(f"  Frame memory: {frame.nbytes} bytes")
            
            # Log analysis object details
            if hasattr(analysis, 'objects') and analysis.objects:
                logging.info(f"  📦 Objects detected: {len(analysis.objects)}")
                for i, obj in enumerate(analysis.objects):
                    logging.info(obj)
                    rect = obj.rectangle
                    logging.info(f"    {i+1}. Object: '{obj.object_property}'")
                    logging.info(f"       Confidence: {obj.confidence:.4f}")
                    logging.info(f"       Rectangle: x={rect.x}, y={rect.y}, w={rect.w}, h={rect.h}")
                    logging.info(f"       Area: {rect.w * rect.h} pixels")
                    logging.info(f"       Center: ({rect.x + rect.w//2}, {rect.y + rect.h//2})")
            else:
                logging.info("  📦 No objects detected by Azure")
            
            # Log tags information
            if hasattr(analysis, 'tags') and analysis.tags:
                logging.info(f"  🏷️  Tags detected: {len(analysis.tags)}")
                high_conf_tags = [tag for tag in analysis.tags if tag.confidence >= 0.7]
                logging.info(f"  🏷️  High confidence tags (≥0.7): {len(high_conf_tags)}")
                for i, tag in enumerate(analysis.tags[:10]):  # Show first 10 tags
                    logging.info(tag)
                    logging.info(f"    {i+1}. Tag: '{tag.name}' (confidence: {tag.confidence:.4f})")
            else:
                logging.info("  🏷️  No tags detected by Azure")
            
            # Log categories information
            if hasattr(analysis, 'categories') and analysis.categories:
                logging.info(f"  📂 Categories detected: {len(analysis.categories)}")
                for i, cat in enumerate(analysis.categories):
                    logging.info(cat)
                    if cat.score > 0.1:  # Only log significant categories
                        logging.info(f"    {i+1}. Category: '{cat.name}' (score: {cat.score:.4f})")
            else:
                logging.info("  📂 No categories detected by Azure")
            
            # Convert Azure results to our standard format
            detections = self._convert_azure_results(analysis, frame.shape)
            
            # Filter by confidence threshold
            filtered_detections = [
                det for det in detections 
                if det['confidence'] >= self.confidence_threshold
            ]
            
            logging.info(f"Azure detected {len(filtered_detections)} objects above threshold")
            logging.info(filtered_detections)
            return filtered_detections
            
        except Exception as e:
            logging.error(f"Azure object detection error: {e}")
            return []
    
    def _frame_to_bytes(self, frame: np.ndarray) -> Optional[io.BytesIO]:
        """Convert OpenCV frame to bytes for Azure API"""
        try:
            logging.debug("🖼️  Frame to bytes conversion:")
            logging.debug(f"  Input frame shape: {frame.shape}")
            logging.debug(f"  Input frame dtype: {frame.dtype}")
            logging.debug(f"  Input frame min/max values: {frame.min()}/{frame.max()}")
            
            # Encode frame as JPEG
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, 90]  # High quality JPEG
            success, buffer = cv2.imencode('.jpg', frame, encode_params)
            
            if not success:
                logging.error("Failed to encode frame as JPEG")
                return None
            
            logging.debug(f"  JPEG encoding successful")
            logging.debug(f"  Encoded buffer size: {len(buffer)} bytes")
            logging.debug(f"  Compression ratio: {frame.nbytes / len(buffer):.2f}:1")
            
            # Convert to BytesIO
            image_bytes = io.BytesIO(buffer.tobytes())
            image_bytes.seek(0)  # Reset pointer to beginning
            
            logging.debug(f"  BytesIO stream created, size: {len(image_bytes.getvalue())} bytes")
            return image_bytes
            
        except Exception as e:
            logging.error(f"Frame to bytes conversion error: {e}")
            return None
    
    def _convert_azure_results(self, analysis, frame_shape: Tuple) -> List[Dict]:
        """Convert Azure analysis results to standard detection format"""
        detections = []
        frame_height, frame_width = frame_shape[:2]
        
        logging.debug("🔄 Converting Azure results to standard format:")
        logging.debug(f"  Target frame dimensions: {frame_width}×{frame_height}")
        
        # Process object detections
        if hasattr(analysis, 'objects') and analysis.objects:
            logging.debug(f"  Processing {len(analysis.objects)} Azure objects...")
            for i, obj in enumerate(analysis.objects):
                # Azure provides bounding box as rectangle
                rect = obj.rectangle
                x1, y1 = rect.x, rect.y
                x2, y2 = rect.x + rect.w, rect.y + rect.h
                
                detection = {
                    'class_id': 0,  # Azure doesn't provide class IDs like YOLO
                    'class_name': obj.object_property.lower(),
                    'confidence': obj.confidence,
                    'bbox': (x1, y1, x2, y2),
                    'center': ((x1 + x2) // 2, (y1 + y2) // 2),
                    'width': rect.w,
                    'height': rect.h,
                    'area': rect.w * rect.h,
                    'source': 'azure_objects'
                }
                detections.append(detection)
                
                logging.debug(f"    Object {i+1}: '{obj.object_property}' -> '{obj.object_property.lower()}'")
                logging.debug(f"      Original rect: {rect.x}, {rect.y}, {rect.w}, {rect.h}")
                logging.debug(f"      Converted bbox: {detection['bbox']}")
                logging.debug(f"      Center: {detection['center']}")
                logging.debug(f"      Area: {detection['area']} pixels")
        else:
            logging.debug("  No Azure objects to process")
        
        # Also process tags as potential objects (with lower confidence)
        if hasattr(analysis, 'tags') and analysis.tags:
            high_conf_tags = [tag for tag in analysis.tags if tag.confidence >= 0.7]
            logging.debug(f"  Processing {len(high_conf_tags)} high-confidence tags (≥0.7)...")
            
            for i, tag in enumerate(high_conf_tags):
                # For tags, we don't have bounding boxes, so create a general detection
                # This is useful for identifying objects that might not have been boxed
                adjusted_confidence = tag.confidence * 0.8  # Reduce confidence since no bbox
                
                detection = {
                    'class_id': 0,
                    'class_name': tag.name.lower(),
                    'confidence': adjusted_confidence,
                    'bbox': (0, 0, frame_width, frame_height),  # Full frame
                    'center': (frame_width // 2, frame_height // 2),
                    'width': frame_width,
                    'height': frame_height,
                    'area': frame_width * frame_height,
                    'source': 'azure_tags'
                }
                detections.append(detection)
                
                logging.debug(f"    Tag {i+1}: '{tag.name}' -> '{tag.name.lower()}'")
                logging.debug(f"      Original confidence: {tag.confidence:.4f}")
                logging.debug(f"      Adjusted confidence: {adjusted_confidence:.4f}")
                logging.debug(f"      Full-frame bbox: {detection['bbox']}")
        else:
            logging.debug("  No high-confidence tags to process")
        
        logging.debug(f"  ✅ Conversion complete: {len(detections)} total detections created")
        return detections
    
    def find_object_by_name(self, frame: np.ndarray, object_name: str) -> Optional[Dict]:
        """
        Find specific object in frame using Azure
        
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
            if (object_name_lower in class_name or 
                class_name in object_name_lower or
                self._are_similar_objects(object_name_lower, class_name)):
                matching_objects.append(detection)
        
        # Prefer objects with bounding boxes (from object detection) over tags
        bbox_objects = [obj for obj in matching_objects if obj['source'] == 'azure_objects']
        if bbox_objects:
            return max(bbox_objects, key=lambda x: x['confidence'])
        
        # Fall back to tag-based detections
        if matching_objects:
            return max(matching_objects, key=lambda x: x['confidence'])
        
        return None
    
    def _are_similar_objects(self, name1: str, name2: str) -> bool:
        """Check if two object names refer to similar things"""
        # Define common object synonyms
        synonyms = {
            'cup': ['mug', 'glass', 'tumbler'],
            'bottle': ['container', 'flask'],
            'person': ['human', 'people', 'man', 'woman'],
            'car': ['vehicle', 'automobile'],
            'phone': ['mobile', 'smartphone', 'cellphone'],
            'laptop': ['computer', 'notebook'],
            'tv': ['television', 'monitor', 'screen'],
            'book': ['magazine', 'publication'],
            'chair': ['seat'],
            'table': ['desk'],
            'bag': ['backpack', 'handbag', 'purse'],
            'food': ['meal', 'snack'],
            'drink': ['beverage'],
        }
        
        # Check if either name is a synonym of the other
        for base_word, synonym_list in synonyms.items():
            if ((name1 == base_word and name2 in synonym_list) or
                (name2 == base_word and name1 in synonym_list) or
                (name1 in synonym_list and name2 == base_word) or
                (name2 in synonym_list and name1 == base_word)):
                return True
        
        return False
    
    def get_detailed_analysis(self, frame: np.ndarray) -> Dict:
        """
        Get detailed analysis of the image including descriptions
        
        Args:
            frame: Input image frame
            
        Returns:
            Detailed analysis results
        """
        if not self.is_initialized or self.client is None:
            return {"success": False, "message": "Azure detector not initialized"}
        
        try:
            image_bytes = self._frame_to_bytes(frame)
            if image_bytes is None:
                return {"success": False, "message": "Failed to process image"}
            
            # Get comprehensive analysis
            analysis = self.client.analyze_image_in_stream(
                image_bytes,
                visual_features=["Objects", "Categories", "Tags", "Description", "Color"],
                language="en"
            )
            
            # Extract information
            result = {
                "success": True,
                "descriptions": [],
                "tags": [],
                "objects": [],
                "colors": {},
                "categories": []
            }
            
            # Descriptions
            if hasattr(analysis, 'description') and analysis.description.captions:
                result["descriptions"] = [
                    {"text": cap.text, "confidence": cap.confidence}
                    for cap in analysis.description.captions
                ]
            
            # Tags
            if hasattr(analysis, 'tags'):
                result["tags"] = [
                    {"name": tag.name, "confidence": tag.confidence}
                    for tag in analysis.tags[:10]  # Top 10 tags
                ]
            
            # Objects
            if hasattr(analysis, 'objects'):
                result["objects"] = [
                    {
                        "name": obj.object_property,
                        "confidence": obj.confidence,
                        "bbox": (obj.rectangle.x, obj.rectangle.y, 
                                obj.rectangle.w, obj.rectangle.h)
                    }
                    for obj in analysis.objects
                ]
            
            # Colors
            if hasattr(analysis, 'color'):
                result["colors"] = {
                    "dominant_foreground": analysis.color.dominant_color_foreground,
                    "dominant_background": analysis.color.dominant_color_background,
                    "accent_color": analysis.color.accent_color
                }
            
            # Categories
            if hasattr(analysis, 'categories'):
                result["categories"] = [
                    {"name": cat.name, "confidence": cat.score}
                    for cat in analysis.categories if cat.score > 0.1
                ]
            
            return result
            
        except Exception as e:
            logging.error(f"Azure detailed analysis error: {e}")
            return {"success": False, "message": f"Analysis failed: {e}"}
    
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
            # Skip tag-based detections (they cover the whole frame)
            if detection.get('source') == 'azure_tags':
                continue
                
            x1, y1, x2, y2 = detection['bbox']
            confidence = detection['confidence']
            class_name = detection['class_name']
            
            # Use different colors for Azure vs local detection
            color = (255, 0, 255)  # Magenta for Azure
            
            # Draw bounding box
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"Azure: {class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(result_frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(result_frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
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
            return "Azure detected no objects"
        
        # Separate object detections from tag detections
        object_detections = [d for d in detections if d.get('source') == 'azure_objects']
        tag_detections = [d for d in detections if d.get('source') == 'azure_tags']
        
        summary_parts = []
        
        # Summarize object detections (with bounding boxes)
        if object_detections:
            object_counts = {}
            for detection in object_detections:
                class_name = detection['class_name']
                object_counts[class_name] = object_counts.get(class_name, 0) + 1
            
            items = []
            for obj_name, count in object_counts.items():
                if count == 1:
                    items.append(f"a {obj_name}")
                else:
                    items.append(f"{count} {obj_name}s")
            
            if len(items) == 1:
                summary_parts.append(f"I can see {items[0]}")
            elif len(items) == 2:
                summary_parts.append(f"I can see {items[0]} and {items[1]}")
            else:
                summary_parts.append(f"I can see {', '.join(items[:-1])}, and {items[-1]}")
        
        # Add high-confidence tags if no objects detected
        if not object_detections and tag_detections:
            high_conf_tags = [d['class_name'] for d in tag_detections[:3]]
            summary_parts.append(f"The image appears to contain: {', '.join(high_conf_tags)}")
        
        return ". ".join(summary_parts) if summary_parts else "Azure analysis completed but found no clear objects"


def test_azure_detection():
    """Test Azure object detection functionality"""
    try:
        detector = AzureObjectDetector()
        print(f"Azure detector initialized: {detector.is_initialized}")
        
        if detector.is_initialized:
            print("Azure Computer Vision detector ready for use")
            print("Make sure to set AZURE_COMPUTER_VISION_ENDPOINT and AZURE_COMPUTER_VISION_KEY environment variables")
        else:
            print("Azure detector initialization failed")
            print("Check your Azure credentials and internet connection")
    
    except Exception as e:
        print(f"Azure detection test failed: {e}")


if __name__ == "__main__":
    test_azure_detection()