"""
Hybrid Object Detection module for Robot Vision
Combines local YOLO detection with Azure Computer Vision for enhanced accuracy
"""

import cv2
import numpy as np
import logging
from typing import List, Dict, Optional, Tuple
import os
from dotenv import load_dotenv

from .object_detection import ObjectDetector as LocalObjectDetector
from .azure_object_detection import AzureObjectDetector

# Load environment variables
load_dotenv()

class HybridObjectDetector:
    """Object detection using both local YOLO and Azure Computer Vision"""
    
    def __init__(self, confidence_threshold: float = 0.5, prefer_local: bool = True):
        """
        Initialize hybrid object detector
        
        Args:
            confidence_threshold: Minimum confidence for detection
            prefer_local: If True, try local detection first, Azure as fallback
        """
        self.confidence_threshold = confidence_threshold
        self.prefer_local = prefer_local
        
        # Initialize detectors
        self.local_detector = None
        self.azure_detector = None
        self.is_initialized = False
        
        self._initialize_detectors()
    
    def _initialize_detectors(self):
        """Initialize both detection systems"""
        local_available = False
        azure_available = False
        
        # Initialize local detector
        try:
            self.local_detector = LocalObjectDetector(confidence_threshold=self.confidence_threshold)
            local_available = self.local_detector.is_initialized
            if local_available:
                logging.info("Local YOLO detector initialized successfully")
        except Exception as e:
            logging.warning(f"Local detector initialization failed: {e}")
        
        # Initialize Azure detector
        try:
            self.azure_detector = AzureObjectDetector(confidence_threshold=self.confidence_threshold)
            azure_available = self.azure_detector.is_initialized
            if azure_available:
                logging.info("Azure detector initialized successfully")
        except Exception as e:
            logging.warning(f"Azure detector initialization failed: {e}")
        
        # Set initialization status
        self.is_initialized = local_available or azure_available
        
        if self.is_initialized:
            mode = []
            if local_available:
                mode.append("Local YOLO")
            if azure_available:
                mode.append("Azure")
            logging.info(f"Hybrid detector initialized with: {', '.join(mode)}")
        else:
            logging.error("No detection systems available")
    
    def detect_objects(self, frame: np.ndarray, method: str = "auto") -> List[Dict]:
        """
        Detect objects in frame using specified method or both
        
        Args:
            frame: Input image frame
            method: "local", "azure", "both", or "auto" (uses preference)
            
        Returns:
            List of detected objects with bounding boxes and confidence
        """
        if not self.is_initialized:
            logging.error("Hybrid object detector not initialized")
            return []
        
        try:
            if method == "auto":
                method = "local" if self.prefer_local and self.local_detector else "azure"
            
            detections = []
            
            if method == "local" and self.local_detector and self.local_detector.is_initialized:
                detections = self._detect_local(frame)
                
            elif method == "azure" and self.azure_detector and self.azure_detector.is_initialized:
                detections = self._detect_azure(frame)
                
            elif method == "both":
                detections = self._detect_both(frame)
            
            else:
                # Fallback to available detector
                if self.local_detector and self.local_detector.is_initialized:
                    detections = self._detect_local(frame)
                elif self.azure_detector and self.azure_detector.is_initialized:
                    detections = self._detect_azure(frame)
            
            logging.info(f"Hybrid detection ({method}): found {len(detections)} objects")
            return detections
            
        except Exception as e:
            logging.error(f"Hybrid object detection error: {e}")
            return []
    
    def _detect_local(self, frame: np.ndarray) -> List[Dict]:
        """Detect objects using local YOLO"""
        detections = self.local_detector.detect_objects(frame)
        # Add source information
        for detection in detections:
            detection['source'] = 'local_yolo'
        return detections
    
    def _detect_azure(self, frame: np.ndarray) -> List[Dict]:
        """Detect objects using Azure"""
        detections = self.azure_detector.detect_objects(frame)
        # Source is already added by Azure detector
        return detections
    
    def _detect_both(self, frame: np.ndarray) -> List[Dict]:
        """Detect objects using both methods and combine results"""
        all_detections = []
        
        # Get local detections
        if self.local_detector and self.local_detector.is_initialized:
            local_detections = self._detect_local(frame)
            all_detections.extend(local_detections)
        
        # Get Azure detections
        if self.azure_detector and self.azure_detector.is_initialized:
            azure_detections = self._detect_azure(frame)
            all_detections.extend(azure_detections)
        
        # Remove duplicates and merge similar detections
        merged_detections = self._merge_similar_detections(all_detections)
        
        return merged_detections
    
    def _merge_similar_detections(self, detections: List[Dict]) -> List[Dict]:
        """Merge similar detections from different sources"""
        if len(detections) <= 1:
            return detections
        
        merged = []
        used_indices = set()
        
        for i, detection1 in enumerate(detections):
            if i in used_indices:
                continue
            
            # Find similar detections
            similar_detections = [detection1]
            used_indices.add(i)
            
            for j, detection2 in enumerate(detections[i+1:], i+1):
                if j in used_indices:
                    continue
                
                if self._are_detections_similar(detection1, detection2):
                    similar_detections.append(detection2)
                    used_indices.add(j)
            
            # Merge similar detections
            if len(similar_detections) > 1:
                merged_detection = self._merge_detection_group(similar_detections)
            else:
                merged_detection = similar_detections[0]
            
            merged.append(merged_detection)
        
        return merged
    
    def _are_detections_similar(self, det1: Dict, det2: Dict) -> bool:
        """Check if two detections refer to the same object"""
        # Check if object names are similar
        name1 = det1['class_name'].lower()
        name2 = det2['class_name'].lower()
        
        if name1 != name2:
            # Check for synonyms
            if not self._are_similar_object_names(name1, name2):
                return False
        
        # Check if bounding boxes overlap significantly
        bbox1 = det1['bbox']
        bbox2 = det2['bbox']
        
        if self._calculate_bbox_overlap(bbox1, bbox2) > 0.3:  # 30% overlap threshold
            return True
        
        return False
    
    def _are_similar_object_names(self, name1: str, name2: str) -> bool:
        """Check if two object names are similar"""
        # Use Azure detector's similarity check if available
        if self.azure_detector:
            return self.azure_detector._are_similar_objects(name1, name2)
        
        # Basic similarity check
        return name1 in name2 or name2 in name1
    
    def _calculate_bbox_overlap(self, bbox1: Tuple, bbox2: Tuple) -> float:
        """Calculate overlap ratio between two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection
        x1_int = max(x1_1, x1_2)
        y1_int = max(y1_1, y1_2)
        x2_int = min(x2_1, x2_2)
        y2_int = min(y2_1, y2_2)
        
        if x2_int <= x1_int or y2_int <= y1_int:
            return 0.0  # No overlap
        
        intersection_area = (x2_int - x1_int) * (y2_int - y1_int)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = area1 + area2 - intersection_area
        
        return intersection_area / union_area if union_area > 0 else 0.0
    
    def _merge_detection_group(self, detections: List[Dict]) -> Dict:
        """Merge a group of similar detections into one"""
        # Use the detection with highest confidence as base
        base_detection = max(detections, key=lambda x: x['confidence'])
        
        # Average the bounding boxes (weighted by confidence)
        total_weight = sum(det['confidence'] for det in detections)
        
        weighted_x1 = sum(det['bbox'][0] * det['confidence'] for det in detections) / total_weight
        weighted_y1 = sum(det['bbox'][1] * det['confidence'] for det in detections) / total_weight
        weighted_x2 = sum(det['bbox'][2] * det['confidence'] for det in detections) / total_weight
        weighted_y2 = sum(det['bbox'][3] * det['confidence'] for det in detections) / total_weight
        
        # Create merged detection
        merged = base_detection.copy()
        merged['bbox'] = (int(weighted_x1), int(weighted_y1), int(weighted_x2), int(weighted_y2))
        merged['center'] = ((int(weighted_x1) + int(weighted_x2)) // 2, 
                           (int(weighted_y1) + int(weighted_y2)) // 2)
        merged['width'] = int(weighted_x2) - int(weighted_x1)
        merged['height'] = int(weighted_y2) - int(weighted_y1)
        merged['area'] = merged['width'] * merged['height']
        
        # Boost confidence slightly for merged detections
        merged['confidence'] = min(1.0, base_detection['confidence'] * 1.1)
        
        # Add source information
        sources = list(set(det.get('source', 'unknown') for det in detections))
        merged['source'] = 'merged_' + '_'.join(sources)
        
        return merged
    
    def find_object_by_name(self, frame: np.ndarray, object_name: str, method: str = "auto") -> Optional[Dict]:
        """
        Find specific object in frame using specified method
        
        Args:
            frame: Input image frame
            object_name: Name of object to find
            method: "local", "azure", "both", or "auto"
            
        Returns:
            Best matching detection or None
        """
        detections = self.detect_objects(frame, method)
        
        # Find objects matching the name
        matching_objects = []
        object_name_lower = object_name.lower()
        
        for detection in detections:
            class_name = detection['class_name'].lower()
            
            # Direct match or partial match
            if (object_name_lower in class_name or 
                class_name in object_name_lower or
                self._are_similar_object_names(object_name_lower, class_name)):
                matching_objects.append(detection)
        
        # Return the detection with highest confidence
        if matching_objects:
            return max(matching_objects, key=lambda x: x['confidence'])
        
        return None
    
    def draw_detections(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        Draw bounding boxes and labels on frame with source-specific colors
        
        Args:
            frame: Input image frame
            detections: List of detections
            
        Returns:
            Frame with drawn detections
        """
        result_frame = frame.copy()
        
        # Color map for different sources
        color_map = {
            'local_yolo': (0, 255, 0),      # Green
            'azure_objects': (255, 0, 255),  # Magenta
            'azure_tags': (255, 255, 0),     # Cyan
            'merged_local_yolo_azure_objects': (0, 255, 255),  # Yellow
            'merged_local_yolo_azure_tags': (128, 0, 255),     # Purple
        }
        
        for detection in detections:
            # Skip full-frame tag detections for drawing
            if detection.get('source') == 'azure_tags':
                continue
                
            x1, y1, x2, y2 = detection['bbox']
            confidence = detection['confidence']
            class_name = detection['class_name']
            source = detection.get('source', 'unknown')
            
            # Get color based on source
            color = color_map.get(source, (128, 128, 128))  # Gray for unknown
            
            # Draw bounding box
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label with source info
            source_prefix = ""
            if 'local' in source:
                source_prefix = "YOLO: "
            elif 'azure' in source:
                source_prefix = "Azure: "
            elif 'merged' in source:
                source_prefix = "Merged: "
            
            label = f"{source_prefix}{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(result_frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(result_frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        return result_frame
    
    def get_detection_summary(self, detections: List[Dict]) -> str:
        """
        Get a text summary of detections including source information
        
        Args:
            detections: List of detections
            
        Returns:
            Human-readable summary
        """
        if not detections:
            return "No objects detected by any method"
        
        # Group by source
        source_groups = {}
        for detection in detections:
            source = detection.get('source', 'unknown')
            if source not in source_groups:
                source_groups[source] = []
            source_groups[source].append(detection)
        
        summary_parts = []
        
        # Summarize each source
        for source, source_detections in source_groups.items():
            if source == 'azure_tags':
                continue  # Skip tag-only detections in summary
                
            object_counts = {}
            for detection in source_detections:
                class_name = detection['class_name']
                object_counts[class_name] = object_counts.get(class_name, 0) + 1
            
            items = []
            for obj_name, count in object_counts.items():
                if count == 1:
                    items.append(f"a {obj_name}")
                else:
                    items.append(f"{count} {obj_name}s")
            
            if items:
                source_name = "Local YOLO" if 'local' in source else "Azure" if 'azure' in source else "Combined"
                if len(items) == 1:
                    summary_parts.append(f"{source_name} found {items[0]}")
                else:
                    summary_parts.append(f"{source_name} found {', '.join(items)}")
        
        return ". ".join(summary_parts) if summary_parts else "Detection completed but no clear objects found"
    
    def get_available_methods(self) -> List[str]:
        """Get list of available detection methods"""
        methods = []
        if self.local_detector and self.local_detector.is_initialized:
            methods.append("local")
        if self.azure_detector and self.azure_detector.is_initialized:
            methods.append("azure")
        if len(methods) > 1:
            methods.append("both")
        return methods


def test_hybrid_detection():
    """Test hybrid object detection functionality"""
    try:
        detector = HybridObjectDetector()
        print(f"Hybrid detector initialized: {detector.is_initialized}")
        print(f"Available methods: {detector.get_available_methods()}")
        
        if detector.is_initialized:
            print("Hybrid object detector ready for use")
        else:
            print("Hybrid detector initialization failed")
    
    except Exception as e:
        print(f"Hybrid detection test failed: {e}")


if __name__ == "__main__":
    test_hybrid_detection()