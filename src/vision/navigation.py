"""
Robot Vision Navigation System
Combines camera, object detection, and motor control for autonomous navigation
"""

import cv2
import numpy as np
import logging
import time
import threading
from typing import Dict, Optional, Tuple, List

from .camera import RobotCamera, CameraError
from .object_detection import ObjectDetector

class VisionNavigationSystem:
    """Complete vision-based navigation system for the robot"""
    
    def __init__(self):
        """Initialize the vision navigation system"""
        self.camera = None
        self.object_detector = None
        self.is_initialized = False
        self.navigation_active = False
        self.current_target = None
        
        # Navigation parameters
        self.target_area_threshold = 0.15  # Stop when object takes 15% of frame
        self.center_tolerance = 0.2  # 20% tolerance for centering
        self.max_navigation_time = 30  # Maximum time to navigate (seconds)
        self.frame_check_interval = 0.5  # Time between frame checks
        
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize camera and object detection"""
        try:
            # Initialize camera
            self.camera = RobotCamera()
            logging.info("Camera initialized for vision navigation")
            
            # Initialize object detector
            self.object_detector = ObjectDetector()
            logging.info("Object detector initialized for vision navigation")
            
            self.is_initialized = self.camera.is_initialized and self.object_detector.is_initialized
            
            if self.is_initialized:
                logging.info("Vision navigation system fully initialized")
            else:
                logging.error("Vision navigation system initialization failed")
        
        except Exception as e:
            logging.error(f"Vision navigation system initialization error: {e}")
            self.is_initialized = False
    
    def scan_area(self) -> Dict:
        """
        Scan the current area and return detected objects
        
        Returns:
            Dictionary with scan results
        """
        if not self.is_initialized:
            return {"success": False, "message": "Vision system not initialized"}
        
        try:
            # Capture frame
            frame = self.camera.capture_frame()
            if frame is None:
                return {"success": False, "message": "Could not capture camera frame"}
            
            # Detect objects
            detections = self.object_detector.detect_objects(frame)
            summary = self.object_detector.get_detection_summary(detections)
            
            return {
                "success": True,
                "message": summary,
                "detections": detections,
                "object_count": len(detections)
            }
        
        except Exception as e:
            logging.error(f"Area scan error: {e}")
            return {"success": False, "message": f"Scan failed: {e}"}
    
    def capture_image(self, filename: str = None) -> Dict:
        """
        Capture and save an image
        
        Args:
            filename: Optional filename for saved image
            
        Returns:
            Dictionary with capture results
        """
        if not self.is_initialized:
            return {"success": False, "message": "Vision system not initialized"}
        
        try:
            # Capture frame
            frame = self.camera.capture_frame()
            if frame is None:
                return {"success": False, "message": "Could not capture camera frame"}
            
            # Get detections for annotation
            detections = self.object_detector.detect_objects(frame)
            annotated_frame = self.object_detector.draw_detections(frame, detections)
            
            # Save image
            if filename is None:
                timestamp = int(time.time())
                filename = f"robot_capture_{timestamp}.jpg"
            
            cv2.imwrite(filename, annotated_frame)
            logging.info(f"Image captured and saved as {filename}")
            
            return {
                "success": True,
                "message": f"Image saved as {filename}",
                "filename": filename,
                "objects_detected": len(detections)
            }
        
        except Exception as e:
            logging.error(f"Image capture error: {e}")
            return {"success": False, "message": f"Capture failed: {e}"}
    
    def navigate_to_object(self, target_object: str, movement_functions: Dict) -> Dict:
        """
        Navigate to a specific object
        
        Args:
            target_object: Name of object to find and navigate to
            movement_functions: Dict of movement functions (forward, left, right, stop)
            
        Returns:
            Dictionary with navigation results
        """
        if not self.is_initialized:
            return {"success": False, "message": "Vision system not initialized"}
        
        if self.navigation_active:
            return {"success": False, "message": "Navigation already in progress"}
        
        self.navigation_active = True
        self.current_target = target_object
        
        try:
            return self._execute_navigation(target_object, movement_functions)
        
        finally:
            self.navigation_active = False
            self.current_target = None
    
    def _execute_navigation(self, target_object: str, movement_functions: Dict) -> Dict:
        """Execute the navigation process"""
        start_time = time.time()
        attempts = 0
        max_attempts = 20
        
        logging.info(f"Starting navigation to: {target_object}")
        
        while attempts < max_attempts:
            if time.time() - start_time > self.max_navigation_time:
                return {
                    "success": False,
                    "message": f"Navigation timeout after {self.max_navigation_time} seconds"
                }
            
            attempts += 1
            
            # Capture and analyze frame
            frame = self.camera.capture_frame()
            if frame is None:
                logging.warning("Could not capture frame, continuing...")
                time.sleep(0.5)
                continue
            
            # Find target object
            detection = self.object_detector.find_object_by_name(frame, target_object)
            
            if detection is None:
                # Object not found, try rotating to find it
                logging.info(f"Object '{target_object}' not found, rotating to search...")
                movement_functions['right'](0.5)  # Turn right for 0.5 seconds
                time.sleep(0.5)
                continue
            
            # Object found, check if we're close enough
            frame_area = frame.shape[0] * frame.shape[1]
            object_area_ratio = detection['area'] / frame_area
            
            if object_area_ratio > self.target_area_threshold:
                # Close enough to target
                movement_functions['stop']()
                logging.info(f"Successfully reached {target_object}")
                return {
                    "success": True,
                    "message": f"Successfully navigated to {target_object}",
                    "attempts": attempts,
                    "final_distance": "close"
                }
            
            # Calculate navigation action
            action = self._calculate_navigation_action(detection, frame.shape)
            
            # Execute movement
            self._execute_movement_action(action, movement_functions)
            
            # Wait before next iteration
            time.sleep(self.frame_check_interval)
        
        # Max attempts reached
        movement_functions['stop']()
        return {
            "success": False,
            "message": f"Could not reach {target_object} within {max_attempts} attempts",
            "attempts": attempts
        }
    
    def _calculate_navigation_action(self, detection: Dict, frame_shape: Tuple) -> str:
        """
        Calculate what movement action to take based on object position
        
        Args:
            detection: Object detection result
            frame_shape: Shape of the camera frame (height, width, channels)
            
        Returns:
            Navigation action: 'forward', 'left', 'right', 'stop'
        """
        frame_height, frame_width = frame_shape[:2]
        center_x, center_y = detection['center']
        
        # Calculate frame center
        frame_center_x = frame_width // 2
        
        # Calculate horizontal offset from center
        offset_x = center_x - frame_center_x
        tolerance = frame_width * self.center_tolerance
        
        if abs(offset_x) < tolerance:
            # Object is centered horizontally, move forward
            return 'forward'
        elif offset_x < 0:
            # Object is to the left, turn left
            return 'left'
        else:
            # Object is to the right, turn right
            return 'right'
    
    def _execute_movement_action(self, action: str, movement_functions: Dict):
        """Execute a movement action"""
        movement_duration = 0.3  # Short movements for precise navigation
        
        if action == 'forward':
            logging.info("Moving forward toward target")
            movement_functions['forward'](movement_duration)
        elif action == 'left':
            logging.info("Turning left to center target")
            movement_functions['left'](movement_duration)
        elif action == 'right':
            logging.info("Turning right to center target")
            movement_functions['right'](movement_duration)
        elif action == 'stop':
            logging.info("Stopping at target")
            movement_functions['stop']()
    
    def stop_navigation(self):
        """Stop current navigation"""
        self.navigation_active = False
        self.current_target = None
        logging.info("Navigation stopped")
    
    def get_status(self) -> Dict:
        """Get current system status"""
        return {
            "initialized": self.is_initialized,
            "navigation_active": self.navigation_active,
            "current_target": self.current_target,
            "camera_available": self.camera is not None and self.camera.is_initialized,
            "detector_available": self.object_detector is not None and self.object_detector.is_initialized
        }
    
    def cleanup(self):
        """Clean up resources"""
        if self.navigation_active:
            self.stop_navigation()
        
        if self.camera:
            self.camera.cleanup()
        
        logging.info("Vision navigation system cleaned up")