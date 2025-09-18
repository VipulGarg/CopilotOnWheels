"""
Robot Vision Navigation System
Combines camera, object detection, and motor control for autonomous navigation
Supports local YOLO, Azure Computer Vision, and hybrid detection
"""

import cv2
import numpy as np
import logging
import time
import threading
from typing import Dict, Optional, Tuple, List
import os
from dotenv import load_dotenv

from .camera import RobotCamera, CameraError
from .object_detection import ObjectDetector
from .azure_object_detection import AzureObjectDetector
from .hybrid_object_detection import HybridObjectDetector
from .config import vision_config

# Load environment variables
load_dotenv()

class VisionNavigationSystem:
    """Complete vision-based navigation system for the robot"""
    
    def __init__(self, detection_method: str = "auto", debug_mode: bool = False):
        """
        Initialize the vision navigation system
        
        Args:
            detection_method: "local", "azure", "hybrid", or "auto"
            debug_mode: Enable detailed debug logging
        """
        self.camera = None
        self.object_detector = None
        self.detection_method = detection_method
        self.debug_mode = debug_mode
        self.is_initialized = False
        self.navigation_active = False
        self.current_target = None
        
        # Set logging level based on debug mode
        if debug_mode:
            logging.getLogger().setLevel(logging.DEBUG)
            logging.info("🐛 Debug mode enabled for vision navigation")
        
        # Navigation parameters (loaded from config)
        nav_config = vision_config.get_navigation_config()
        self.target_area_threshold = nav_config["target_area_threshold"]
        self.center_tolerance = nav_config["center_tolerance"] 
        self.max_navigation_time = nav_config["max_navigation_time"]
        self.frame_check_interval = nav_config["frame_check_interval"]
        self.max_search_rotations = nav_config["max_search_rotations"]
        self.rotation_steps_per_full_rotation = nav_config["rotation_steps_per_full_rotation"]
        self.rotation_duration = nav_config["rotation_duration"]
        
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize camera and object detection"""
        try:
            # Initialize camera
            self.camera = RobotCamera()
            logging.info("Camera initialized for vision navigation")
            
            # Initialize object detector based on method
            detector_initialized = False
            
            if self.detection_method == "local":
                self.object_detector = ObjectDetector()
                detector_initialized = self.object_detector.is_initialized
                logging.info("Local YOLO detector initialized for vision navigation")
                
            elif self.detection_method == "azure":
                self.object_detector = AzureObjectDetector()
                detector_initialized = self.object_detector.is_initialized
                logging.info("Azure detector initialized for vision navigation")
                
            elif self.detection_method == "hybrid":
                self.object_detector = HybridObjectDetector()
                detector_initialized = self.object_detector.is_initialized
                logging.info("Hybrid detector initialized for vision navigation")
                
            else:  # auto
                # Try hybrid first, then local, then Azure
                try:
                    self.object_detector = HybridObjectDetector()
                    if self.object_detector.is_initialized:
                        detector_initialized = True
                        self.detection_method = "hybrid"
                        logging.info("Auto-selected hybrid detector for vision navigation")
                    else:
                        raise Exception("Hybrid detector not available")
                except:
                    try:
                        self.object_detector = ObjectDetector()
                        if self.object_detector.is_initialized:
                            detector_initialized = True
                            self.detection_method = "local"
                            logging.info("Auto-selected local YOLO detector for vision navigation")
                        else:
                            raise Exception("Local detector not available")
                    except:
                        self.object_detector = AzureObjectDetector()
                        detector_initialized = self.object_detector.is_initialized
                        self.detection_method = "azure"
                        logging.info("Auto-selected Azure detector for vision navigation")
            
            self.is_initialized = self.camera.is_initialized and detector_initialized
            
            if self.is_initialized:
                logging.info(f"Vision navigation system fully initialized with {self.detection_method} detection")
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
        """Execute the navigation process with controlled search pattern"""
        start_time = time.time()
        attempts = 0
        max_attempts = 20
        search_rotations_completed = 0
        current_rotation_step = 0
        
        logging.info(f"Starting navigation to: {target_object}")
        logging.info(f"Search parameters: {self.max_search_rotations} rotation(s), {self.rotation_steps_per_full_rotation} steps per rotation")
        logging.info(f"Navigation thresholds - target_area: {self.target_area_threshold}, center_tolerance: {self.center_tolerance}")
        logging.info(f"Detection method: {self.detection_method}")
        
        # Log initial scan to see what's visible
        try:
            initial_frame = self.camera.capture_frame()
            if initial_frame is not None:
                logging.info(f"Initial frame captured: {initial_frame.shape}")
                initial_detections = self.object_detector.detect_objects(initial_frame)
                logging.info(f"Initial scan found {len(initial_detections)} objects:")
                for i, det in enumerate(initial_detections):
                    logging.info(f"  {i+1}. {det['class_name']} (confidence: {det['confidence']:.2f}, area: {det['area']}, center: {det['center']})")
                
                # Check if target is already visible
                initial_target = self.object_detector.find_object_by_name(initial_frame, target_object)
                if initial_target:
                    logging.info(f"Target '{target_object}' already visible with confidence {initial_target['confidence']:.2f}")
                else:
                    logging.info(f"Target '{target_object}' not found in initial scan")
            else:
                logging.error("Failed to capture initial frame for debugging")
        except Exception as e:
            logging.error(f"Error during initial scan: {e}")
        
        while attempts < max_attempts:
            elapsed_time = time.time() - start_time
            logging.debug(f"Navigation attempt {attempts}/{max_attempts}, elapsed time: {elapsed_time:.1f}s")
            
            if elapsed_time > self.max_navigation_time:
                logging.warning(f"Navigation timeout reached: {elapsed_time:.1f}s > {self.max_navigation_time}s")
                return {
                    "success": False,
                    "message": f"Navigation timeout after {self.max_navigation_time} seconds"
                }
            
            attempts += 1
            logging.debug(f"--- Navigation Attempt {attempts} ---")
            
            # Capture and analyze frame
            frame = self.camera.capture_frame()
            if frame is None:
                logging.warning(f"Could not capture frame on attempt {attempts}, continuing...")
                time.sleep(0.5)
                continue
            
            logging.debug(f"Frame captured successfully: {frame.shape}")
            
            # Detect all objects for debugging
            all_detections = self.object_detector.detect_objects(frame)
            logging.debug(f"Frame analysis found {len(all_detections)} total objects:")
            for i, det in enumerate(all_detections[:5]):  # Log first 5 objects
                logging.debug(f"  {i+1}. {det['class_name']} (conf: {det['confidence']:.2f}, area: {det['area']}, center: {det['center']})")
            
            # Save debug frame if in debug mode
            if self.debug_mode and attempts % 3 == 1:  # Save every 3rd frame to avoid spam
                try:
                    debug_frame = self.object_detector.draw_detections(frame, all_detections)
                    debug_filename = f"debug_nav_frame_{attempts:03d}_{int(time.time())}.jpg"
                    cv2.imwrite(debug_filename, debug_frame)
                    logging.debug(f"💾 Saved debug frame: {debug_filename}")
                except Exception as e:
                    logging.debug(f"Failed to save debug frame: {e}")
            
            # Find target object
            detection = self.object_detector.find_object_by_name(frame, target_object)
            
            if detection is None:
                logging.debug(f"Target '{target_object}' not found in current frame")
                # Object not found, use controlled rotation pattern
                if search_rotations_completed >= self.max_search_rotations:
                    # We've completed our search rotation(s), give up
                    movement_functions['stop']()
                    logging.info(f"Object '{target_object}' not found after completing {search_rotations_completed} full rotation(s)")
                    return {
                        "success": False,
                        "message": f"Could not find {target_object} after systematic search (completed {search_rotations_completed} rotation(s))",
                        "attempts": attempts,
                        "rotations_completed": search_rotations_completed
                    }
                
                # Continue with systematic rotation
                current_rotation_step += 1
                logging.info(f"🔄 Object '{target_object}' not found, rotating step {current_rotation_step}/{self.rotation_steps_per_full_rotation} (rotation {search_rotations_completed + 1}/{self.max_search_rotations})")
                logging.debug(f"Rotation details:")
                logging.debug(f"  Duration: {self.rotation_duration}s")
                logging.debug(f"  Direction: RIGHT")
                logging.debug(f"  Current step: {current_rotation_step}")
                logging.debug(f"  Steps remaining in this rotation: {self.rotation_steps_per_full_rotation - current_rotation_step}")
                
                # Rotate right for approximately 90 degrees
                # Duration is configurable based on robot's turning speed
                movement_functions['right'](self.rotation_duration)
                time.sleep(0.3)  # Brief pause after rotation
                
                # Check if we completed a full rotation
                if current_rotation_step >= self.rotation_steps_per_full_rotation:
                    search_rotations_completed += 1
                    current_rotation_step = 0
                    logging.info(f"✅ Completed rotation {search_rotations_completed}/{self.max_search_rotations}")
                    if search_rotations_completed < self.max_search_rotations:
                        logging.info(f"Starting next rotation...")
                
                continue
            
            # Object found! Reset rotation tracking and proceed with navigation
            logging.info(f"✅ Found {target_object}! Detection details:")
            logging.info(f"  Confidence: {detection['confidence']:.3f}")
            logging.info(f"  Center: {detection['center']}")
            logging.info(f"  BBox: {detection['bbox']}")
            logging.info(f"  Area: {detection['area']} pixels")
            logging.info(f"  Source: {detection.get('source', 'unknown')}")
            
            if current_rotation_step > 0:
                logging.info(f"Found {target_object} during search rotation (step {current_rotation_step})!")
            
            search_rotations_completed = 0  # Reset since we found the object
            current_rotation_step = 0
            
            # Object found, check if we're close enough
            frame_area = frame.shape[0] * frame.shape[1]
            object_area_ratio = detection['area'] / frame_area
            
            logging.info(f"Distance analysis:")
            logging.info(f"  Frame area: {frame_area} pixels")
            logging.info(f"  Object area: {detection['area']} pixels")
            logging.info(f"  Area ratio: {object_area_ratio:.4f} (threshold: {self.target_area_threshold})")
            
            if object_area_ratio > self.target_area_threshold:
                # Close enough to target
                movement_functions['stop']()
                logging.info(f"🎯 Successfully reached {target_object}! Object is close enough (ratio: {object_area_ratio:.4f} > {self.target_area_threshold})")
                return {
                    "success": True,
                    "message": f"Successfully navigated to {target_object}",
                    "attempts": attempts,
                    "final_distance": "close",
                    "final_area_ratio": object_area_ratio
                }
            
            # Calculate navigation action
            action = self._calculate_navigation_action(detection, frame.shape)
            logging.info(f"Object not close enough yet (ratio: {object_area_ratio:.4f} <= {self.target_area_threshold}), action: {action}")
            
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
        
        logging.debug(f"Navigation calculation:")
        logging.debug(f"  Frame dimensions: {frame_width}x{frame_height}")
        logging.debug(f"  Frame center X: {frame_center_x}")
        logging.debug(f"  Object center: ({center_x}, {center_y})")
        logging.debug(f"  Horizontal offset: {offset_x} pixels")
        logging.debug(f"  Center tolerance: ±{tolerance:.1f} pixels ({self.center_tolerance*100:.1f}% of frame width)")
        
        if abs(offset_x) < tolerance:
            # Object is centered horizontally, move forward
            logging.debug(f"  Decision: FORWARD (object centered within tolerance)")
            return 'forward'
        elif offset_x < 0:
            # Object is to the left, turn left
            logging.debug(f"  Decision: LEFT (object is {abs(offset_x):.1f} pixels to the left)")
            return 'left'
        else:
            # Object is to the right, turn right
            logging.debug(f"  Decision: RIGHT (object is {offset_x:.1f} pixels to the right)")
            return 'right'
    
    def _execute_movement_action(self, action: str, movement_functions: Dict):
        """Execute a movement action"""
        movement_duration = 0.3  # Short movements for precise navigation
        
        logging.info(f"🤖 Executing movement: {action.upper()}")
        
        if action == 'forward':
            logging.info(f"Moving forward toward target for {movement_duration}s")
            movement_functions['forward'](movement_duration)
        elif action == 'left':
            logging.info(f"Turning left to center target for {movement_duration}s")
            movement_functions['left'](movement_duration)
        elif action == 'right':
            logging.info(f"Turning right to center target for {movement_duration}s")
            movement_functions['right'](movement_duration)
        elif action == 'stop':
            logging.info("Stopping at target")
            movement_functions['stop']()
        else:
            logging.warning(f"Unknown movement action: {action}")
        
        # Log completion
        logging.debug(f"Movement {action} completed")
    
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
            "detector_available": self.object_detector is not None and self.object_detector.is_initialized,
            "detection_method": self.detection_method,
            "debug_mode": self.debug_mode,
            "navigation_params": {
                "target_area_threshold": self.target_area_threshold,
                "center_tolerance": self.center_tolerance,
                "max_navigation_time": self.max_navigation_time,
                "max_search_rotations": self.max_search_rotations,
                "rotation_steps_per_full_rotation": self.rotation_steps_per_full_rotation,
                "rotation_duration": self.rotation_duration
            }
        }
    
    def enable_debug_mode(self):
        """Enable debug mode with detailed logging"""
        self.debug_mode = True
        logging.getLogger().setLevel(logging.DEBUG)
        logging.info("🐛 Debug mode enabled")
    
    def disable_debug_mode(self):
        """Disable debug mode"""
        self.debug_mode = False
        logging.getLogger().setLevel(logging.INFO)
        logging.info("Debug mode disabled")
    
    def cleanup(self):
        """Clean up resources"""
        if self.navigation_active:
            self.stop_navigation()
        
        if self.camera:
            self.camera.cleanup()
        
        logging.info("Vision navigation system cleaned up")