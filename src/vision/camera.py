"""
Camera module for Raspberry Pi robot
Handles image capture using PiCamera2 or USB camera as fallback
"""

import cv2
import numpy as np
import logging
import time
from typing import Optional, Tuple
import os

try:
    from picamera2 import Picamera2
    PICAMERA2_AVAILABLE = True
except ImportError:
    PICAMERA2_AVAILABLE = False
    logging.warning("PiCamera2 not available - using OpenCV fallback")

class CameraError(Exception):
    """Custom exception for camera-related errors"""
    pass

class RobotCamera:
    """Camera interface for robot vision system"""
    
    def __init__(self, width: int = 640, height: int = 480):
        """
        Initialize camera   
        
        Args:
            width: Image width
            height: Image height
        """
        self.width = width
        self.height = height
        self.camera = None
        self.camera_type = None
        self.is_initialized = False
        
        # Try to initialize camera
        self._initialize_camera()
    
    def _initialize_camera(self):
        """Initialize camera (try PiCamera2 first, then USB camera)"""
        # Try PiCamera2 first (native Raspberry Pi camera)
        if PICAMERA2_AVAILABLE:
            try:
                self.camera = Picamera2()
                config = self.camera.create_preview_configuration(
                    main={"size": (self.width, self.height)}
                )
                self.camera.configure(config)
                self.camera.start()
                time.sleep(2)  # Allow camera to warm up
                
                self.camera_type = "PiCamera2"
                self.is_initialized = True
                logging.info("PiCamera2 initialized successfully")
                return
                
            except Exception as e:
                logging.warning(f"PiCamera2 initialization failed: {e}")
                self.camera = None
        
        # Try USB camera as fallback
        try:
            self.camera = cv2.VideoCapture(0)
            if not self.camera.isOpened():
                raise CameraError("Could not open USB camera")
            
            # Set resolution
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            
            # Configure camera for minimal buffering to reduce latency
            self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Set FPS if supported (lower FPS can help with buffer issues)
            self.camera.set(cv2.CAP_PROP_FPS, 15)
            
            # Give the USB camera time to warm up and flush initial frames from buffer
            time.sleep(2)
            for _ in range(5):
                try:
                    ret, _ = self.camera.read()
                except Exception:
                    ret = False
                if not ret:
                    # small delay before trying again
                    time.sleep(0.1)
            
            self.camera_type = "USB"
            self.is_initialized = True
            logging.info("USB camera initialized successfully")
            print ("here")
            
        except Exception as e:
            logging.error(f"USB camera initialization failed: {e}")
            self.is_initialized = False
            self.camera = None
            raise CameraError("No camera available")
    
    def capture_frame(self) -> Optional[np.ndarray]:
        """
        Capture a single frame
        
        Returns:
            numpy array representing the image, or None if capture fails
        """
        if not self.is_initialized or self.camera is None:
            logging.error("Camera not initialized")
            return None
        
        try:
            if self.camera_type == "PiCamera2":
                # Capture with PiCamera2
                frame = self.camera.capture_array()
                # Picamera2 returns RGB arrays; convert to BGR for OpenCV consistency
                if len(frame.shape) == 3 and frame.shape[2] == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                return frame
                
            elif self.camera_type == "USB":
                # Capture with USB camera
                # For USB cameras, we need to flush the buffer to get the most recent frame
                # Read and discard several frames to clear the buffer, then capture the fresh one
                for _ in range(3):
                    ret, _ = self.camera.read()
                    if not ret:
                        break
                
                # Now capture the actual frame we want to return
                ret, frame = self.camera.read()
                if ret:
                    return frame
                else:
                    logging.error("Failed to capture frame from USB camera")
                    return None
        
        except Exception as e:
            logging.error(f"Frame capture error: {e}")
            return None
    
    def capture_and_save(self, filename: str = None) -> str:
        """
        Capture frame and save to file
        
        Args:
            filename: Output filename (if None, auto-generate with timestamp)
            
        Returns:
            Filename of saved image
        """
        frame = self.capture_frame()
        if frame is None:
            raise CameraError("Could not capture frame")
        
        if filename is None:
            timestamp = int(time.time())
            filename = f"capture_{timestamp}.jpg"
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
        
        # Save image
        cv2.imwrite(filename, frame)
        logging.info(f"Image saved to {filename}")
        return filename
    
    def get_camera_info(self) -> dict:
        """Get camera information"""
        return {
            'type': self.camera_type,
            'initialized': self.is_initialized,
            'resolution': (self.width, self.height),
            'available': self.camera is not None
        }
    
    def set_resolution(self, width: int, height: int) -> bool:
        """
        Change camera resolution
        
        Args:
            width: New width
            height: New height
            
        Returns:
            True if successful
        """
        try:
            if self.camera_type == "USB" and self.camera is not None:
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                self.width = width
                self.height = height
                logging.info(f"Resolution changed to {width}x{height}")
                return True
            elif self.camera_type == "PiCamera2":
                # For PiCamera2, we'd need to reconfigure
                logging.warning("Resolution change not implemented for PiCamera2")
                return False
        
        except Exception as e:
            logging.error(f"Resolution change failed: {e}")
            return False
        
        return False
    
    def cleanup(self):
        """Clean up camera resources"""
        if self.camera is not None:
            try:
                if self.camera_type == "PiCamera2":
                    self.camera.stop()
                elif self.camera_type == "USB":
                    self.camera.release()
                
                logging.info("Camera cleanup completed")
            
            except Exception as e:
                logging.error(f"Camera cleanup error: {e}")
            
            finally:
                self.camera = None
                self.is_initialized = False
    
    def __del__(self):
        """Destructor to ensure cleanup"""
        self.cleanup()


def test_camera():
    """Test camera functionality"""
    try:
        camera = RobotCamera()
        print(f"Camera info: {camera.get_camera_info()}")
        
        if camera.is_initialized:
            print("Capturing test image...")
            filename = camera.capture_and_save("test_capture.jpg")
            print(f"Test image saved as: {filename}")
        else:
            print("Camera not initialized")
    
    except Exception as e:
        print(f"Camera test failed: {e}")
    
    finally:
        if 'camera' in locals():
            camera.cleanup()


if __name__ == "__main__":
    test_camera()