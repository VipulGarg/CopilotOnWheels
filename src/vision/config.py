"""
Vision Configuration module
Handles configuration for different object detection methods
"""

import os
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class VisionConfig:
    """Configuration class for vision system"""
    
    # Detection method options
    DETECTION_LOCAL = "local"
    DETECTION_AZURE = "azure"
    DETECTION_HYBRID = "hybrid"
    DETECTION_AUTO = "auto"
    
    def __init__(self):
        """Initialize vision configuration"""
        self.detection_method = os.getenv("DETECTION_METHOD", self.DETECTION_AUTO)
        self.confidence_threshold = float(os.getenv("CONFIDENCE_THRESHOLD", "0.5"))
        self.prefer_local = os.getenv("PREFER_LOCAL", "true").lower() == "true"
        
        # Navigation parameters
        self.target_area_threshold = float(os.getenv("TARGET_AREA_THRESHOLD", "0.15"))
        self.center_tolerance = float(os.getenv("CENTER_TOLERANCE", "0.2"))
        self.max_navigation_time = int(os.getenv("MAX_NAVIGATION_TIME", "30"))
        self.frame_check_interval = float(os.getenv("FRAME_CHECK_INTERVAL", "0.5"))
        
        # Search rotation parameters
        self.max_search_rotations = int(os.getenv("MAX_SEARCH_ROTATIONS", "1"))
        self.rotation_steps_per_full_rotation = int(os.getenv("ROTATION_STEPS_PER_FULL_ROTATION", "4"))
        self.rotation_duration = float(os.getenv("ROTATION_DURATION", "0.5"))
        
        # Camera parameters
        self.camera_width = int(os.getenv("CAMERA_WIDTH", "640"))
        self.camera_height = int(os.getenv("CAMERA_HEIGHT", "480"))
        
        # Azure configuration
        self.azure_endpoint = os.getenv("AZURE_COMPUTER_VISION_ENDPOINT")
        self.azure_key = os.getenv("AZURE_COMPUTER_VISION_KEY")
    
    def get_detection_config(self) -> Dict[str, Any]:
        """Get detection-related configuration"""
        return {
            "method": self.detection_method,
            "confidence_threshold": self.confidence_threshold,
            "prefer_local": self.prefer_local,
            "azure_available": bool(self.azure_endpoint and self.azure_key)
        }
    
    def get_navigation_config(self) -> Dict[str, Any]:
        """Get navigation-related configuration"""
        return {
            "target_area_threshold": self.target_area_threshold,
            "center_tolerance": self.center_tolerance,
            "max_navigation_time": self.max_navigation_time,
            "frame_check_interval": self.frame_check_interval,
            "max_search_rotations": self.max_search_rotations,
            "rotation_steps_per_full_rotation": self.rotation_steps_per_full_rotation,
            "rotation_duration": self.rotation_duration
        }
    
    def get_camera_config(self) -> Dict[str, Any]:
        """Get camera-related configuration"""
        return {
            "width": self.camera_width,
            "height": self.camera_height
        }
    
    def is_azure_configured(self) -> bool:
        """Check if Azure Computer Vision is properly configured"""
        return bool(self.azure_endpoint and self.azure_key and 
                   self.azure_endpoint != "your-computer-vision-endpoint-here" and
                   self.azure_key != "your-computer-vision-key-here")
    
    def get_recommended_method(self) -> str:
        """Get recommended detection method based on available services"""
        if self.detection_method != self.DETECTION_AUTO:
            return self.detection_method
        
        # Auto-select based on available services
        if self.is_azure_configured():
            return self.DETECTION_HYBRID  # Use both local and Azure
        else:
            return self.DETECTION_LOCAL   # Fall back to local only
    
    def __str__(self) -> str:
        """String representation of configuration"""
        return f"""Vision Configuration:
  Detection Method: {self.detection_method}
  Confidence Threshold: {self.confidence_threshold}
  Prefer Local: {self.prefer_local}
  Azure Configured: {self.is_azure_configured()}
  
  Navigation:
    Target Area Threshold: {self.target_area_threshold}
    Center Tolerance: {self.center_tolerance}
    Max Navigation Time: {self.max_navigation_time}s
    Frame Check Interval: {self.frame_check_interval}s
  
  Camera:
    Resolution: {self.camera_width}x{self.camera_height}
"""

# Global configuration instance
vision_config = VisionConfig()