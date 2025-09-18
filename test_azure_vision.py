#!/usr/bin/env python3
"""
Test script for Azure Computer Vision object detection
Run this to verify your Azure setup is working correctly
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.vision.azure_object_detection import AzureObjectDetector, test_azure_detection
from src.vision.hybrid_object_detection import HybridObjectDetector, test_hybrid_detection
from src.vision.config import vision_config
import cv2
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_azure_with_sample_image():
    """Test Azure detection with a sample image"""
    print("=" * 50)
    print("Testing Azure Computer Vision Object Detection")
    print("=" * 50)
    
    # Check configuration
    print(f"Azure Endpoint: {vision_config.azure_endpoint}")
    print(f"Azure Key: {'***' + vision_config.azure_key[-4:] if vision_config.azure_key else 'Not set'}")
    print(f"Azure Configured: {vision_config.is_azure_configured()}")
    print()
    
    if not vision_config.is_azure_configured():
        print("❌ Azure Computer Vision not configured!")
        print("Please set AZURE_COMPUTER_VISION_ENDPOINT and AZURE_COMPUTER_VISION_KEY in your .env file")
        return False
    
    # Initialize detector
    detector = AzureObjectDetector()
    if not detector.is_initialized:
        print("❌ Azure detector failed to initialize")
        return False
    
    print("✅ Azure detector initialized successfully")
    
    # Try to capture a frame from camera for testing
    try:
        from src.vision.camera import RobotCamera
        camera = RobotCamera()
        if camera.is_initialized:
            print("📷 Capturing test frame from camera...")
            frame = camera.capture_frame()
            if frame is not None:
                print(f"✅ Camera frame captured: {frame.shape}")
                
                # Test object detection
                print("🔍 Running Azure object detection...")
                detections = detector.detect_objects(frame)
                print(f"✅ Azure detected {len(detections)} objects")
                
                for i, detection in enumerate(detections[:5]):  # Show first 5
                    print(f"  {i+1}. {detection['class_name']} (confidence: {detection['confidence']:.2f})")
                
                # Test detailed analysis
                print("📊 Getting detailed Azure analysis...")
                analysis = detector.get_detailed_analysis(frame)
                if analysis.get('success'):
                    print("✅ Detailed analysis successful")
                    if analysis.get('descriptions'):
                        print(f"  Description: {analysis['descriptions'][0]['text']}")
                    if analysis.get('tags'):
                        top_tags = [tag['name'] for tag in analysis['tags'][:3]]
                        print(f"  Top tags: {', '.join(top_tags)}")
                else:
                    print(f"❌ Detailed analysis failed: {analysis.get('message')}")
                
                camera.cleanup()
                return True
            else:
                print("❌ Failed to capture camera frame")
                camera.cleanup()
        else:
            print("⚠️  Camera not available, skipping image test")
            
    except Exception as e:
        print(f"❌ Camera test failed: {e}")
    
    return False

def test_hybrid_detection():
    """Test hybrid detection system"""
    print("=" * 50)
    print("Testing Hybrid Object Detection System")
    print("=" * 50)
    
    detector = HybridObjectDetector()
    print(f"Hybrid detector initialized: {detector.is_initialized}")
    print(f"Available methods: {detector.get_available_methods()}")
    
    if detector.is_initialized:
        print("✅ Hybrid detection system ready")
        return True
    else:
        print("❌ Hybrid detection system failed")
        return False

def main():
    """Run all tests"""
    print("🤖 Azure Computer Vision Test Suite")
    print()
    
    # Show configuration
    print("Configuration:")
    print(vision_config)
    
    # Test individual components
    azure_ok = test_azure_with_sample_image()
    print()
    hybrid_ok = test_hybrid_detection()
    
    print("\n" + "=" * 50)
    print("Test Summary:")
    print(f"Azure Detection: {'✅ PASS' if azure_ok else '❌ FAIL'}")
    print(f"Hybrid Detection: {'✅ PASS' if hybrid_ok else '❌ FAIL'}")
    
    if not azure_ok:
        print("\n💡 To fix Azure issues:")
        print("1. Create an Azure Computer Vision resource in Azure Portal")
        print("2. Copy the endpoint and key to your .env file:")
        print("   AZURE_COMPUTER_VISION_ENDPOINT=https://your-resource.cognitiveservices.azure.com/")
        print("   AZURE_COMPUTER_VISION_KEY=your-32-character-key")
        print("3. Install required packages: pip install azure-cognitiveservices-vision-computervision msrest")
        print("4. Ensure you have internet connectivity")
    
    return azure_ok or hybrid_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)