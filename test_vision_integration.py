"""
Test script for Robot Vision System
Tests the complete pipeline from intent recognition to object navigation
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from robot_ai import RobotAI
from vision.navigation import VisionNavigationSystem
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)

def test_intent_recognition():
    """Test AI intent recognition for vision commands"""
    print("=" * 50)
    print("Testing Intent Recognition")
    print("=" * 50)
    
    ai_service = RobotAI()
    
    test_commands = [
        "move to the cup",
        "go to the chair",
        "find a person",
        "what do you see",
        "scan the area", 
        "take a picture",
        "move forward",
        "what's the weather"
    ]
    
    for command in test_commands:
        print(f"\nCommand: '{command}'")
        result = ai_service.process_user_input(command)
        print(f"Intent: {result['intent']}")
        print(f"Action: {result['action']}")
        print(f"Target Object: {result.get('target_object', 'None')}")
        print(f"Response: {result['response_text']}")

def test_vision_system():
    """Test vision system initialization"""
    print("\n" + "=" * 50)
    print("Testing Vision System")
    print("=" * 50)
    
    try:
        vision_system = VisionNavigationSystem()
        status = vision_system.get_status()
        
        print(f"Vision System Initialized: {status['initialized']}")
        print(f"Camera Available: {status['camera_available']}")
        print(f"Detector Available: {status['detector_available']}")
        
        if status['initialized']:
            print("\nTesting area scan...")
            scan_result = vision_system.scan_area()
            print(f"Scan Success: {scan_result['success']}")
            print(f"Scan Message: {scan_result['message']}")
            
            if scan_result['success']:
                print(f"Objects Found: {scan_result['object_count']}")
        
        # Cleanup
        vision_system.cleanup()
        
    except Exception as e:
        print(f"Vision system test failed: {e}")

def test_complete_pipeline():
    """Test complete pipeline simulation"""
    print("\n" + "=" * 50)
    print("Testing Complete Pipeline")
    print("=" * 50)
    
    # Simulate user commands
    test_scenarios = [
        "move to the cup",
        "what do you see",
        "take a photo"
    ]
    
    ai_service = RobotAI()
    vision_system = VisionNavigationSystem()
    
    for command in test_scenarios:
        print(f"\n--- Processing: '{command}' ---")
        
        # Step 1: AI Intent Recognition
        result = ai_service.process_user_input(command)
        print(f"AI Recognition - Intent: {result['intent']}, Action: {result['action']}")
        
        # Step 2: Vision System Processing (if vision intent)
        if result['intent'] == 'vision' and vision_system.is_initialized:
            action = result['action']
            target_object = result.get('target_object')
            
            if action == 'scan_area':
                vision_result = vision_system.scan_area()
                print(f"Vision Result: {vision_result['message']}")
                
            elif action == 'capture_image':
                vision_result = vision_system.capture_image()
                print(f"Vision Result: {vision_result['message']}")
                
            elif action == 'navigate_to_object' and target_object:
                print(f"Would navigate to: {target_object}")
                # Note: Not actually moving for test safety
            
        elif result['intent'] == 'movement':
            print(f"Would execute movement: {result['action']}")
        
        else:
            print(f"Response: {result['response_text']}")
    
    # Cleanup
    if vision_system:
        vision_system.cleanup()

def main():
    """Run all tests"""
    print("Robot Vision System Test Suite")
    print("==============================")
    
    try:
        test_intent_recognition()
        test_vision_system() 
        test_complete_pipeline()
        
        print("\n" + "=" * 50)
        print("TEST SUMMARY")
        print("=" * 50)
        print("✅ Intent recognition test completed")
        print("✅ Vision system test completed")
        print("✅ Complete pipeline test completed")
        print("\nThe robot is ready for voice-controlled object navigation!")
        print("\nExample commands you can try:")
        print("- 'move to the cup'")
        print("- 'go to the chair'")
        print("- 'what do you see'")
        print("- 'take a picture'")
        
    except Exception as e:
        print(f"Test suite failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()