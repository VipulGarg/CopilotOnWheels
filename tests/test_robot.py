#!/usr/bin/env python3
"""
Test script for Robot Assistant
Tests all components individually before running the main program
"""

import os
import sys
import time
import logging
from dotenv import load_dotenv

def test_imports():
    """Test if all required modules can be imported"""
    print("🧪 Testing imports...")
    
    try:
        import speech_recognition as sr
        print("✅ SpeechRecognition imported successfully")
    except ImportError as e:
        print(f"❌ SpeechRecognition import failed: {e}")
        return False
    
    try:
        import pyttsx3
        print("✅ pyttsx3 (TTS) imported successfully")
    except ImportError as e:
        print(f"❌ pyttsx3 import failed: {e}")
        return False
    
    try:
        from openai import AzureOpenAI
        print("✅ Azure OpenAI imported successfully")
    except ImportError as e:
        print(f"❌ Azure OpenAI import failed: {e}")
        return False
    
    try:
        from robot_ai import RobotAI
        from robot_tts import RobotTTS
        print("✅ Robot modules imported successfully")
    except ImportError as e:
        print(f"❌ Robot modules import failed: {e}")
        return False
    
    return True

def test_environment():
    """Test environment configuration"""
    print("\n⚙️ Testing environment configuration...")
    
    # Load environment variables
    load_dotenv()
    
    required_vars = [
        'AZURE_OPENAI_ENDPOINT',
        'AZURE_OPENAI_API_KEY',
        'AZURE_OPENAI_DEPLOYMENT_NAME'
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"❌ Missing required environment variables: {missing_vars}")
        print("📝 Please check your .env file")
        return False
    
    print("✅ All required environment variables are set")
    return True

def test_tts():
    """Test text-to-speech functionality"""
    print("\n🗣️ Testing text-to-speech...")
    
    try:
        from robot_tts import RobotTTS
        
        tts = RobotTTS()
        if not tts.engine:
            print("❌ TTS engine initialization failed")
            return False
        
        print("🔊 Testing TTS (you should hear a voice)...")
        success = tts.speak("Text to speech test successful!", wait=True)
        
        if success:
            print("✅ TTS test completed successfully")
            return True
        else:
            print("❌ TTS test failed")
            return False
            
    except Exception as e:
        print(f"❌ TTS test error: {e}")
        return False

def test_speech_recognition():
    """Test speech recognition"""
    print("\n🎤 Testing speech recognition...")
    
    try:
        import speech_recognition as sr
        
        recognizer = sr.Recognizer()
        
        # Test microphone access
        try:
            with sr.Microphone() as source:
                print("🎙️ Microphone access successful")
                print("📊 Adjusting for ambient noise...")
                recognizer.adjust_for_ambient_noise(source, duration=1)
                
            print("✅ Speech recognition setup successful")
            print("💡 For full test, say something after starting the main program")
            return True
            
        except OSError as e:
            print(f"❌ Microphone access failed: {e}")
            print("💡 Check microphone connection and permissions")
            return False
            
    except Exception as e:
        print(f"❌ Speech recognition test error: {e}")
        return False

def test_azure_ai():
    """Test Azure OpenAI connection"""
    print("\n🧠 Testing Azure OpenAI connection...")
    
    try:
        from robot_ai import RobotAI
        
        ai = RobotAI()
        
        # Test simple intent classification
        test_input = "move forward"
        result = ai.classify_intent(test_input)
        
        if result and 'intent' in result:
            print(f"✅ Azure AI test successful")
            print(f"📋 Test result: {result}")
            return True
        else:
            print("❌ Azure AI returned unexpected result")
            return False
            
    except Exception as e:
        print(f"❌ Azure AI test error: {e}")
        print("💡 Check your Azure credentials and internet connection")
        return False

def test_gpio_simulation():
    """Test GPIO functionality (simulation mode for non-Pi systems)"""
    print("\n🔌 Testing GPIO functionality...")
    
    try:
        # Try to import RPi.GPIO
        import RPi.GPIO as GPIO
        print("✅ RPi.GPIO imported successfully (running on Raspberry Pi)")
        return True
    except ImportError:
        print("ℹ️ RPi.GPIO not available (not running on Raspberry Pi)")
        print("💡 GPIO functionality will be simulated for testing")
        return True
    except Exception as e:
        print(f"⚠️ GPIO test warning: {e}")
        return True  # Non-critical for testing

def main():
    """Run all tests"""
    print("🤖 Robot Assistant Test Suite")
    print("=" * 40)
    
    tests = [
        ("Module Imports", test_imports),
        ("Environment Config", test_environment),
        ("Text-to-Speech", test_tts),
        ("Speech Recognition", test_speech_recognition),
        ("Azure AI Services", test_azure_ai),
        ("GPIO Functionality", test_gpio_simulation)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 40)
    print("📊 Test Results Summary:")
    print("=" * 40)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name}: {status}")
        if success:
            passed += 1
    
    print(f"\n🏆 Tests Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! Your robot assistant is ready to run.")
        print("🚀 Run 'python3 main.py' to start the robot.")
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
        print("📖 Refer to README.md for troubleshooting guide.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)