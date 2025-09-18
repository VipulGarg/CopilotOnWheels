import speech_recognition as sr
import RPi.GPIO as GPIO
import time
import logging
import sys
from robot_ai import RobotAI
from robot_tts import RobotTTS, create_movement_speech, create_error_speech, create_greeting_speech
from vision.navigation import VisionNavigationSystem
from vision.config import vision_config
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Setup centralized logging to print to command line (stdout)
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Create stream handler to stdout with a clear formatter
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
stream_handler.setFormatter(formatter)

# Replace existing handlers to ensure logs go to stdout once
logger.handlers = [stream_handler]

# -------------------------------
# GPIO Pin Configuration
# -------------------------------
IN1 = 23
IN2 = 24
IN3 = 27
IN4 = 22
ENA = 25
ENB = 17

GPIO.setmode(GPIO.BCM)
GPIO.setup([IN1, IN2, IN3, IN4, ENA, ENB], GPIO.OUT)
GPIO.output(ENA, GPIO.HIGH)
GPIO.output(ENB, GPIO.HIGH)

# -------------------------------
# Motor Control Functions
# -------------------------------
def stop():
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.LOW)

def forward():
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)

def backward():
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.HIGH)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.HIGH)

def left():
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.HIGH)
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)

def right():
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.HIGH)

# -------------------------------
# Speech Recognition Setup
# -------------------------------
def setup_robot_services():
    """Initialize all robot services"""
    try:
        # Log vision configuration
        logging.info("Vision Configuration:")
        logging.info(f"  Detection Method: {vision_config.detection_method}")
        logging.info(f"  Azure Configured: {vision_config.is_azure_configured()}")
        logging.info(f"  Recommended Method: {vision_config.get_recommended_method()}")
        
        # Initialize AI service
        ai_service = RobotAI()
        logging.info("AI service initialized successfully")
        
        # Initialize TTS service
        tts_service = RobotTTS()
        logging.info("TTS service initialized successfully")
        
        # Initialize Vision Navigation System with configured method
        recommended_method = vision_config.get_recommended_method()
        vision_system = VisionNavigationSystem(detection_method=recommended_method)
        if vision_system.is_initialized:
            logging.info(f"Vision navigation system initialized successfully with {vision_system.detection_method} detection")
        else:
            logging.warning("Vision navigation system initialization failed")
        
        # Test TTS
        if tts_service.engine:
            detection_msg = f"Using {vision_system.detection_method} detection" if vision_system.is_initialized else "Vision system unavailable"
            tts_service.speak(f"Robot services initialized successfully! {detection_msg}", wait=True)
        
        return ai_service, tts_service, vision_system
    
    except Exception as e:
        logging.error(f"Failed to initialize robot services: {e}")
        return None, None, None

def execute_movement_command(action: str, tts_service):
    """Execute movement command and provide audio feedback"""
    try:
        movement_functions = {
            'forward': forward,
            'backward': backward,
            'left': left,
            'right': right,
            'stop': stop
        }
        
        if action in movement_functions:
            # Speak before moving
            speech_text = create_movement_speech(action)
            if tts_service:
                tts_service.speak(speech_text)
            
            # Execute movement
            movement_functions[action]()
            logging.info(f"Executed movement: {action}")
            
            # Auto-stop after 0.5 seconds for safety (except for stop command)
            if action != 'stop':
                time.sleep(0.5)
                stop()
                if tts_service:
                    tts_service.speak("Movement complete.")
            
            return True
        else:
            logging.warning(f"Unknown movement command: {action}")
            return False
            
    except Exception as e:
        logging.error(f"Movement execution error: {e}")
        if tts_service:
            tts_service.speak(create_error_speech('movement'))
        return False

def execute_vision_command(action: str, target_object: str, vision_system, tts_service):
    """Execute vision-based command and provide audio feedback"""
    try:
        if not vision_system or not vision_system.is_initialized:
            error_msg = "Vision system not available"
            logging.error(error_msg)
            if tts_service:
                tts_service.speak("Sorry, my vision system is not working right now.")
            return False
        
        # Create movement function references
        movement_functions = {
            'forward': lambda duration=1: (forward(), time.sleep(duration), stop()),
            'left': lambda duration=1: (left(), time.sleep(duration), stop()),
            'right': lambda duration=1: (right(), time.sleep(duration), stop()),
            'stop': lambda: stop()
        }
        
        if action == 'navigate_to_object':
            if not target_object:
                if tts_service:
                    tts_service.speak("I need to know what object to find.")
                return False
            
            if tts_service:
                tts_service.speak(f"Looking for {target_object}. I'll move toward it when I find it.")
            
            # Execute navigation
            result = vision_system.navigate_to_object(target_object, movement_functions)
            
            if result['success']:
                if tts_service:
                    tts_service.speak(result['message'])
                logging.info(f"Vision navigation successful: {result['message']}")
            else:
                if tts_service:
                    tts_service.speak(f"I couldn't find or reach the {target_object}. {result['message']}")
                logging.warning(f"Vision navigation failed: {result['message']}")
            
            return result['success']
            
        elif action == 'scan_area':
            if tts_service:
                tts_service.speak("Scanning the area...")
            
            result = vision_system.scan_area()
            
            if result['success']:
                if tts_service:
                    tts_service.speak(result['message'])
                logging.info(f"Area scan: {result['message']}")
            else:
                if tts_service:
                    tts_service.speak("I'm having trouble seeing the area right now.")
                logging.error(f"Area scan failed: {result['message']}")
            
            return result['success']
            
        elif action == 'capture_image':
            if tts_service:
                tts_service.speak("Taking a picture...")
            
            result = vision_system.capture_image()
            
            if result['success']:
                if tts_service:
                    objects_msg = f" with {result['objects_detected']} objects detected" if result['objects_detected'] > 0 else ""
                    tts_service.speak(f"Picture saved{objects_msg}.")
                logging.info(f"Image capture: {result['message']}")
            else:
                if tts_service:
                    tts_service.speak("I couldn't take the picture.")
                logging.error(f"Image capture failed: {result['message']}")
            
            return result['success']
        
        else:
            logging.warning(f"Unknown vision action: {action}")
            return False
            
    except Exception as e:
        logging.error(f"Vision command execution error: {e}")
        if tts_service:
            tts_service.speak("There was an error with my vision system.")
        return False

# Initialize services
ai_service, tts_service, vision_system = setup_robot_services()

recognizer = sr.Recognizer()
mic = sr.Microphone()

# Conversation context for multi-turn conversations
conversation_context = []

wake_word = os.getenv('WAKE_WORD', 'hello olaf')
robot_name = os.getenv('ROBOT_NAME', 'Olaf')

print(f"Say '{wake_word}' to activate {robot_name}...")
if tts_service:
    tts_service.speak(f"Say {wake_word} to activate me!")

try:
    # Wake word loop
    while True:
        with mic as source:
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source)

        try:
            wake_command = recognizer.recognize_google(audio).lower()
            print(f"Heard: {wake_command}")

            if wake_word in wake_command:
                print(f"Wake word detected. {robot_name} is now active!")
                if tts_service:
                    tts_service.speak(create_greeting_speech())
                break
        except sr.UnknownValueError:
            print("Didn't catch that. Try again.")
        except sr.RequestError as e:
            print(f"Speech recognition error: {e}")
            if tts_service:
                tts_service.speak(create_error_speech('speech_recognition'))

    # Main interaction loop
    print(f"{robot_name} is ready! You can give me movement commands or ask questions.")
    
    while True:
        try:
            with mic as source:
                recognizer.adjust_for_ambient_noise(source)
                print("Listening...")
                audio = recognizer.listen(source, timeout=30)

            # Convert speech to text
            user_input = recognizer.recognize_google(audio).lower()
            print(f"You said: {user_input}")
            
            # Check for exit commands
            if any(word in user_input for word in ['goodbye', 'bye', 'exit', 'quit']):
                print("Goodbye!")
                if tts_service:
                    tts_service.speak("Goodbye! It was nice talking with you!")
                break
            
            # Process input with AI service
            if ai_service:
                result = ai_service.process_user_input(user_input, conversation_context)
                
                # Handle different types of responses
                if result['intent'] == 'movement' and result['action']:
                    # Execute movement command
                    execute_movement_command(result['action'], tts_service)
                
                elif result['intent'] == 'vision' and result['action']:
                    # Execute vision command
                    target_object = result.get('target_object')
                    execute_vision_command(result['action'], target_object, vision_system, tts_service)
                    
                elif result['intent'] in ['question', 'conversation']:
                    # Provide AI-generated response
                    response_text = result['response_text']
                    print(f"{robot_name}: {response_text}")
                    
                    if tts_service:
                        tts_service.speak(response_text)
                    
                    # Update conversation context
                    conversation_context.append({"role": "user", "content": user_input})
                    conversation_context.append({"role": "assistant", "content": response_text})
                    
                    # Keep context manageable (last 6 messages)
                    if len(conversation_context) > 6:
                        conversation_context = conversation_context[-6:]
                
                else:
                    # Unknown intent
                    response_text = result['response_text']
                    print(f"{robot_name}: {response_text}")
                    if tts_service:
                        tts_service.speak(response_text)
                        
            else:
                # Fallback without AI service
                print("AI service unavailable. Processing basic commands...")
                if tts_service:
                    tts_service.speak("AI service unavailable. I can only do basic movement commands.")
                
                # Simple keyword matching for movement
                if any(word in user_input for word in ['forward', 'ahead']):
                    execute_movement_command('forward', tts_service)
                elif any(word in user_input for word in ['backward', 'back']):
                    execute_movement_command('backward', tts_service)
                elif 'left' in user_input:
                    execute_movement_command('left', tts_service)
                elif 'right' in user_input:
                    execute_movement_command('right', tts_service)
                elif 'stop' in user_input:
                    execute_movement_command('stop', tts_service)
                else:
                    print("I didn't understand that command.")
                    if tts_service:
                        tts_service.speak("I didn't understand that command.")

        except sr.WaitTimeoutError:
            print("Listening timeout - say something!")
            if tts_service:
                tts_service.speak("I'm still here! Say something.")
                
        except sr.UnknownValueError:
            print("Could not understand audio")
            if tts_service:
                tts_service.speak(create_error_speech('speech_recognition'))
                
        except sr.RequestError as e:
            print(f"Could not request results; {e}")
            if tts_service:
                tts_service.speak(create_error_speech('speech_recognition'))
        
        except KeyboardInterrupt:
            print("Exiting...")
            break
        
        except Exception as e:
            logging.error(f"Unexpected error: {e}")
            print("An unexpected error occurred.")
            if tts_service:
                tts_service.speak(create_error_speech('general'))

        time.sleep(0.5)  # Brief pause between interactions

except KeyboardInterrupt:
    print("Exiting...")
    if tts_service:
        tts_service.speak("Shutting down. Goodbye!")
    if vision_system:
        vision_system.cleanup()
    GPIO.cleanup()
    
except Exception as e:
    logging.error(f"Fatal error: {e}")
    print("A fatal error occurred. Shutting down.")
    if tts_service:
        tts_service.speak("A fatal error occurred. Shutting down.")
    if vision_system:
        vision_system.cleanup()
    GPIO.cleanup()

finally:
    # Ensure cleanup
    try:
        if vision_system:
            vision_system.cleanup()
        GPIO.cleanup()
        print("Cleanup completed.")
    except:
        pass
