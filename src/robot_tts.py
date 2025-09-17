"""
Text-to-Speech Module for Robot Assistant
Handles converting AI responses to speech output
"""

import pyttsx3
import threading
import logging
from typing import Optional
import os

class RobotTTS:
    def __init__(self):
        """Initialize text-to-speech engine"""
        try:
            self.engine = pyttsx3.init()
            self.setup_voice()
            self.is_speaking = False
            self._setup_logging()
        except Exception as e:
            logging.error(f"Failed to initialize TTS engine: {e}")
            self.engine = None
    
    def _setup_logging(self):
        """Setup logging for TTS module"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
    
    def setup_voice(self):
        """Configure voice properties"""
        if not self.engine:
            return
            
        try:
            # Get available voices
            voices = self.engine.getProperty('voices')
            
            # Try to set a more pleasant voice (prefer female voices for assistant)
            if voices:
                for voice in voices:
                    if 'english' in voice.name.lower() or 'zira' in voice.name.lower():
                        self.engine.setProperty('voice', voice.id)
                        break
                else:
                    # If no female voice found, use the first available
                    self.engine.setProperty('voice', voices[0].id)
            
            # Set speech rate (words per minute)
            self.engine.setProperty('rate', 160)  # Slightly faster than default
            
            # Set volume (0.0 to 1.0)
            self.engine.setProperty('volume', 0.9)
            
            logging.info("TTS voice configured successfully")
            
        except Exception as e:
            logging.error(f"Failed to setup voice: {e}")
    
    def speak(self, text: str, wait: bool = True) -> bool:
        """
        Convert text to speech
        
        Args:
            text: Text to speak
            wait: If True, block until speech is complete
        
        Returns:
            bool: True if speech was initiated successfully
        """
        if not self.engine or not text.strip():
            return False
        
        try:
            logging.info(f"Speaking: {text}")
            
            if wait:
                self.is_speaking = True
                self.engine.say(text)
                self.engine.runAndWait()
                self.is_speaking = False
            else:
                # Speak in background thread
                thread = threading.Thread(target=self._speak_async, args=(text,))
                thread.daemon = True
                thread.start()
            
            return True
            
        except Exception as e:
            logging.error(f"Speech error: {e}")
            return False
    
    def _speak_async(self, text: str):
        """Speak text in background thread"""
        try:
            self.is_speaking = True
            self.engine.say(text)
            self.engine.runAndWait()
            self.is_speaking = False
        except Exception as e:
            logging.error(f"Async speech error: {e}")
            self.is_speaking = False
    
    def stop_speaking(self):
        """Stop current speech"""
        if self.engine:
            try:
                self.engine.stop()
                self.is_speaking = False
            except Exception as e:
                logging.error(f"Error stopping speech: {e}")
    
    def is_busy(self) -> bool:
        """Check if TTS is currently speaking"""
        return self.is_speaking
    
    def test_speech(self):
        """Test the TTS system"""
        test_message = f"Hello! I am {os.getenv('ROBOT_NAME', 'Olaf')}, your robot assistant. Text to speech is working correctly!"
        return self.speak(test_message, wait=True)

# Utility functions for common robot responses
def create_movement_speech(action: str) -> str:
    """Generate speech text for movement actions"""
    movement_phrases = {
        'forward': "Moving forward now!",
        'backward': "Going backward!",
        'left': "Turning left!",
        'right': "Turning right!",
        'stop': "Stopping now!"
    }
    return movement_phrases.get(action, f"Executing {action} movement!")

def create_error_speech(error_type: str = "general") -> str:
    """Generate speech text for error situations"""
    error_phrases = {
        'speech_recognition': "Sorry, I didn't catch that. Could you repeat?",
        'ai_service': "I'm having trouble connecting to my AI services right now.",
        'movement': "I'm having trouble with my movement systems.",
        'general': "Sorry, something went wrong. Please try again."
    }
    return error_phrases.get(error_type, error_phrases['general'])

def create_greeting_speech() -> str:
    """Generate greeting speech"""
    robot_name = os.getenv('ROBOT_NAME', 'Olaf')
    return f"Hello! I'm {robot_name}, your robot assistant. I can move around and answer your questions. How can I help you?"