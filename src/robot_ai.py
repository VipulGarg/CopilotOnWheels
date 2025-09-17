"""
Azure AI Service Integration for Robot Assistant
Handles intent recognition, general Q&A, and AI-powered responses
"""

import os
import json
from typing import Dict, Tuple, Optional
from openai import AzureOpenAI
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

class RobotAI:
    def __init__(self):
        """Initialize Azure OpenAI client and configuration"""
        self.client = AzureOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
        )
        self.deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4")
        self.robot_name = os.getenv("ROBOT_NAME", "Olaf")
        
        # System prompt for the robot assistant
        self.system_prompt = f"""You are {self.robot_name}, an intelligent robot assistant that can move around and answer questions. 
        You have the following movement capabilities: forward, backward, left, right, and stop.
        
        Your responses should be:
        1. Helpful and friendly
        2. Concise (suitable for text-to-speech)
        3. Contextually aware that you're a physical robot
        
        When users ask movement-related questions, you can execute them. For general questions, 
        provide informative answers while maintaining your robot persona."""

    def classify_intent(self, user_input: str) -> Dict[str, any]:
        """
        Classify user intent using Azure OpenAI
        Returns: {
            'intent': 'movement' | 'question' | 'conversation',
            'action': specific action if movement,
            'confidence': float,
            'response_text': text to speak back
        }
        """
        try:
            messages = [
                {
                    "role": "system",
                    "content": """You are an intent classifier for a robot assistant. Analyze the user's input and return a JSON response with:
                    - intent: 'movement' (basic movement), 'vision' (object detection/navigation), 'question' (asking for information), or 'conversation' (casual chat)
                    - action: for movement intent ('forward', 'backward', 'left', 'right', 'stop') or vision intent ('navigate_to_object', 'scan_area', 'capture_image'), otherwise null
                    - target_object: if vision intent with object navigation, specify the object name, otherwise null
                    - confidence: float between 0.0-1.0
                    - response_text: friendly response to say while performing action or answering question
                    
                    Movement keywords: forward, backward, back, left, right, turn, stop (without specific objects)
                    Vision keywords: move to, go to, find, navigate to, look for, show me, take picture, what do you see, scan
                    Question keywords: what, how, when, where, why, who, tell me, explain (general knowledge)
                    
                    Examples:
                    "move forward" -> {"intent": "movement", "action": "forward", "target_object": null, "confidence": 0.9, "response_text": "Moving forward now!"}
                    "move to the cup" -> {"intent": "vision", "action": "navigate_to_object", "target_object": "cup", "confidence": 0.9, "response_text": "I'll look for a cup and move to it!"}
                    "what do you see?" -> {"intent": "vision", "action": "scan_area", "target_object": null, "confidence": 0.8, "response_text": "Let me scan the area and tell you what I see!"}
                    "take a picture" -> {"intent": "vision", "action": "capture_image", "target_object": null, "confidence": 0.9, "response_text": "Taking a picture now!"}
                    "what's the weather?" -> {"intent": "question", "action": null, "target_object": null, "confidence": 0.8, "response_text": "I'd need internet access to check the weather, but I can help with other questions!"}
                    """
                },
                {
                    "role": "user",
                    "content": f"Classify this input: '{user_input}'"
                }
            ]

            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=messages,
                max_tokens=300,
                temperature=0.1
            )
            
            # Parse the JSON response
            result = json.loads(response.choices[0].message.content)
            return result
            
        except json.JSONDecodeError:
            # Fallback if JSON parsing fails
            return self.fallback_intent_classification(user_input)
        except Exception as e:
            logging.error(f"Intent classification error: {e}")
            return self.fallback_intent_classification(user_input)

    def fallback_intent_classification(self, user_input: str) -> Dict[str, any]:
        """Simple fallback intent classification without AI"""
        user_input_lower = user_input.lower()
        
        # Vision/navigation keywords
        vision_navigate_patterns = [
            ('move to', 'navigate_to_object'),
            ('go to', 'navigate_to_object'), 
            ('find', 'navigate_to_object'),
            ('navigate to', 'navigate_to_object'),
            ('look for', 'navigate_to_object')
        ]
        
        vision_scan_patterns = [
            'what do you see',
            'look around', 
            'scan',
            'show me what',
            'what objects'
        ]
        
        vision_capture_patterns = [
            'take picture',
            'take photo',
            'capture image',
            'take a snapshot'
        ]
        
        # Check for vision navigation commands
        for pattern, action in vision_navigate_patterns:
            if pattern in user_input_lower:
                # Extract object name (everything after the pattern)
                parts = user_input_lower.split(pattern)
                if len(parts) > 1:
                    target_object = parts[1].strip().replace('the ', '').replace('a ', '')
                    return {
                        'intent': 'vision',
                        'action': action,
                        'target_object': target_object if target_object else None,
                        'confidence': 0.8,
                        'response_text': f"I'll look for {target_object if target_object else 'that object'} and move to it!"
                    }
        
        # Check for vision scanning commands
        for pattern in vision_scan_patterns:
            if pattern in user_input_lower:
                return {
                    'intent': 'vision',
                    'action': 'scan_area',
                    'target_object': None,
                    'confidence': 0.7,
                    'response_text': "Let me scan the area and tell you what I see!"
                }
        
        # Check for vision capture commands
        for pattern in vision_capture_patterns:
            if pattern in user_input_lower:
                return {
                    'intent': 'vision',
                    'action': 'capture_image',
                    'target_object': None,
                    'confidence': 0.8,
                    'response_text': "Taking a picture now!"
                }
        
        # Movement keywords
        movement_keywords = {
            'forward': ['forward', 'ahead', 'front'],
            'backward': ['backward', 'back', 'reverse'],
            'left': ['left', 'turn left'],
            'right': ['right', 'turn right'],
            'stop': ['stop', 'halt', 'pause']
        }
        
        for action, keywords in movement_keywords.items():
            if any(keyword in user_input_lower for keyword in keywords):
                return {
                    'intent': 'movement',
                    'action': action,
                    'target_object': None,
                    'confidence': 0.7,
                    'response_text': f"Executing {action} movement!"
                }
        
        # If no movement or vision detected, treat as question
        return {
            'intent': 'question',
            'action': None,
            'target_object': None,
            'confidence': 0.6,
            'response_text': "I heard your question. Let me think about that..."
        }

    def answer_question(self, question: str) -> str:
        """
        Generate an answer to a general question using Azure OpenAI
        """
        try:
            messages = [
                {
                    "role": "system",
                    "content": self.system_prompt
                },
                {
                    "role": "user",
                    "content": question
                }
            ]

            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=messages,
                max_tokens=500,
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logging.error(f"Question answering error: {e}")
            return f"I'm sorry, I'm having trouble processing your question right now. My AI services might be unavailable."

    def generate_conversational_response(self, user_input: str, context: list = None) -> str:
        """
        Generate a conversational response for casual chat
        """
        try:
            messages = [
                {
                    "role": "system",
                    "content": self.system_prompt
                }
            ]
            
            # Add context if provided
            if context:
                messages.extend(context[-6:])  # Keep last 3 exchanges
            
            messages.append({
                "role": "user",
                "content": user_input
            })

            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=messages,
                max_tokens=300,
                temperature=0.8
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logging.error(f"Conversational response error: {e}")
            return "I'm having trouble thinking of a response right now. Can you try asking something else?"

    def process_user_input(self, user_input: str, context: list = None) -> Dict[str, any]:
        """
        Main method to process user input and return appropriate response
        """
        # First classify the intent
        intent_result = self.classify_intent(user_input)
        
        if intent_result['intent'] == 'movement':
            # For movement, return the classification result directly
            return intent_result
            
        elif intent_result['intent'] == 'vision':
            # For vision commands, return the classification result directly
            # The vision system will handle the actual execution
            return intent_result
            
        elif intent_result['intent'] == 'question':
            # Generate a detailed answer
            answer = self.answer_question(user_input)
            intent_result['response_text'] = answer
            return intent_result
            
        elif intent_result['intent'] == 'conversation':
            # Generate conversational response
            response = self.generate_conversational_response(user_input, context)
            intent_result['response_text'] = response
            return intent_result
            
        else:
            # Unknown intent
            return {
                'intent': 'unknown',
                'action': None,
                'target_object': None,
                'confidence': 0.0,
                'response_text': "I'm not sure how to help with that. You can ask me questions, tell me to move around, or ask me to find objects!"
            }