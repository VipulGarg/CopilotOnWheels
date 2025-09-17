# Robot Vision System Documentation

## Overview

This robot now has complete voice-controlled object navigation capabilities! Here's how the system works:

## System Architecture

```
Voice Input → Azure OpenAI → Intent Recognition → Vision System → Motor Control
```

### Components

1. **Azure OpenAI Integration** (`robot_ai.py`)
   - Recognizes voice commands and classifies intents
   - Supports three types of intents:
     - `movement`: Basic movement commands (forward, backward, left, right, stop)
     - `vision`: Object detection and navigation commands
     - `question`: General knowledge queries

2. **Vision System** (`src/vision/`)
   - **Camera Module** (`camera.py`): Handles PiCamera2 or USB camera
   - **Object Detection** (`object_detection.py`): Uses YOLO for real-time object detection
   - **Navigation System** (`navigation.py`): Combines camera and detection for autonomous navigation

3. **Main Controller** (`src/main.py`)
   - Orchestrates all components
   - Handles voice recognition and text-to-speech
   - Executes commands based on AI intent classification

## Voice Commands Supported

### Vision/Navigation Commands
- **"move to the cup"** → Finds and navigates to a cup
- **"go to the chair"** → Finds and navigates to a chair  
- **"find a person"** → Looks for and moves toward a person
- **"what do you see?"** → Scans area and describes visible objects
- **"take a picture"** → Captures and saves an annotated image

### Basic Movement Commands
- **"move forward"** → Moves forward for 2 seconds
- **"turn left"** → Turns left for 2 seconds
- **"go backward"** → Moves backward for 2 seconds
- **"stop"** → Stops all movement

### Conversation
- **"what's the weather?"** → Uses AI to respond to general questions
- **"hello"** → Casual conversation with the robot

## How Object Navigation Works

1. **Voice Recognition**: User says "move to the cup"

2. **Intent Classification**: Azure OpenAI identifies this as a `vision` intent with action `navigate_to_object` and target_object `cup`

3. **Vision Processing**:
   - Camera captures live video frames
   - YOLO detects objects in each frame
   - System searches for objects matching "cup"

4. **Navigation Logic**:
   - If object not found: Robot rotates to scan the area
   - If object found: Robot calculates movement direction
   - Robot moves toward object until it's close enough (15% of frame)

5. **Movement Control**:
   - **Object to the left** → Turn left to center it
   - **Object to the right** → Turn right to center it  
   - **Object centered** → Move forward toward it
   - **Object large enough** → Stop (arrived at target)

## Detectable Objects

The system can detect and navigate to 80+ different objects including:
- **People**: person
- **Furniture**: chair, couch, bed, dining table
- **Electronics**: tv, laptop, cell phone, remote
- **Kitchen items**: cup, bottle, bowl, microwave, refrigerator
- **Animals**: cat, dog, bird
- **Vehicles**: car, bicycle, motorcycle
- **And many more...**

## Installation Requirements

```bash
# Computer Vision
pip install opencv-python==4.8.1.78
pip install ultralytics==8.0.196
pip install picamera2==0.3.12  # For Raspberry Pi camera
pip install numpy==1.24.3
pip install pillow==10.0.0

# Existing dependencies
pip install -r requirements.txt
```

## Usage

1. **Start the robot**: `python src/main.py`

2. **Activate with wake word**: Say "hello olaf" (or your configured wake word)

3. **Give vision commands**:
   ```
   You: "move to the cup"
   Robot: "I'll look for a cup and move to it!"
   [Robot starts scanning with camera, detects cup, navigates toward it]
   Robot: "Successfully reached the cup"
   ```

4. **Ask what it sees**:
   ```
   You: "what do you see?"
   Robot: "Let me scan the area..."
   Robot: "I can see a chair, a laptop, and a cup"
   ```

5. **Take photos**:
   ```
   You: "take a picture"
   Robot: "Taking a picture..."
   Robot: "Photo saved with 3 objects detected"
   ```

## Safety Features

- **Timeout Protection**: Navigation stops after 30 seconds if object not reached
- **Movement Limits**: Short movement bursts (0.3 seconds) for precise control
- **Emergency Stop**: Say "goodbye" or press Ctrl+C to stop immediately
- **Fallback Mode**: If vision system fails, basic movement commands still work

## Configuration

### Environment Variables (.env file)
```bash
# Azure OpenAI
AZURE_OPENAI_ENDPOINT=your-endpoint
AZURE_OPENAI_API_KEY=your-api-key
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4

# Robot Settings
ROBOT_NAME=Olaf
WAKE_WORD=hello olaf
```

### Hardware Setup
- **Raspberry Pi 4** (recommended)
- **L298N Motor Driver**
- **PiCamera2 or USB Camera**
- **Two DC Motors**
- **Microphone and Speaker**

## Troubleshooting

### Vision System Not Working
- Check camera connection: `python -c "from src.vision.camera import test_camera; test_camera()"`
- Verify YOLO installation: `python -c "from ultralytics import YOLO; print('YOLO OK')"`

### Object Not Detected
- Ensure good lighting conditions
- Object might not be in YOLO's training set (try common objects first)
- Check if camera is focused properly

### Navigation Issues
- Robot might be moving too fast/slow (adjust `movement_duration` in navigation.py)
- Check motor connections and GPIO pins
- Ensure adequate battery power

## Example Session

```
$ python src/main.py

Robot services initialized successfully!
Say 'hello olaf' to activate me!

You: hello olaf
Robot: Hello! I'm ready to help you. I can move around and see objects with my camera!

You: what do you see?
Robot: Let me scan the area... I can see a chair and a laptop.

You: move to the chair
Robot: I'll look for a chair and move to it!
[Robot rotates, finds chair, moves toward it]
Robot: Successfully reached the chair

You: take a picture
Robot: Taking a picture... Photo saved with 2 objects detected.

You: goodbye
Robot: Goodbye! It was nice talking with you!
```

## Advanced Features

### Custom Object Training
- Replace YOLO model with custom-trained version for specific objects
- Add object synonyms in `object_detection.py`

### Navigation Tuning
- Adjust `target_area_threshold` for stopping distance
- Modify `center_tolerance` for centering precision
- Change `movement_duration` for navigation speed

### Multiple Camera Support  
- System automatically detects PiCamera2 vs USB camera
- Falls back gracefully if one camera type fails

This system provides a complete voice-to-vision pipeline that enables natural language control of robot navigation and object interaction!