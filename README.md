# CopilotOnWheels - AI-Powered Vision Robot

A voice-controlled robot assistant that combines physical movement capabilities with computer vision, object detection, autonomous navigation, and AI-powered conversation using Azure OpenAI services.

## Features

🤖 **Voice-Controlled Movement**: Control robot movement with voice commands (forward, backward, left, right, stop)

👁️ **Computer Vision & Object Detection**: Real-time object detection using YOLO for navigation and environment understanding

🎯 **Autonomous Navigation**: Voice-commanded object finding and navigation ("move to the cup", "find a person")

📸 **Visual Intelligence**: Take pictures, scan environment, and describe what the robot sees

🧠 **AI-Powered Conversations**: Ask questions and have conversations powered by Azure OpenAI

🗣️ **Text-to-Speech Responses**: Robot speaks back answers and confirmations

🎯 **Intent Recognition**: Automatically distinguishes between movement, vision, and conversation commands

💬 **Context-Aware**: Maintains conversation context for natural multi-turn discussions

## Hardware Requirements

- Raspberry Pi (4B recommended for vision processing)
- Robot car chassis with motors
- Motor driver (L298N or similar)
- **Camera**: PiCamera2 or USB camera for vision system
- USB microphone
- Speakers or audio output device
- Internet connection for Azure AI services

## Software Requirements

- Python 3.7+
- Azure OpenAI account and API key
- Required Python packages (see requirements.txt)
- **Computer Vision Dependencies**:
  - OpenCV (opencv-python==4.8.1.78)
  - YOLO (ultralytics==8.0.196)
  - PiCamera2 (for Raspberry Pi camera)
  - NumPy and Pillow for image processing

## Quick Start

### 1. Clone and Setup

```bash
git clone <repo-url>
cd CopilotOnWheels
bash setup.sh
```

### 2. Configure Azure Credentials

Edit `.env` file with your Azure OpenAI credentials:

```bash
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
AZURE_OPENAI_API_KEY=your-api-key-here
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4
```

### 3. Run the Robot

```bash
1. Edit .env file with your Azure OpenAI credentials
2. Activate virtual environment: source robot_env/bin/activate
3. sudo apt remove python3-rpi.gpio  # Only if on Raspberry Pi and not needed
4. pip3 install rpi-lgpio pyaudio
5. pip3 install httpx==0.27.2
6. Run the robot: python3 src/main.py
```

## Usage

### Wake Word Activation
Say "Hello Olaf" (or your configured wake word) to activate the robot.

### Voice Commands

#### Movement Commands
- "Move forward" / "Go ahead"
- "Go backward" / "Reverse"
- "Turn left"
- "Turn right"  
- "Stop"

#### Vision & Navigation Commands
- **"move to the cup"** → Finds and navigates to a cup
- **"go to the chair"** → Finds and navigates to a chair  
- **"find a person"** → Looks for and moves toward a person
- **"what do you see?"** → Scans area and describes visible objects
- **"take a picture"** → Captures and saves an annotated image

#### Questions & Conversations
- "What's the weather like?" (AI will explain it needs internet)
- "Tell me a joke"
- "What can you do?"
- "How do robots work?"
- Any general question or conversation

### Exit
Say "goodbye", "bye", "exit", or "quit" to shut down the robot.

## Example Usage Sessions

### Basic Movement and Vision
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

## Configuration Options

### Environment Variables (.env)

| Variable | Description | Default |
|----------|-------------|---------|
| `AZURE_OPENAI_ENDPOINT` | Your Azure OpenAI endpoint URL | Required |
| `AZURE_OPENAI_API_KEY` | Your Azure OpenAI API key | Required |
| `AZURE_OPENAI_DEPLOYMENT_NAME` | Model deployment name | gpt-4 |
| `ROBOT_NAME` | Robot's name for personality | Olaf |
| `WAKE_WORD` | Activation phrase | hello olaf |
| `RESPONSE_TIMEOUT` | AI response timeout (seconds) | 10 |

### Hardware Setup
- **Raspberry Pi 4** (recommended for vision processing)
- **L298N Motor Driver**
- **PiCamera2 or USB Camera**
- **Two DC Motors**
- **Microphone and Speaker**

## Safety Features

- **Timeout Protection**: Navigation stops after 30 seconds if object not reached
- **Movement Limits**: Short movement bursts (0.3 seconds) for precise control
- **Emergency Stop**: Say "goodbye" or press Ctrl+C to stop immediately
- **Fallback Mode**: If vision system fails, basic movement commands still work


## System Architecture

The robot uses a sophisticated pipeline that combines voice processing, AI intent recognition, and computer vision:

```
Voice Input → Speech Recognition → Azure OpenAI → Intent Classification → Action Execution
                                                        ↓
                                               Movement | Vision | Conversation
                                                        ↓
                                            Camera → YOLO → Navigation → Motor Control
```

### Core Components

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

## How Object Navigation Works

When you give the robot a vision command like "move to the cup", here's what happens:

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

### Detectable Objects

The system can detect and navigate to 80+ different objects including:
- **People**: person
- **Furniture**: chair, couch, bed, dining table
- **Electronics**: tv, laptop, cell phone, remote
- **Kitchen items**: cup, bottle, bowl, microwave, refrigerator
- **Animals**: cat, dog, bird
- **Vehicles**: car, bicycle, motorcycle
- **And many more...**

### 3. Install Dependencies

```bash
# Install all requirements including computer vision
pip install -r requirements.txt

# Additional vision system dependencies
pip install opencv-python==4.8.1.78
pip install ultralytics==8.0.196
pip install picamera2==0.3.12  # For Raspberry Pi camera
pip install numpy==1.24.3
pip install pillow==10.0.0
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

## Troubleshooting

### Vision System Issues
- **Camera not working**: Check camera connection: `python -c "from src.vision.camera import test_camera; test_camera()"`
- **YOLO not installed**: Verify installation: `python -c "from ultralytics import YOLO; print('YOLO OK')"`
- **Object not detected**: Ensure good lighting conditions and try common objects first
- **Navigation problems**: Check motor connections, GPIO pins, and battery power

### Common Issues
- **"No module named 'robot_ai'"**: Make sure you're in the correct directory and virtual environment is activated
- **TTS not working**: Check audio devices: `aplay -l` and test speakers: `speaker-test -t wav -c 2`
- **Azure OpenAI errors**: Verify credentials in `.env` file and check resource/deployment names
- **Speech recognition issues**: Test microphone: `arecord -d 5 test.wav && aplay test.wav`

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

## GPIO Pin Configuration

```python
# Motor control pins (in main.py)
IN1 = 23  # Motor 1 control pin 1
IN2 = 24  # Motor 1 control pin 2
IN3 = 27  # Motor 2 control pin 1
IN4 = 22  # Motor 2 control pin 2
ENA = 25  # Motor 1 enable pin
ENB = 17  # Motor 2 enable pin
```

---

**CopilotOnWheels** provides a complete voice-to-vision pipeline that enables natural language control of robot navigation and object interaction! 🤖👁️

Add debug logging to see detailed information:

```python
logging.basicConfig(level=logging.DEBUG)
```

## Safety Features

- **Auto-stop**: Movement commands automatically stop after 2 seconds
- **GPIO cleanup**: Ensures proper shutdown of GPIO pins
- **Error handling**: Graceful error handling with audio feedback
- **Timeout protection**: Prevents indefinite listening/waiting

## Extending the Robot

### Adding New Movement Commands
1. Add motor control function in main.py
2. Update `movement_keywords` in `robot_ai.py`
3. Add speech response in `robot_tts.py`

### Customizing AI Personality
Edit the `system_prompt` in `robot_ai.py` to change the robot's personality and responses.

### Adding Sensors
Extend the system to include sensors (ultrasonic, camera, etc.) and incorporate sensor data into AI decision making.

## License

This project is open source and available under the MIT License.

## Contributing

Contributions welcome! Please read the contributing guidelines and submit pull requests for any improvements.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review Azure OpenAI documentation
3. Open an issue on GitHub