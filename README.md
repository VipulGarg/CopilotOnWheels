# Robot Assistant with Azure AI Integration

A voice-controlled robot assistant that combines physical movement capabilities with AI-powered conversation and question answering using Azure OpenAI services.

## Features

🤖 **Voice-Controlled Movement**: Control robot movement with voice commands (forward, backward, left, right, stop)

🧠 **AI-Powered Conversations**: Ask questions and have conversations powered by Azure OpenAI

🗣️ **Text-to-Speech Responses**: Robot speaks back answers and confirmations

🎯 **Intent Recognition**: Automatically distinguishes between movement commands and questions

💬 **Context-Aware**: Maintains conversation context for natural multi-turn discussions

## Hardware Requirements

- Raspberry Pi (3B+ or newer recommended)
- Robot car chassis with motors
- Motor driver (L298N or similar)
- USB microphone
- Speakers or audio output device
- Internet connection for Azure AI services

## Software Requirements

- Python 3.7+
- Azure OpenAI account and API key
- Required Python packages (see requirements.txt)

## Quick Start

### 1. Clone and Setup

```bash
git clone <your-repo>
cd RobotDIY
chmod +x setup.sh
./setup.sh
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
source robot_env/bin/activate
python3 main.py
```

## Usage

### Wake Word Activation
Say "Hello Olaf" (or your configured wake word) to activate the robot.

### Movement Commands
- "Move forward" / "Go ahead"
- "Go backward" / "Reverse"
- "Turn left"
- "Turn right"  
- "Stop"

### Questions & Conversations
- "What's the weather like?" (AI will explain it needs internet)
- "Tell me a joke"
- "What can you do?"
- "How do robots work?"
- Any general question or conversation

### Exit
Say "goodbye", "bye", "exit", or "quit" to shut down the robot.

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

### GPIO Pin Configuration (main.py)

```python
IN1 = 23  # Motor 1 control pin 1
IN2 = 24  # Motor 1 control pin 2
IN3 = 27  # Motor 2 control pin 1
IN4 = 22  # Motor 2 control pin 2
ENA = 25  # Motor 1 enable pin
ENB = 17  # Motor 2 enable pin
```

## Architecture

### Core Components

1. **main.py**: Main program orchestrating all components
2. **robot_ai.py**: Azure OpenAI integration for intent recognition and Q&A
3. **robot_tts.py**: Text-to-speech functionality
4. **requirements.txt**: Python dependencies
5. **.env**: Configuration and credentials

### AI Processing Flow

1. **Speech Recognition**: Convert voice to text using Google Speech Recognition
2. **Intent Classification**: Azure OpenAI determines if input is movement command or question
3. **Action Execution**: 
   - Movement commands → Control motors
   - Questions → Generate AI response
4. **Text-to-Speech**: Speak response back to user
5. **Context Management**: Maintain conversation history for natural dialogue

## Troubleshooting

### Common Issues

**"No module named 'robot_ai'"**
- Make sure you're in the correct directory and virtual environment is activated

**TTS not working**
- Check audio devices: `aplay -l`
- Test speakers: `speaker-test -t wav -c 2`
- Install additional audio packages: `sudo apt install alsa-utils pulseaudio`

**Azure OpenAI errors**
- Verify credentials in `.env` file
- Check Azure OpenAI resource and deployment names
- Ensure you have sufficient quota

**Speech recognition issues**
- Check microphone: `arecord -l`
- Test microphone: `arecord -d 5 test.wav && aplay test.wav`
- Adjust microphone sensitivity in the code

### Debug Mode

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