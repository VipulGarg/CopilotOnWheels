#!/bin/bash

# Robot Assistant Setup Script
# This script installs all dependencies and sets up the robot assistant

echo "🤖 Setting up Robot Assistant with Azure AI Integration..."

# Update system packages
echo "📦 Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install system dependencies
echo "🔧 Installing system dependencies..."
sudo apt install -y python3-pip python3-venv portaudio19-dev python3-pyaudio espeak espeak-data libespeak1 libespeak-dev

# Create virtual environment
echo "🐍 Creating Python virtual environment..."
python3 -m venv robot_env
source robot_env/bin/activate

# Install Python dependencies
echo "📚 Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Setup audio permissions
echo "🔊 Setting up audio permissions..."
sudo usermod -a -G audio $USER

# Create config from template
if [ ! -f .env ]; then
    echo "⚙️ Creating config file from template..."
    cp .env.example .env
    echo "📝 Please edit .env file with your Azure credentials!"
fi

echo "✅ Setup complete!"
echo ""
echo "📋 Next steps:"
echo "1. Edit .env file with your Azure OpenAI credentials"
echo "2. Activate virtual environment: source robot_env/bin/activate"
echo "3. sudo apt remove python3-rpi.gpio  # Only if on Raspberry Pi and not needed"
echo "4. pip3 install rpi-lgpio pyaudio"
echo "5. Run the robot: python3 src/main.py"
echo ""
echo "🔗 Azure OpenAI setup guide:"
echo "   https://learn.microsoft.com/en-us/azure/cognitive-services/openai/quickstart"