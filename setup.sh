#!/bin/bash

menu() {
    clear
    echo "================================"
    echo "   Python Program Manager"
    echo "================================"
    echo "1. Install Dependencies"
    echo "2. Run Program"
    echo "3. Exit"
    echo

    read -p "Enter your choice (1-3): " choice

    case $choice in
        1) setup ;;
        2) run ;;
        3) exit 0 ;;
        *) 
            echo "Invalid option"
            sleep 1
            menu
            ;;
    esac
}

setup() {
    echo "Checking for required packages..."

    # Check if Python 3 is installed
    if ! command -v python3 &> /dev/null; then
        echo "Python 3 is not installed. Installing..."
        sudo apt-get update
        sudo apt-get install -y python3 python3-pip python3-venv
    fi

    # Create virtual environment if it doesn't exist
    if [ ! -d "venv" ]; then
        echo "Creating virtual environment..."
        python3 -m venv venv
        echo "Virtual environment created!"
    else
        echo "Virtual environment already exists."
    fi

    # Activate virtual environment
    echo "Activating virtual environment..."
    source venv/bin/activate

    # Upgrade pip
    echo "Upgrading pip..."
    pip install --upgrade pip

    # Check if requirements.txt exists and install if it does
    if [ -f "requirements.txt" ]; then
        echo "Installing Python dependencies..."
        pip install -r requirements.txt
    else
        echo "No requirements.txt found. Skipping package installation."
    fi

    # Create config.py from template if it doesn't exist
    if [ ! -f "config.py" ]; then
        if [ -f "config.template.py" ]; then
            echo "Creating config.py from template..."
            cp config.template.py config.py
            echo "✓ config.py created"
        else
            echo "⚠️  Warning: config.template.py not found!"
        fi
    else
        echo "✓ config.py already exists"
    fi

    # Create .env from example if it doesn't exist
    CONFIG_NEEDED=false
    if [ ! -f ".env" ]; then
        if [ -f ".env.example" ]; then
            echo "Creating .env from example..."
            cp .env.example .env
            echo "✓ .env created"
            CONFIG_NEEDED=true
        else
            echo "⚠️  Warning: .env.example not found!"
        fi
    else
        echo "✓ .env already exists"
        # Check if API key is still placeholder
        if grep -q "PASTE_YOUR_ACTUAL_API_KEY_HERE" .env 2>/dev/null || \
           grep -q "your_anthropic_api_key_here" .env 2>/dev/null; then
            CONFIG_NEEDED=true
        fi
    fi

    echo ""
    echo "================================"
    echo "   Setup Complete!"
    echo "================================"

    if [ "$CONFIG_NEEDED" = true ]; then
        echo ""
        echo "⚠️  IMPORTANT: Configure your API keys!"
        echo ""
        echo "Edit the .env file and add your Anthropic API key:"
        echo "  nano .env"
        echo ""
        echo "Replace this line:"
        echo "  ANTHROPIC_API_KEY=PASTE_YOUR_ACTUAL_API_KEY_HERE"
        echo ""
        echo "With your actual key:"
        echo "  ANTHROPIC_API_KEY=sk-ant-api03-xxxxx..."
        echo ""
        echo "Get your API key at: https://console.anthropic.com/"
        echo ""
        read -p "Press Enter once you've configured your API key..."
    else
        echo "✓ Virtual environment is activated"
        echo "✓ Dependencies installed"
        echo "✓ Configuration files ready"
        sleep 2
    fi

    menu
}

run() {
    # Activate virtual environment if it exists
    if [ -d "venv" ]; then
        echo "Activating virtual environment..."
        source venv/bin/activate
    fi

    # Check if config.py exists
    if [ ! -f "config.py" ]; then
        echo "⚠️  Error: config.py not found!"
        echo "Please run option 1 (Install Dependencies) first."
        sleep 2
        menu
        return
    fi

    # Check if .env exists
    if [ ! -f ".env" ]; then
        echo "⚠️  Error: .env file not found!"
        echo "Please run option 1 (Install Dependencies) first."
        sleep 2
        menu
        return
    fi

    # Check if API key is configured
    if grep -q "PASTE_YOUR_ACTUAL_API_KEY_HERE" .env 2>/dev/null || \
       grep -q "your_anthropic_api_key_here" .env 2>/dev/null; then
        echo ""
        echo "⚠️  API Key Not Configured!"
        echo ""
        echo "You need to add your Anthropic API key to the .env file."
        echo ""
        echo "Edit .env file:"
        echo "  nano .env"
        echo ""
        echo "Replace:"
        echo "  ANTHROPIC_API_KEY=PASTE_YOUR_ACTUAL_API_KEY_HERE"
        echo ""
        echo "With your actual key:"
        echo "  ANTHROPIC_API_KEY=sk-ant-api03-xxxxx..."
        echo ""
        echo "Get your API key at: https://console.anthropic.com/"
        echo ""
        read -p "Press Enter to return to menu..."
        menu
        return
    fi

    # Run the Streamlit application
    echo ""
    echo "================================"
    echo "   Starting ShelfLife..."
    echo "================================"
    echo ""
    streamlit run shelflife.py

    echo ""
    read -p "Press Enter to continue..."
    menu
}

# Make the script executable on first run
chmod +x setup.sh

# Start the menu
menu 