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
            echo "Please edit config.py with your API keys and settings!"
        fi
    fi

    # Create .env from example if it doesn't exist
    if [ ! -f ".env" ]; then
        if [ -f ".env.example" ]; then
            echo "Creating .env from example..."
            cp .env.example .env
            echo "Please edit .env with your API keys!"
        fi
    fi

    echo ""
    echo "Setup complete!"
    echo "Virtual environment is activated."
    echo "Don't forget to configure your API keys in .env file!"
    sleep 2
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
        echo "Warning: config.py not found!"
        echo "Please run option 1 (Install Dependencies) first."
        sleep 2
        menu
        return
    fi

    # Run the Streamlit application
    echo "Starting ShelfLife..."
    streamlit run shelflife.py
    read -p "Press Enter to continue..."
    menu
}

# Make the script executable on first run
chmod +x setup.sh

# Start the menu
menu 