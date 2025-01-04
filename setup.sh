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

    # Update package list
    echo "Updating package list..."
    sudo apt-get update

    # Check if pip is installed
    if ! command -v pip3 &> /dev/null; then
        echo "Installing pip..."
        sudo apt-get install -y python3-pip
    fi

    # Check if requirements.txt exists and install if it does
    if [ -f "requirements.txt" ]; then
        echo "Installing Python dependencies..."
        pip3 install -r requirements.txt
    else
        echo "No requirements.txt found. Skipping package installation."
    fi

    echo "Setup complete!"
    sleep 1
    menu
}

run() {
    # Replace 'main.py' with your actual Python file name
    python3 main.py
    read -p "Press Enter to continue..."
    menu
}

# Make the script executable on first run
chmod +x setup.sh

# Start the menu
menu 