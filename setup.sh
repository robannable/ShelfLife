#!/bin/bash

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
source venv/Scripts/activate

# Upgrade pip
python -m pip install --upgrade pip

# Install dependencies
pip install -e .

# Install requirements
pip install -r requirements.txt

# Run the Streamlit app
streamlit run shelflife.py

echo "Setup complete! Virtual environment is activated."
echo "To deactivate, run: deactivate" 