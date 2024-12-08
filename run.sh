#!/bin/bash

# Display banner
echo "   ___                            ____                  _              ____  ";
echo "  / _ \   _ __     ___   _ __    / ___|  _ __    ___   | | __         |___ \ ";
echo " | | | | | '_ \   / _ \ | '_ \  | |  _  | '__|  / _ \  | |/ /  _____    __) |";
echo " | |_| | | |_) | |  __/ | | | | | |_| | | |    | (_) | |   <  |_____|  / __/ ";
echo "  \___/  | .__/   \___| |_| |_|  \____| |_|     \___/  |_|\_\         |_____|";
echo "         |_|                                                                 ";
# Function to check Python version
check_python() {
    # First try Python 3.10
    if command -v python3.10 &> /dev/null; then
        PYTHON_CMD="python3.10"
    else
        # Try to find any Python 3.x
        for ver in {9..11}; do
            if command -v python3.$ver &> /dev/null; then
                PYTHON_CMD="python3.$ver"
                break
            fi
        done
    fi

    # Check if we found a suitable Python version
    if [ -z "$PYTHON_CMD" ]; then
        echo "Error: No suitable Python 3.x version found. Please install Python 3.x"
        exit 1
    fi

    echo "Using Python: $($PYTHON_CMD --version)"
}

# Function to check system requirements
check_system_requirements() {
    # Check RAM (in GB)
    total_ram=$(free -g | awk '/^Mem:/{print $2}')
    if [ $total_ram -lt 8 ]; then
        echo "Warning: System RAM ($total_ram GB) is less than recommended (8 GB)"
        echo "The application may not run properly"
    fi

    # Check VRAM using nvidia-smi (if available)
    if command -v nvidia-smi &> /dev/null; then
        vram=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | awk '{print $1/1024}')
        if (( $(echo "$vram < 12" | bc -l) )); then
            echo "Warning: GPU VRAM ($vram GB) is less than recommended (12 GB)"
            echo "The application may not run properly"
        fi
    else
        echo "Warning: Unable to detect GPU. Please ensure you have a compatible GPU with at least 12GB VRAM"
    fi
}

# Main execution
echo "Checking system requirements..."
check_system_requirements

echo "Checking Python installation..."
check_python

# Check if dataset directory exists
if [ ! -d "dataset" ]; then
    echo "Error: dataset directory not found"
    exit 1
fi

# Check for firstrun.lock
if [ ! -f "dataset/firstrun.lock" ]; then
    echo "First run detected. Starting training..."
    pip install -r requirements.txt
    $PYTHON_CMD train.py
    if [ $? -ne 0 ]; then
        echo "Error: Training failed"
        exit 1
    fi
fi

echo "Starting chat interface..."
$PYTHON_CMD -m streamlit run chat.py

