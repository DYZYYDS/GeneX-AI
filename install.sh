#!/bin/bash
echo "======================================================="
echo "    XenoGenesis OpenSource Environment Installer (Linux/Mac)"
echo "======================================================="
echo ""

if ! command -v python3 &> /dev/null
then
    echo "[ERROR] python3 is not installed or not in PATH."
    echo "Please install Python 3.10 or higher."
    exit 1
fi

echo "[INFO] Creating virtual environment..."
python3 -m venv venv
if [ $? -ne 0 ]; then
    echo "[ERROR] Failed to create virtual environment. You may need to run: sudo apt install python3-venv"
    exit 1
fi

echo "[INFO] Activating virtual environment and installing dependencies..."
source venv/bin/activate
python3 -m pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "======================================================="
echo "[SUCCESS] Environment setup complete!"
echo ""
echo "To start the system, run:"
echo "  1. source venv/bin/activate"
echo "  2. python run_gene_research.py"
echo "======================================================="
