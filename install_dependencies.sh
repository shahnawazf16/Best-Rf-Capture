#!/bin/bash
echo "Installing RF Capture dependencies..."

# Update package list
sudo apt update

# Install system dependencies
sudo apt install -y python3-pip python3-venv libhdf5-dev

# Create virtual environment
python3 -m venv myenv
source myenv/bin/activate

# Install Python packages
pip install --upgrade pip
pip install numpy scipy tensorflow matplotlib pyyaml scikit-learn h5py tqdm

# Install RTL-SDR support
pip install pyrtlsdr

echo "Installation complete!"
echo "Activate virtual environment: source myenv/bin/activate"
echo "Install package: pip install -e ."
