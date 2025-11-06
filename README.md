# Best-Rf-Capture
Best Rf Capture
# RF Capture - Advanced RF Signal Analysis with Machine Learning

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.8%2B-orange)
![Machine Learning](https://img.shields.io/badge/Machine-Learning-brightgreen)
![SDR](https://img.shields.io/badge/SDR-Compatible-yellow)
![License](https://img.shields.io/badge/License-MIT-green)

A comprehensive command-line interface (CLI) software for RF signal capture, analysis, and classification using Software Defined Radio (SDR) hardware and advanced Machine Learning/Deep Learning algorithms.

## 🚀 Features

### 📡 Hardware Support
- **RTL-SDR** (Primary)
- HackRF One (Planned)
- USRP (Planned)
- **Simulation Mode** - Works without hardware

### 🤖 Machine Learning Models
- **CNN** (Convolutional Neural Networks) for signal classification
- **LSTM** for temporal signal analysis  
- **ResNet** inspired architectures
- **Random Forest** & **SVM** for traditional ML
- Real-time signal classification

### 🛠️ Core Capabilities
- Multi-frequency signal capture
- IQ data recording and playback
- Advanced signal preprocessing
- Feature extraction (spectral, statistical)
- Model training and evaluation
- Real-time spectrum analysis

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- RTL-SDR hardware (optional - simulation mode available)

### Quick Install
```bash
# Clone the repository
git clone https://github.com/shahnawazf16/Best-Rf-Capture.git
cd Best-Rf-Capture

# Create virtual environment
python -m venv myenv
source myenv/bin/activate  # On Windows: myenv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
Dependencies

The system automatically installs:

    numpy, scipy - Scientific computing

    tensorflow - Deep Learning

    scikit-learn - Machine Learning

    pyrtlsdr - RTL-SDR support

    matplotlib - Visualization

    pyyaml, h5py - Configuration and data storage


🎯 Quick Start
Capture RF Signals# Capture signals at 433MHz for 10 seconds
rfcapture capture --freq 433e6 --duration 10 --output my_signal.npy

# With custom sample rate and gain
rfcapture capture --freq 868e6 --rate 2.4e6 --gain 30 --duration 5 --output capture.iq
Train Machine Learning Models
bash

# Train CNN model
rfcapture train --model cnn --epochs 100

# Train Random Forest model
rfcapture train --model rf

# Train with custom parameters
rfcapture train --model cnn --epochs 50 --batch-size 64

Classify Signals
bash

# Classify captured signals
rfcapture classify --input my_signal.npy --model cnn

# Save results to file
rfcapture classify --input capture.iq --model rf --output results.json

Real-time Analysis
bash

# Monitor frequency band in real-time
rfcapture analyze --freq 433e6 --model cnn --threshold 0.8

📁 Project Structure
text

Best-Rf-Capture/
├── src/
│   ├── cli/                 # Command-line interface
│   │   ├── main.py         # Main CLI application
│   │   └── simple_main.py  # Simplified version
│   ├── core/               # Core functionality
│   │   ├── capture_manager.py
│   │   └── realtime_analyzer.py
│   ├── hardware/           # SDR hardware abstraction
│   │   └── sdr_controller.py
│   ├── ml/                 # Machine learning
│   │   ├── signal_classifier.py
│   │   └── light_classifier.py
│   └── utils/              # Utilities
│       └── config_loader.py
├── models/                 # Trained ML models
├── data/                  # Sample datasets
├── configs/               # Configuration files
├── tests/                 # Test suites
├── rfcapture             # CLI executable
├── setup.py             # Package installation
└── requirements.txt     # Python dependencies

🔧 Usage Examples
Basic Signal Capture
bash

# Capture 15 seconds of data at 915MHz
rfcapture capture --freq 915e6 --duration 15 --output sample.npy

# Capture with high sample rate
rfcapture capture --freq 2.4e9 --rate 10e6 --gain 25 --output wifi_capture.h5

Advanced ML Training
bash

# Train deep learning model with custom parameters
rfcapture train --model cnn --epochs 200 --batch-size 32

# Train lightweight model for fast inference
rfcapture train --model rf --data custom_dataset/

Signal Classification
bash

# Classify with confidence threshold
rfcapture classify --input unknown_signal.npy --model cnn

# Batch classify multiple files
rfcapture classify --input signals/ --model rf --output batch_results.json

🎮 Simulation Mode

The software includes a built-in simulation mode that generates realistic RF signals when no hardware is available:
bash

# Works even without SDR hardware
rfcapture capture --freq 433e6 --duration 5 --output simulated.npy

The simulator creates:

    FSK, ASK, PSK modulated signals

    Noise patterns

    Mixed signal types

    Realistic IQ data

🔬 Advanced Features
Custom Signal Processing
python

from src.ml.signal_classifier import SignalClassifier
from src.core.capture_manager import CaptureManager

# Custom processing pipeline
classifier = SignalClassifier()
cm = CaptureManager()

# Load and preprocess signals
iq_data, sample_rate = cm.load_iq_data('capture.npy')
features = classifier.extract_features(iq_data)

Real-time Monitoring
python

from src.core.realtime_analyzer import RealTimeAnalyzer

def detection_callback(detection):
    print(f"Detected: {detection['class']} at {detection['frequency']}Hz")

analyzer = RealTimeAnalyzer()
analyzer.add_callback(detection_callback)
analyzer.start_analysis(frequency=868e6, model_type='cnn')

📊 Output Formats

The system supports multiple output formats:

    NumPy (.npy) - Fast binary format

    HDF5 (.h5) - Structured data with metadata

    JSON (.json) - Human-readable results

    CSV (.csv) - Spreadsheet compatible

🐛 Troubleshooting
Common Issues

RTL-SDR Hardware Not Found
text

Error: LIBUSB_ERROR_IO - Could not open SDR

✅ Solution: The system automatically switches to simulation mode and generates realistic test signals.

TensorFlow Import Errors
text

Error: type 'List' is not subscriptable

✅ Solution: Use the light classifier with --model rf or --model svm

Memory Issues with Large Captures
text

Error: Memory allocation failed

✅ Solution: Reduce capture duration or use --rate to lower sample rate
Debug Mode
bash

# Enable verbose logging
python -m src.cli.main capture --freq 433e6 --duration 5 --output debug.npy -v

🤝 Contributing

We welcome contributions! Please see our Contributing Guidelines for details.
Development Setup
bash

# Fork and clone the repository
git clone https://github.com/shahnawazf16/Best-Rf-Capture.git
cd Best-Rf-Capture

# Set up development environment
pip install -r requirements-dev.txt
pre-commit install

# Run tests
pytest tests/

Areas for Contribution

    New SDR hardware support

    Additional ML models

    Signal processing algorithms

    Documentation improvements

    Bug fixes and performance optimizations

📝 License

This project is released under the MIT License - completely free and open source. See the LICENSE file for details.

Author: Shahnawaz
Email: shahnawazzai@gmail.com
GitHub: https://github.com/shahnawazf16/Best-Rf-Capture
🙏 Acknowledgments

    RTL-SDR community for hardware support

    TensorFlow team for ML framework

    SciPy and NumPy communities for scientific computing tools

📞 Support

    📧 Email: shahnawazzai@gmail.com

    🐛 Issues: GitHub Issues

    💬 Discussions: GitHub Discussions

🚀 Future Roadmap

    Web interface for remote monitoring

    Docker containerization

    Additional SDR hardware support

    Pre-trained models for common signals

    Cloud deployment options

    Mobile application

1. LICENSE File

Create a LICENSE file in your repository root:
text

MIT License

Copyright (c) 2024 Shahnawaz

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

2. requirements.txt
text

numpy>=1.21.0
scipy>=1.7.0
tensorflow>=2.8.0
matplotlib>=3.5.0
pyyaml>=6.0
pyrtlsdr>=0.3.0
scikit-learn>=1.0.0
h5py>=3.6.0
tqdm>=4.62.0
joblib>=1.2.0

3. .gitignore
gitignore

# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environment
myenv/
env/
venv/

# Models and data
models/
data/
*.npy
*.h5
*.iq

# Logs
*.log

# IDE
.vscode/
.idea/



