# ASL Translator

A real-time American Sign Language (ASL) alphabet translator using computer vision and machine learning.

## Overview

This project implements two approaches for ASL alphabet sign recognition:

1. **Image-based CNN Classifier**: Trains a convolutional neural network directly on images of ASL hand signs
2. **Keypoint-based MLP Classifier**: Uses MediaPipe to extract hand landmarks (keypoints) and trains a multilayer perceptron on these features

Both approaches provide real-time ASL alphabet translation using a webcam.

## Features

- Hand landmark detection using MediaPipe
- Two model architectures:
  - CNN for direct image classification
  - MLP for keypoint-based classification
- Pre-computed keypoints to speed up training
- Real-time translation through webcam feed
- Training with data augmentation
- Model validation and performance visualization

## Project Structure

```
.
├── src/                    # Source code
│   ├── utils/              # Utility functions
│   ├── models/             # Model definitions
│   ├── data_processing/    # Data loading and processing
│   └── web/                # Web interface
├── data/                   # Training data and pre-processed features
├── models/                 # Saved trained models
└── web/                    # Web application code
```

## Requirements

- Python 3.8+
- PyTorch
- OpenCV
- MediaPipe
- NumPy
- Matplotlib
- tqdm

## Getting Started

1. **Clone the repository:**
   ```
   git clone https://github.com/yubster4525/asl_translator.git
   cd asl_translator
   ```

2. **Set up a virtual environment (recommended):**
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```
   pip install -r requirements.txt
   ```

4. **Download the ASL Alphabet dataset** (if not already included)

## Training a Model

### Training the Keypoint-based Model

```
python src/models/train_keypoint_model.py --data_dir data/asl_alphabet --epochs 20
```

### Training the CNN-based Model

```
python src/models/train_cnn_model.py --data_dir data/asl_alphabet --epochs 20
```

## Running the Translator

### Console-based Demo

```
python src/demo.py --model keypoint  # or --model cnn
```

### Web Interface

```
python src/web/app.py
```

Then open your browser to http://localhost:5000

## Future Improvements

- Support for full ASL sentences and phrases
- Improved accuracy for challenging hand positions
- Mobile application deployment
- Integration with text-to-speech for audible output

## License

MIT

## Acknowledgments

- The ASL Alphabet Dataset
- MediaPipe for their hand landmark detection model