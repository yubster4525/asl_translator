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

4. **Download the ASL Alphabet dataset:**
   
   You can download the ASL Alphabet dataset from Kaggle:
   https://www.kaggle.com/datasets/grassknoted/asl-alphabet

   After downloading:
   
   a. Extract the zip file
   b. Place the extracted folders in the `data/` directory with the following structure:
   
   ```
   data/
   └── asl_alphabet/
       ├── A/
       │   ├── image_1.jpg
       │   ├── image_2.jpg
       │   └── ...
       ├── B/
       │   ├── image_1.jpg
       │   └── ...
       └── ...  # Folders for each letter (A-Z), plus 'space' and 'nothing'
   ```

## Training a Model

### Dataset Preparation

The dataset should contain 28 class folders:
- 26 alphabet letters (A-Z)
- 'space' folder for the space gesture
- 'nothing' folder for no gesture

Each folder should contain multiple image samples of hands forming the corresponding sign.

### Training the Keypoint-based Model

This approach first extracts hand landmarks using MediaPipe and trains an MLP classifier:

```bash
# Train directly
python src/models/train_keypoint_model.py --data_dir data/asl_alphabet --epochs 20

# For faster training, precompute keypoints first
python src/data_processing/precompute_keypoints.py --data_dir data/asl_alphabet --output data/precomputed_keypoints.npz
python src/models/train_keypoint_model.py --data_dir data/asl_alphabet --precomputed_keypoints data/precomputed_keypoints.npz --epochs 20
```

**Optional Parameters:**
- `--batch_size`: Set the batch size (default: 32)
- `--learning_rate`: Set the learning rate (default: 0.001)
- `--output_dir`: Specify where to save the model (default: 'models/')

### Training the CNN-based Model

This approach trains a CNN directly on hand images:

```bash
python src/models/train_cnn_model.py --data_dir data/asl_alphabet --epochs 20
```

**Optional Parameters:**
- `--batch_size`: Set the batch size (default: 32)
- `--learning_rate`: Set the learning rate (default: 0.001)
- `--output_dir`: Specify where to save the model (default: 'models/')

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

The web interface provides:

1. **Translation Tab**: Real-time ASL alphabet translation through your webcam
2. **Training Tab**: Interface to train new models with customizable parameters
   - Select model type (CNN or Keypoint)
   - Set training hyperparameters (epochs, batch size, learning rate)
   - Monitor training progress with real-time logs
   - View training history plots after completion

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