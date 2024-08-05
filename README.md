# Sound Classification Project

## Overview

This project implements a sound classification system using machine learning techniques. It processes audio files, extracts features, and uses a Random Forest classifier to categorize sounds into predefined classes.

## Features

- Audio file loading from a specified directory structure
- Audio preprocessing:
  - Volume normalization
  - Low-pass filtering
  - Sequence length standardization
- Feature extraction:
  - Mel-frequency cepstral coefficients (MFCCs)
  - Chroma
  - Mel spectrogram
  - Spectral contrast
  - Tonnetz
- Random Forest classification
- Performance evaluation

## Requirements

- Python 3.x
- Libraries:
  - librosa
  - numpy
  - scipy
  - scikit-learn

## Project Structure

The main script contains several functions:

1. `load_audio_files`: Loads audio files from a specified directory structure.
2. `normalize_volume`: Normalizes the volume of audio sequences.
3. `lowpass_filter`: Applies a low-pass filter to audio sequences.
4. `pad_truncate_sequence`: Standardizes the length of audio sequences.
5. `preprocess_audio_data`: Applies preprocessing steps to the audio data.
6. `extract_features`: Extracts audio features from preprocessed data.

The main workflow:
1. Load audio files
2. Preprocess the audio data
3. Extract features
4. Split the data into training and testing sets
5. Train a Random Forest classifier
6. Evaluate the classifier's performance

## Usage

1. Ensure all required libraries are installed.
2. Prepare your audio dataset in a folder structure where each subfolder represents a class.
3. Update the `data_folder` variable with the path to your dataset.
4. Run the script to train and evaluate the model.

## Results

The script outputs:
- List of existing classes
- Accuracy of the Random Forest classifier
- Confusion matrix
- Detailed classification report

## Future Improvements

- Implement cross-validation for more robust evaluation
- Experiment with different classifiers or deep learning models
- Add a command-line interface for easier usage
- Implement real-time classification of audio input


