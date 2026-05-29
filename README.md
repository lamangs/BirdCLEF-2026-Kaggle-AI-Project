# BirdCLEF 2026 Audio Classification

This project builds a bird sound classification pipeline using PyTorch and mel spectrograms from the BirdCLEF 2026 dataset.

## Project Overview

The goal of this project is to classify bird species from audio recordings using deep learning techniques. Audio recordings are converted into mel spectrograms, which are then used as input features for a Convolutional Neural Network (CNN).

The project includes:

* Audio visualization
* Dataset preprocessing
* CNN model training
* Validation and testing
* Prediction generation
* Kaggle submission pipeline

## Technologies Used

* Python
* PyTorch
* Librosa
* NumPy
* Pandas
* Matplotlib
* Kaggle Notebook Environment

## Dataset

Dataset used:

* BirdCLEF 2026 competition dataset from Kaggle

Main files:

* `train.csv`
* `train_audio/`
* `sample_submission.csv`

## Audio Processing

Audio files are loaded using Librosa with a sampling rate of 32000 Hz.

Each audio recording is converted into a mel spectrogram using:

* 128 mel bands
* FFT size of 1024

The spectrograms are converted to decibel scale and resized/padded before being used for training.

## Model Architecture

The project uses a small Convolutional Neural Network (CNN) built with PyTorch.

Architecture:

* Conv2D
* ReLU activation
* MaxPooling
* Fully connected layers
* BCEWithLogitsLoss for multi-label classification

## Files

* `visualize_audio.py`
  Visualizes mel spectrograms from bird recordings.

* `dataset_test.py`
  Tests dataset loading and tensor generation.

* `train_model.py`
  Trains the CNN model and saves model weights.

* `train_validate.py`
  Evaluates validation performance during training.

* `predict_one.py`
  Runs prediction on a single audio sample.

* `make_submission.py`
  Generates the Kaggle submission CSV file.

## Results

The notebook successfully:

* Trained a CNN model on BirdCLEF audio data
* Generated valid predictions
* Created a valid Kaggle submission file
* Produced a public leaderboard score on Kaggle

## Future Improvements

Possible future improvements include:

* Training on larger datasets
* Using pretrained audio models
* Adding audio augmentation
* Hyperparameter tuning
* Using GPU acceleration
* Improving inference on long soundscape recordings

## Author

Leman Gasimova
