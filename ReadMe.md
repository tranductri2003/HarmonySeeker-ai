# HarmonySeeker AI

HarmonySeeker AI is an intelligent system for musical analysis that helps musicians and music enthusiasts discover harmonic elements in songs.

## Overview

This project focuses on building AI models for music analysis, featuring:

1. **Voice Separation**: Separates vocals and instrumental tracks from songs
2. **Chord Recognition**: Uses CNN and CRNN models to recognize chord sequences from audio files

## Project Structure

- **VoiceSeparator_Pipeline/**: Models and processing pipeline for separating vocals from background music
  - Uses time-frequency transformation and convolutional neural networks
  - Implements custom Keras layers for audio processing
  
- **SongChordRecognizer_Pipeline/**: Pipeline for chord recognition from audio using pre-trained models
  - Includes data preprocessing tools
  - Features key recognition capabilities
  
- **SongChordRecognizer_Training/**: Training processes for chord recognition models
  - Contains dataset preparation and augmentation techniques
  - Includes model architecture definitions and training scripts

- **SongChordRecognizer_Report/**: Analysis and evaluation of model performance
  - Statistical comparison of CNN and CRNN models
  - Visualization of training progress and confusion matrices
  - Performance metrics across different testing scenarios
  
- **tests/**: Test suites for various system functionalities

## AI Models

The project implements several AI architectures:

### Voice Separation Models
- **Time-Frequency Transform Networks**: Custom architecture that processes audio in both time and frequency domains
- **Encoder-Decoder Architecture**: For extracting vocals from mixed audio signals

### Chord Recognition Models
- **CNN (Convolutional Neural Network)**: Effective for extracting features from spectrograms
  - 1.3M parameters with multiple convolutional layers
  - Achieves ~60% accuracy on validation data
  
- **CRNN (Convolutional Recurrent Neural Network)**: Combines CNN's feature extraction capabilities with RNN's sequence learning
  - 2.2M parameters with bidirectional layers
  - Achieves ~68% accuracy on validation data

## Technical Details

### Voice Separation
- Input: Mixed audio (songs with vocals and instruments)
- Output: Separated vocal track and instrumental track
- Uses custom Time-Frequency Transform blocks with Keras
- Implements Signal-to-Distortion Ratio (SDR) metrics for evaluation

### Chord Recognition
- Supports 24 chord classes (12 major and 12 minor chords)
- Uses Constant-Q Transform (CQT) spectrograms for input features
- Multiple evaluation modes:
  - Strict (exact match) vs. Semi-strict (allows semitone tolerance)
  - With/without major-minor distinction

### Performance Analysis
- CRNN model shows significantly better performance than CNN (up to 84% vs 29.6%)
- Voice separation improves chord recognition accuracy in certain contexts
- Best results achieved with semi-strict evaluation and major-minor distinction

## Features

- Separate vocals from instrumental music in songs
- Recognize and analyze chord progressions in music
- Process different audio formats and sample rates
- Support for musical key detection

## Future Development

The project is continuously being improved to enhance accuracy in chord recognition and voice separation quality. Future plans include:

- Integration with music production software
- Real-time chord recognition capabilities
- Expanded harmonic analysis beyond basic chords
- Support for more complex musical structures