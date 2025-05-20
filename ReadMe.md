# Song Chords Recognizer    

1) Have you ever heard a **song** on YouTube or the radio
that you would like to **play on guitar and sing** with your friends? 

2) Is there some **song** you like and you want to play it **for yourself** or **improvise** on it **with your music friends**?

3) Are you trying to create **sheet music** for a specific **song** for your **band**?

**CHORDS and HARMONY analysis is a very good start!!** - And that is exactly what this application offers.



## Application Overview

Song Chords Recognizer is a Python-based application for automatic chord recognition from audio. It provides two main models to process audio and return the chord sequence of the song:

 1. MLP (Multi-Layer Perceptron) - A simple but effective neural network model for chord recognition
 2. CRNN (Convolutional Recurrent Neural Network) - A more advanced model combining CNN and RNN layers for better accuracy


## Research

Part of the project is also the [Automatic Chord Recognition task RESEARCH](./SongChordRecognizer_Training/ReadMe.md).
The approach is to use CRNN models with transpose preprocessing and also vote the most frequent chord for each beat duration.
