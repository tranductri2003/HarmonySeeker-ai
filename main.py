import os
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
import numpy as np
import librosa
import tempfile
import joblib
from fastapi.middleware.cors import CORSMiddleware

# Load environment variables from .env file
load_dotenv()

# Import your custom modules
from ACR_Pipeline.DataPreprocessor import DataPreprocessor
from ACR_Training.Spectrograms import log_mel_spectrogram, cqt_spectrogram
from ACR_Training.Models import CRNN_basic_WithStandardScaler

# Get model paths from environment variables
MLP_MODEL_PATH = os.getenv("MLP_MODEL_PATH")
CRNN_MODEL_PATH = os.getenv("CRNN_MODEL_PATH")
CRNN_SCALER_PATH = os.getenv("CRNN_SCALER_PATH")

# Load MLP model once at startup
mlp_model = joblib.load(MLP_MODEL_PATH)

# FastAPI app instance
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/predict-chord-mlp")
async def predict_chord_mlp(file: UploadFile = File(...)):
    """
    Predict the main chord and chord sequence of a song using the MLP model.
    """
    try:
        # Save uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        # Load audio file
        y, sr = librosa.load(tmp_path, sr=44100)

        # Preprocess audio for MLP
        x = DataPreprocessor.flatten_preprocess(
            waveform=y,
            sample_rate=sr,
            hop_length=1024,
            window_size=5,
            spectrogram_generator=log_mel_spectrogram,
            norm_to_C=False,
            skip_coef=22,
        )
        if len(x.shape) > 2:
            x = x.reshape(x.shape[0], -1)

        # Predict chords
        predictions = mlp_model.predict(x)
        # Find the most frequent chord index
        main_chord_index = np.bincount(predictions).argmax()
        # Convert index to chord name
        main_chord_name = DataPreprocessor.chord_indices_to_notations(
            [main_chord_index]
        )[0]

        # Get chord sequence
        chord_sequence = DataPreprocessor.chord_indices_to_notations(predictions)

        return JSONResponse(
            content={"main_chord": main_chord_name, "chord_sequence": chord_sequence}
        )

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.post("/predict-chord-crnn")
async def predict_chord_crnn(file: UploadFile = File(...)):
    """
    Predict the main chord and chord sequence of a song using the CRNN model.
    """
    try:
        # Save uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        # Load audio file
        y, sr = librosa.load(tmp_path, sr=22050)

        # Load CRNN model and scaler
        crnn_model = CRNN_basic_WithStandardScaler()
        crnn_model.load(CRNN_MODEL_PATH, CRNN_SCALER_PATH)

        # Preprocess audio for CRNN
        x = DataPreprocessor.sequence_preprocess(
            waveform=y,
            sample_rate=sr,
            hop_length=512,
            n_frames=1000,
            spectrogram_generator=cqt_spectrogram,
            norm_to_C=False,
        )

        # Predict chords (output shape: (samples, frames, classes))
        predictions = crnn_model.predict(x)
        chord_prediction = predictions.argmax(axis=2).flatten()
        # Find the most frequent chord index
        main_chord_index = np.bincount(chord_prediction).argmax()
        # Convert index to chord name
        main_chord_name = DataPreprocessor.chord_indices_to_notations(
            [main_chord_index]
        )[0]

        # Get chord sequence
        chord_sequence = DataPreprocessor.chord_indices_to_notations(chord_prediction)

        print("Main chord: ", main_chord_name)

        return JSONResponse(
            content={"main_chord": main_chord_name, "chord_sequence": chord_sequence}
        )

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
