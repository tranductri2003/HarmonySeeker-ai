import os
import tempfile

import numpy as np
import librosa
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from SongChordRecognizer_Pipeline.DataPreprocessor import DataPreprocessor
from SongChordRecognizer_Training.Models import CRNN_basic_WithStandardScaler
from SongChordRecognizer_Training.Spectrograms import cqt_spectrogram

# Load environment variables
load_dotenv()
CRNN_MODEL_PATH = os.getenv("CRNN_MODEL_PATH")
CRNN_SCALER_PATH = os.getenv("CRNN_SCALER_PATH")

# Constants
SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", 22050))
HOP_LENGTH = int(os.getenv("HOP_LENGTH", 512))
N_FRAMES = int(os.getenv("N_FRAMES", 1000))

# Initialize FastAPI app
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/predict-chord")
async def predict_chord(file: UploadFile = File(...)):
    """
    Predict the main chord and full chord sequence from an uploaded audio file.
    """
    tmp_path = None
    try:
        # Save the uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        # Load audio
        y, sr = librosa.load(tmp_path, sr=SAMPLE_RATE)

        # Load model and scaler
        model = CRNN_basic_WithStandardScaler()
        model.load(CRNN_MODEL_PATH, CRNN_SCALER_PATH)

        # Preprocess the audio for prediction
        x = DataPreprocessor.sequence_preprocess(
            waveform=y,
            sample_rate=sr,
            hop_length=HOP_LENGTH,
            n_frames=N_FRAMES,
            spectrogram_generator=cqt_spectrogram,
            norm_to_C=False,
        )

        # Predict chords and get the most frequent one
        predictions = model.predict(x)
        chord_indices = predictions.argmax(axis=2).flatten()
        main_chord_index = np.bincount(chord_indices).argmax()
        main_chord = DataPreprocessor.chord_indices_to_notations([main_chord_index])[0]
        chord_sequence = DataPreprocessor.chord_indices_to_notations(chord_indices)

        return JSONResponse(
            content={"main_chord": main_chord, "chord_sequence": chord_sequence}
        )

    except Exception as e:
        return JSONResponse(
            content={"error": f"Internal error: {str(e)}"}, status_code=500
        )
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.post("/voice-removal")
async def voice_removal(file: UploadFile = File(...)):
    """
    Placeholder for future voice removal implementation.
    """
    return JSONResponse(
        content={"message": "Voice removal not implemented yet."}, status_code=501
    )
