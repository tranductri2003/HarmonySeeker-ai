import os
import tempfile
import soundfile as sf
import io
import warnings

import numpy as np
import librosa

from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi import HTTPException

# Suppress warnings
warnings.filterwarnings("ignore")

# Disable TensorFlow logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = (
    "3"  # 0 = all, 1 = filter INFO, 2 = filter WARNING, 3 = filter ERROR
)
import tensorflow as tf

tf.get_logger().setLevel("ERROR")

from SongChordRecognizer_Pipeline.DataPreprocessor import DataPreprocessor
from SongChordRecognizer_Training.Models import CRNN_basic_WithStandardScaler
from SongChordRecognizer_Training.Spectrograms import cqt_spectrogram
from SongChordRecognizer_Pipeline.KeyRecognizer import KeyRecognizer
from VoiceSeparator_Pipeline.inferrence import separate_audio

# Import Essentia for scale detection
from essentia.standard import KeyExtractor

# Load environment variables
load_dotenv()
CRNN_MODEL_PATH = os.getenv("CRNN_MODEL_PATH")
CRNN_SCALER_PATH = os.getenv("CRNN_SCALER_PATH")
VOICE_MODEL_PATH = os.getenv("VOICE_MODEL_PATH")

# Constants
SAMPLE_RATE = int(os.getenv("SAMPLE_RATE"))
HOP_LENGTH = int(os.getenv("HOP_LENGTH"))
N_FRAMES = int(os.getenv("N_FRAMES"))

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


def estimate_scale_with_essentia(y, sr):
    """
    Estimate the scale (major/minor) using Essentia KeyExtractor.
    Only the scale is returned and used to assist CRNN prediction.
    """
    if sr != 44100:
        y = librosa.resample(y, orig_sr=sr, target_sr=44100)
        sr = 44100
    y = y.astype(np.float32)

    key_extractor = KeyExtractor()
    key, scale, _ = key_extractor(y)

    return scale


@app.get("/health")
async def health():
    return JSONResponse(content={"status": "ok"})


@app.post("/ai/predict-chord")
async def predict_chord(file: UploadFile = File(...)):
    """
    Predict the main chord, full chord sequence, and key from an uploaded audio file.
    Uses Essentia to first detect the scale (major/minor) for better accuracy.
    """
    tmp_path = None
    try:
        # Save the uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        # Load audio
        y, sr = librosa.load(tmp_path, sr=SAMPLE_RATE)

        # 1. Get scale from Essentia
        scale = estimate_scale_with_essentia(y, sr)

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

        # Predict chords
        predictions = model.predict(x)
        chord_indices = predictions.argmax(axis=2).flatten()

        # Main chord (most frequent)
        main_chord_index = np.bincount(chord_indices).argmax()
        main_chord = DataPreprocessor.chord_indices_to_notations([main_chord_index])[0]
        chord_sequence = DataPreprocessor.chord_indices_to_notations(chord_indices)

        # Key detection (music key, not just most frequent chord)
        chords, counts = np.unique(chord_indices, return_counts=True)
        chord_counts = dict(zip(chords, counts))
        key = KeyRecognizer.estimate_key(
            chord_counts, use_relative_mode=True, target_scale=scale
        )

        return JSONResponse(
            content={
                "key": key,
                "main_chord": main_chord,
                "chord_sequence": chord_sequence,
                "scale": scale,
            }
        )

    except Exception as e:
        return JSONResponse(
            content={"error": f"Internal error: {str(e)}"}, status_code=500
        )
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.post("/ai/separate-voice")
async def voice_removal(file: UploadFile = File(...)):
    """
    Process voice separation on an audio file.
    """
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")

        # Process the audio
        file_bytes = await file.read()
        audio, sr = sf.read(io.BytesIO(file_bytes))
        result = separate_audio(audio, sr, model_path=VOICE_MODEL_PATH)

        # Check if separation was successful
        if result is None:
            raise HTTPException(status_code=500, detail="Voice separation failed")

        return StreamingResponse(
            content=result,
            media_type="application/zip",
            headers={"Content-Disposition": "attachment; filename=separated_audio.zip"},
        )

    except Exception as e:
        return JSONResponse(
            content={"error": f"Internal error: {str(e)}"}, status_code=500
        )
