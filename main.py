import os
import tempfile
import soundfile as sf
import io
import warnings
import boto3
from botocore.config import Config
import time
from datetime import datetime
import json


import numpy as np
import librosa

from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi import HTTPException

# Import our custom logger and health module
from app.logger import setup_logger, log_exception
from app.health import get_system_info

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
S3_BUCKET = os.getenv("S3_BUCKET")
S3_REGION = os.getenv("S3_REGION", "ap-southeast-1")
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")

# Constants
SAMPLE_RATE = int(os.getenv("SAMPLE_RATE"))
HOP_LENGTH = int(os.getenv("HOP_LENGTH"))
N_FRAMES = int(os.getenv("N_FRAMES"))

# Setup logger
logger = setup_logger()

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


# Add middleware for request logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    request_id = f"{int(time.time() * 1000)}-{os.getpid()}"

    # Get client IP and request details
    client_host = request.client.host if request.client else "unknown"
    method = request.method
    url = request.url.path

    # Create a context dictionary to store request information
    context = {
        "request_id": request_id,
        "client": client_host,
        "method": method,
        "path": url,
        "details": {},
    }

    # Process the request
    try:
        response = await call_next(request)

        # Calculate processing time
        process_time = time.time() - start_time

        # Add response information to context
        context["status_code"] = response.status_code
        context["process_time"] = f"{process_time:.3f}s"

        # Log the complete request information in a single line
        logger.info(f"REQUEST: {json.dumps(context)}")

        return response
    except Exception as e:
        # Add error information to context
        context["error"] = str(e)
        context["process_time"] = f"{time.time() - start_time:.3f}s"

        # Log the error in a single line using the custom log_exception function
        log_exception(logger, f"REQUEST_ERROR: {json.dumps(context)}")
        raise


s3_client = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=S3_REGION,
    endpoint_url=f"https://s3.{S3_REGION}.amazonaws.com",
    config=Config(signature_version="s3v4", s3={"addressing_style": "virtual"}),
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
    """
    Health check endpoint that returns system information
    """
    system_info = get_system_info()
    return JSONResponse(content={"status": "ok", "system": system_info})


@app.post("/ai/predict-chord")
async def predict_chord(request: Request, file: UploadFile = File(...)):
    """
    Predict the main chord, full chord sequence, and key from an uploaded audio file.
    Uses Essentia to first detect the scale (major/minor) for better accuracy.
    """
    tmp_path = None
    request_id = (
        request.state.request_id
        if hasattr(request.state, "request_id")
        else f"{int(time.time() * 1000)}-{os.getpid()}"
    )

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
        # Use the custom log_exception function instead of logger.error with exc_info=True
        log_exception(
            logger,
            f"Chord prediction error: {str(e)}",
            filename=file.filename if file and hasattr(file, "filename") else "unknown",
        )
        return JSONResponse(
            content={"error": f"Internal error: {str(e)}"}, status_code=500
        )
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.post("/ai/separate-voice")
async def voice_removal(file: UploadFile = File(...)):
    """
    Process voice separation on an audio file, upload vocals/music to S3, return presigned URLs.
    """
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")

        # Process the audio
        file_bytes = await file.read()
        audio, sr = sf.read(io.BytesIO(file_bytes))

        vocals_buffer, music_buffer = separate_audio(
            audio, sr, model_path=VOICE_MODEL_PATH
        )

        # Set timestamped filenames
        now_str = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        vocals_key = f"vocals_{now_str}.wav"
        music_key = f"music_{now_str}.wav"

        # Upload to S3
        s3_client.upload_fileobj(
            vocals_buffer, S3_BUCKET, vocals_key, ExtraArgs={"ContentType": "audio/wav"}
        )
        s3_client.upload_fileobj(
            music_buffer, S3_BUCKET, music_key, ExtraArgs={"ContentType": "audio/wav"}
        )

        # Generate presigned URLs
        expiration = 3600  # 1 hour
        vocals_url = s3_client.generate_presigned_url(
            "get_object",
            Params={"Bucket": S3_BUCKET, "Key": vocals_key},
            ExpiresIn=expiration,
        )
        music_url = s3_client.generate_presigned_url(
            "get_object",
            Params={"Bucket": S3_BUCKET, "Key": music_key},
            ExpiresIn=expiration,
        )

        return JSONResponse(
            content={
                "vocals_url": vocals_url,
                "music_url": music_url,
                "expiration": expiration,
            }
        )

    except Exception as e:
        # Use the custom log_exception function instead of logger.error with exc_info=True
        log_exception(
            logger,
            f"Voice separation error: {str(e)}",
            filename=file.filename if file and hasattr(file, "filename") else "unknown",
        )
        return JSONResponse(
            content={"error": f"Internal error: {str(e)}"}, status_code=500
        )


@app.get("/error")
async def test_error():
    """
    Test endpoint to generate an error for testing logging
    """
    try:
        # Simulate an error
        result = 1 / 0
    except Exception as e:
        # Use the custom log_exception function instead of logger.error with exc_info=True
        log_exception(logger, f"Test error generated: {str(e)}")
        return JSONResponse(content={"error": "This is a test error"}, status_code=500)
