import os
import numpy as np
import soundfile as sf
from keras import saving, ops, layers
import keras
from dotenv import load_dotenv
from VoiceSeparator_Pipeline.model import *
from VoiceSeparator_Pipeline.metric import *
from VoiceSeparator_Pipeline.audio_process import *
import io
import zipfile

# Load environment variables
load_dotenv()
MODEL_PATH = os.getenv("VOICE_MODEL_PATH")


def separate_audio(audio, sr, model_path=MODEL_PATH):
    print(f"Loading model from: {model_path}")

    # Load the trained model with custom objects
    try:
        # Register custom objects before loading
        custom_objects = {
            "TimeFrequencyTransformBlock": TimeFrequencyTransformBlock,
            "TimeDistributedDenseBlock": TimeDistributedDenseBlock,
            "TimeFrequencyConvolution": TimeFrequencyConvolution,
            "Downscale": Downscale,
            "Upscale": Upscale,
            "spectral_loss": spectral_loss,
            "sdr": sdr,
        }

        model = saving.load_model(model_path, custom_objects=custom_objects)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

    # Load and preprocess audio
    try:
        original_audio = load_audio(audio, sr)
        original_length = len(original_audio)
        print(
            f"Audio loaded: {len(original_audio)} samples ({len(original_audio)/TARGET_SAMPLE_RATE:.2f} seconds)"
        )
    except Exception as e:
        print(f"Error loading audio: {e}")
        return None

    # Preprocess into chunks
    audio_chunks = preprocess_audio(original_audio)
    print(f"Created {len(audio_chunks)} chunks for processing")

    # Predict vocals for each chunk
    predicted_chunks = []
    original_chunks_processed = []

    for i, chunk in enumerate(audio_chunks):
        print(f"Processing chunk {i+1}/{len(audio_chunks)}")

        # Add batch dimension
        chunk_batch = np.expand_dims(chunk, axis=0)

        # Predict vocals
        prediction = model.predict(chunk_batch, verbose=0)

        # Convert prediction to waveform
        vocals_chunk = prediction_to_wave(prediction)
        predicted_chunks.append(vocals_chunk)

        # Store original chunk for music extraction
        original_chunks_processed.append(chunk)

    # Concatenate all predicted chunks
    all_vocals_predictions = np.concatenate(predicted_chunks, axis=0)
    all_original_chunks = np.array(original_chunks_processed)

    # Postprocess to get final vocals
    separated_vocals = postprocess_audio(all_vocals_predictions, original_length)

    # Extract music by subtracting vocals from original
    processed_original = postprocess_audio(all_original_chunks, original_length)

    # Create music track by subtracting vocals from original
    separated_music = processed_original - separated_vocals

    # Normalize both tracks
    def normalize_audio(audio, max_amplitude=0.8):
        if np.max(np.abs(audio)) > 0:
            return audio / np.max(np.abs(audio)) * max_amplitude
        return audio

    separated_vocals = normalize_audio(separated_vocals)
    separated_music = normalize_audio(separated_music)
    processed_original = normalize_audio(processed_original)

    # Write to BytesIO and return buffers (not zip)
    vocals_buffer = io.BytesIO()
    music_buffer = io.BytesIO()
    sf.write(vocals_buffer, separated_vocals, TARGET_SAMPLE_RATE, format="WAV")
    sf.write(music_buffer, separated_music, TARGET_SAMPLE_RATE, format="WAV")
    vocals_buffer.seek(0)
    music_buffer.seek(0)
    return vocals_buffer, music_buffer


def separate_vocals(input_path, output_path, model_path=MODEL_PATH):
    """
    Backward compatibility function - separates only vocals.
    """
    results = separate_audio(
        input_path, vocals_output_path=output_path, model_path=model_path
    )
    return results["vocals"] if results else None
