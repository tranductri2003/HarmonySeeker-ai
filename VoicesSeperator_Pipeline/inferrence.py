import os
import numpy as np
import soundfile as sf
from keras import saving, ops, layers
import keras
from VoicesSeperator_Pipeline.model import *
from VoicesSeperator_Pipeline.metric import *
from VoicesSeperator_Pipeline.audio_process import *
import io
import zipfile

MODEL_PATH = "/kaggle/input/model-remover/model_jax (4).keras"

def separate_audio(original_audio, model_path=MODEL_PATH):
    """
    Main function to separate vocals and music from input audio.
    
    Args:
        input_path: Path to input audio file
        vocals_output_path: Path to save separated vocals (optional)
        music_output_path: Path to save separated music/instrumental (optional)
        model_path: Path to trained model
    
    Returns:
        dict: Dictionary containing 'vocals', 'music', and 'original' audio arrays
    """
    print(f"Loading model from: {model_path}")
    
    # Load the trained model with custom objects
    try:
        # Register custom objects before loading
        custom_objects = {
            'TimeFrequencyTransformBlock': TimeFrequencyTransformBlock,
            'TimeDistributedDenseBlock': TimeDistributedDenseBlock,
            'TimeFrequencyConvolution': TimeFrequencyConvolution,
            'Downscale': Downscale,
            'Upscale': Upscale,
            'spectral_loss': spectral_loss,
            'sdr': sdr,
        }
        
        model = saving.load_model(model_path, custom_objects=custom_objects)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
    
    # print(f"Loading audio from: {input_path}")
    
    # Load and preprocess audio
    try:
        #original_audio = load_audio(input_path)
        original_length = len(original_audio)
        print(f"Audio loaded: {len(original_audio)} samples ({len(original_audio)/TARGET_SAMPLE_RATE:.2f} seconds)")
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
    # First, process original audio to match vocals length
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
    
        # Ghi vào BytesIO thay vì file
    vocals_buffer = io.BytesIO()
    music_buffer = io.BytesIO()

    sf.write(vocals_buffer, separated_vocals, TARGET_SAMPLE_RATE, format='WAV')
    sf.write(music_buffer, separated_music, TARGET_SAMPLE_RATE, format='WAV')

    vocals_buffer.seek(0)
    music_buffer.seek(0)

    zip_buffer = create_zip_from_buffers({
        "vocals": vocals_buffer,
        "music": music_buffer
    })

    return zip_buffer


def separate_vocals(input_path, output_path, model_path=MODEL_PATH):
    """
    Backward compatibility function - separates only vocals.
    """
    results = separate_audio(input_path, vocals_output_path=output_path, model_path=model_path)
    return results['vocals'] if results else None

def create_zip_from_buffers(buffers_dict):
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zipf:
        for name, buf in buffers_dict.items():
            buf.seek(0)
            zipf.writestr(f"{name}.wav", buf.read())
    zip_buffer.seek(0)
    return zip_buffer