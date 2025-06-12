import os
import numpy as np
import soundfile as sf
from dotenv import load_dotenv
from VoiceSeparator_Pipeline.model import *
from VoiceSeparator_Pipeline.metric import *
from VoiceSeparator_Pipeline.audio_process import *
from tqdm import tqdm
import shutil

from keras import saving

# Load .env variables
load_dotenv()
MODEL_PATH = os.getenv("VOICE_MODEL_PATH")
TARGET_SR = os.getenv("VOICE_TARGET_SAMPLE_RATE")

# Define input and output directories
INPUT_DIR = "/Users/triductran/SpartanDev/my-work/datn/HarmonySeeker-ai/SongChordRecognizer_Training/Datasets/test"
OUTPUT_DIR = "/Users/triductran/SpartanDev/my-work/datn/HarmonySeeker-ai/SongChordRecognizer_Training/Datasets/test_vocal_only"


# Normalize function (tùy chọn)
def normalize_audio(audio, max_amplitude=0.8):
    if np.max(np.abs(audio)) > 0:
        return audio / np.max(np.abs(audio)) * max_amplitude
    return audio


# Load model with custom objects
def load_separator_model(model_path):
    custom_objects = {
        "TimeFrequencyTransformBlock": TimeFrequencyTransformBlock,
        "TimeDistributedDenseBlock": TimeDistributedDenseBlock,
        "TimeFrequencyConvolution": TimeFrequencyConvolution,
        "Downscale": Downscale,
        "Upscale": Upscale,
        "spectral_loss": spectral_loss,
        "sdr": sdr,
    }
    return saving.load_model(model_path, custom_objects=custom_objects)


# Process a single wav file and save vocal track
def process_single_file(model, file_path, output_path):
    try:
        audio, sr = sf.read(file_path)
        original_audio = load_audio(audio, sr)
        original_length = len(original_audio)

        # Preprocess
        chunks = preprocess_audio(original_audio)
        predicted_chunks = []

        for chunk in chunks:
            chunk_batch = np.expand_dims(chunk, axis=0)
            prediction = model.predict(chunk_batch, verbose=0)
            vocals_chunk = prediction_to_wave(prediction)
            predicted_chunks.append(vocals_chunk)

        all_vocals = np.concatenate(predicted_chunks, axis=0)
        separated_vocals = postprocess_audio(all_vocals, original_length)
        separated_vocals = normalize_audio(separated_vocals)

        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save vocals
        sf.write(output_path, separated_vocals, TARGET_SR)
        return True

    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")
        return False


# Process all files in the input directory and maintain structure in output directory
def process_directory(input_dir, output_dir):
    # Make sure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Load model once
    print("🔄 Loading voice separation model...")
    model = load_separator_model(MODEL_PATH)

    # Find all .wav files in input directory (including subdirectories)
    all_files = []

    print("🔍 Finding audio files in input directory...")
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".wav"):
                input_path = os.path.join(root, file)
                # Create equivalent path in output directory
                rel_path = os.path.relpath(input_path, input_dir)
                output_path = os.path.join(output_dir, rel_path)
                all_files.append((input_path, output_path))

    print(f"📊 Found {len(all_files)} audio files to process")

    # Process all files with progress bar
    successful = 0
    failed = 0

    for input_path, output_path in tqdm(all_files, desc="Separating vocals"):
        if process_single_file(model, input_path, output_path):
            successful += 1
        else:
            failed += 1

    # Print summary
    print("\n✅ Processing complete!")
    print(f"  Successfully processed: {successful} files")
    print(f"  Failed to process: {failed} files")
    print(f"  Vocals saved to: {output_dir}")

    # Check if no files were successfully processed
    if successful == 0:
        print("\n⚠️ Warning: No files were successfully processed!")


# Main function
def main():
    print("🎵 Voice Separation - Creating Vocal-Only Dataset 🎵")
    print(f"  Input directory: {INPUT_DIR}")
    print(f"  Output directory: {OUTPUT_DIR}")

    # Ask for confirmation
    confirmation = input(
        "\n⚠️ This will process all audio files in the input directory. Continue? (y/n): "
    )

    if confirmation.lower() != "y":
        print("Operation cancelled.")
        return

    # Ensure output directory exists
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Process all files
    process_directory(INPUT_DIR, OUTPUT_DIR)


# ---------- USAGE ----------

if __name__ == "__main__":
    main()
