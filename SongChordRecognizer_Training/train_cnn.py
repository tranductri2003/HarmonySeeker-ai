import time
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import joblib
import os

import tensorflow as tf
from Datasets import IsophonicsDataset
from Models import CNN

# ==== CONFIG PATHS ====
MODEL_SAVE_PATH = "/Users/triductran/SpartanDev/my-work/datn/HarmonySeeker-ai/SongChordRecognizer_Pipeline/my_models/cnn.keras"
SCALER_SAVE_PATH = "/Users/triductran/SpartanDev/my-work/datn/HarmonySeeker-ai/SongChordRecognizer_Pipeline/my_models/cnn_scaler.pkl"
DATASET_PATH = "/Users/triductran/SpartanDev/my-work/datn/HarmonySeeker-ai/SongChordRecognizer_Training/PreprocessedDatasets/isophonics_crnn_ws500_hop1024_sc10.ds"

# ==== LOAD DATA ====
x, y = IsophonicsDataset.load_preprocessed_dataset(DATASET_PATH)
print("✅ Loaded data:")
print("  Shape of x:", x.shape)
print("  Shape of y:", y.shape)

# ==== TRAIN/DEV SPLIT ====
train_x, dev_x, train_y, dev_y = sklearn.model_selection.train_test_split(
    x, y, test_size=0.2, random_state=42
)

# ==== NORMALIZE FEATURES ====
scaler = StandardScaler()
train_x = scaler.fit_transform(train_x)
dev_x = scaler.transform(dev_x)

# ==== CREATE & TRAIN MODEL ====
model = CNN()

print("\n🚀 Training CNN...")
start_time = time.time()
model.fit(train_x, train_y, dev_data=dev_x, dev_targets=dev_y, epochs=10)
end_time = time.time()
print(f"\n⏱️ Training completed in {end_time - start_time:.2f} seconds")

# ==== EVALUATE ====
loss, acc = model.evaluate(dev_x, dev_y)
print(f"\n📊 Dev accuracy: {acc * 100:.2f}%")
print(f"📉 Dev loss: {loss:.4f}")

# ==== CONFUSION MATRIX ====
print("\n📊 Confusion matrix:")
model.display_confusion_matrix(dev_x, dev_y)

# ==== CLASSIFICATION REPORT ====
predictions = model.predict(dev_x)
predicted_labels = np.argmax(predictions, axis=1)
print("\n📄 Classification Report:")
print(classification_report(dev_y, predicted_labels, zero_division=0))

# ==== DISPLAY TRAINING PROGRESS ====
model.display_training_progress()

# ==== SAVE MODEL ====
print("\n💾 Saving model...")
try:
    model.save(MODEL_SAVE_PATH)
    print(f"✅ Model saved at: {MODEL_SAVE_PATH}")
except Exception as e:
    print(f"❌ Error saving model: {e}")

# ==== SAVE SCALER ====
print("\n💾 Saving scaler...")
try:
    joblib.dump(scaler, SCALER_SAVE_PATH)
    print(f"✅ Scaler saved at: {SCALER_SAVE_PATH}")
except Exception as e:
    print(f"❌ Error saving scaler: {e}")
