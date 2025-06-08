from Datasets import IsophonicsDataset
from Models import CRNN
from sklearn.preprocessing import StandardScaler
import sklearn
import time
import numpy as np

# Load preprocessed dataset
x, y = IsophonicsDataset.load_preprocessed_dataset(
    "/Users/triductran/SpartanDev/my-work/datn/HarmonySeeker-ai/SongChordRecognizer_Training/PreprocessedDatasets/isophonics_crnn_ws500_hop1024_sc10.ds"
)
print("✅ Loaded data:")
print("  Shape of x:", x.shape)  # (samples, frames, features)
print("  Shape of y:", y.shape)

# Get dimensions
n_samples, n_frames, n_chromas = x.shape

# Train/dev split
train_x, dev_x, train_y, dev_y = sklearn.model_selection.train_test_split(
    x, y, test_size=0.2, random_state=42
)

# ======= Preprocessing =======
# Fit StandardScaler on flattened train_x
scaler = StandardScaler()
scaler.fit(train_x.reshape(-1, n_chromas))

# Transform train_x and dev_x using the scaler
train_x = scaler.transform(train_x.reshape(-1, n_chromas)).reshape(
    -1, n_frames, n_chromas, 1
)
dev_x = scaler.transform(dev_x.reshape(-1, n_chromas)).reshape(
    -1, n_frames, n_chromas, 1
)

# ======= Train model =======
model = CRNN(input_shape=(n_frames, n_chromas, 1))

print("\n🚀 Training CRNN...")
start_time = time.time()
model.fit(train_x, train_y, dev_x, dev_y, epochs=50)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"\n⏱️ Training completed in {elapsed_time:.2f} seconds")

# ======= Evaluate model =======
accuracy = model.score(dev_x, dev_y)
print(f"\n📊 Dev accuracy: {accuracy * 100:.2f}%")

try:
    model.display_confusion_matrix(dev_x, dev_y)
except Exception as e:
    print(f"⚠️ Warning: Error displaying confusion matrix: {e}")

# ======= Model summary =======
print("\n📋 Model Summary:")
model.model.summary()

total_params = np.sum([np.prod(v.get_shape()) for v in model.model.trainable_weights])
print(f"\n🧮 Total trainable parameters: {total_params}")

# ======= Save model =======
print("\n💾 Saving model...")
try:
    model.model.save(
        r"/Users/triductran/SpartanDev/my-work/datn/HarmonySeeker-ai/SongChordRecognizer_Pipeline/my_models/crnn.keras"
    )
    print("✅ Model saved successfully!")
except Exception as e:
    print(f"❌ Error saving model: {e}")
