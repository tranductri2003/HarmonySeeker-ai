from ACR_Training.Datasets import IsophonicsDataset
from ACR_Training.Models import CRNN
import sklearn
import time
import numpy as np
import joblib

# Load preprocessed dataset
x, y = IsophonicsDataset.load_preprocessed_dataset(
    "/Users/triductran/SpartanDev/my-work/key-finding/SongChordsRecognizer/ACR_Training/PreprocessedDatasets/isophonics_crnn_ws500_hop1024_sc10.ds"
)
print("✅ Loaded data:")
print("  Shape of x:", x.shape)
print("  Shape of y:", y.shape)

# Train/dev split
train_x, dev_x, train_y, dev_y = sklearn.model_selection.train_test_split(
    x, y, test_size=0.3, random_state=42
)

# Reshape to 4D (samples, time, features, channel)
train_x = train_x.reshape((-1, train_x.shape[1], train_x.shape[2], 1))
dev_x = dev_x.reshape((-1, dev_x.shape[1], dev_x.shape[2], 1))

# Create model
model = CRNN()

# Training with timer
print("\n🚀 Training CRNN...")
start_time = time.time()
model.fit(train_x, train_y, dev_x, dev_y, epochs=50)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"\n⏱️ Training completed in {elapsed_time:.2f} seconds")

# Evaluation
accuracy = model.score(dev_x, dev_y)
print(f"\n📊 Dev accuracy: {accuracy * 100:.2f}%")

# Show confusion matrix
try:
    model.display_confusion_matrix(dev_x, dev_y)
except Exception as e:
    print(f"⚠️ Warning: Error displaying confusion matrix: {e}")

# Model summary
print("\n📋 Model Summary:")
model.model.summary()

# Count trainable parameters
total_params = np.sum([np.prod(v.get_shape()) for v in model.model.trainable_weights])
print(f"\n🧮 Total trainable parameters: {total_params}")

# Save model
print("\n💾 Saving model...")
try:
    model.model.save(
        r"/Users/triductran/SpartanDev/my-work/key-finding/SongChordsRecognizer/ACR_Pipeline/my_models/crnn10_isophonics_206songs.model"
    )
    print("✅ Model saved successfully!")
except Exception as e:
    print(f"❌ Error saving model: {e}")
