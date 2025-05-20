from SongChordRecognizer_Training.Datasets import IsophonicsDataset
from SongChordRecognizer_Training.Models import MLP_scalered
import sklearn
import time
import joblib

# Load preprocessed dataset
x, y = IsophonicsDataset.load_preprocessed_dataset(
    r"/Users/triductran/SpartanDev/my-work/key-finding/SongChordsRecognizer/SongChordRecognizer_Training/PreprocessedDatasets/isophonics_mlp_ws5_hop1024_sc22_logmel_flatT_noNormC.ds"
)
print("✅ Loaded data:")
print("  Shape of x:", x.shape)
print("  Shape of y:", y.shape)

# Split into train/dev sets
train_x, dev_x, train_y, dev_y = sklearn.model_selection.train_test_split(
    x, y, test_size=0.2, random_state=42
)

# Create model
mlp_model = MLP_scalered(max_iter=500, random_state=7)

# Train the model with timer
print("\n🚀 Training MLP...")
start_time = time.time()
mlp_model.fit(train_x, train_y)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"\n⏱️ Training completed in {elapsed_time:.2f} seconds")

# Evaluate model
accuracy = mlp_model.score(dev_x, dev_y)
print(f"\n📊 Dev accuracy: {accuracy * 100:.2f}%")

# Show and save confusion matrix
try:
    mlp_model.display_confusion_matrix(dev_x, dev_y)
except Exception as e:
    print(f"⚠️ Warning: Error displaying confusion matrix: {e}")

# Save model and scaler (pipeline)
print("\n💾 Saving model...")
try:
    joblib.dump(
        mlp_model.model,
        r"/Users/triductran/SpartanDev/my-work/key-finding/SongChordsRecognizer/SongChordRecognizer_Pipeline/my_models/mlp22_isophonics_206songs_noNormC.model",
    )
    print("✅ Model and scaler saved successfully!")
except Exception as e:
    print(f"❌ Error saving model: {e}")
