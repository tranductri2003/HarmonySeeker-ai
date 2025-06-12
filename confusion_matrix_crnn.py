from SongChordRecognizer_Training.Models import CRNN_basic_WithStandardScaler
from SongChordRecognizer_Training.Datasets import IsophonicsDataset
from joblib import load
import numpy as np
from sklearn.model_selection import train_test_split

# ===== Load preprocessed data =====
print("📥 Loading dataset...")
x, y = IsophonicsDataset.load_preprocessed_dataset(
    "/Users/triductran/SpartanDev/my-work/datn/HarmonySeeker-ai/SongChordRecognizer_Training/PreprocessedDatasets/isophonics_cnn_sr22050_hop512_ws8_nf100_cqtspec_noFlat_nonNormC.ds"
)
print("✅ Loaded data:")
print("  Shape of x:", x.shape)
print("  Shape of y:", y.shape)

# ===== Reshape and split =====
n_samples, n_frames, n_chromas = x.shape
x = x.reshape((n_samples, n_frames, n_chromas, 1))
_, dev_x, _, dev_y = train_test_split(x, y, test_size=0.2, random_state=42)

# ===== Load model and scaler =====
print("🧠 Loading CRNN model and StandardScaler...")
model = CRNN_basic_WithStandardScaler(
    input_shape=(n_frames, n_chromas, 1), output_classes=25
)
model_path = "/Users/triductran/SpartanDev/my-work/datn/HarmonySeeker-ai/SongChordRecognizer_Pipeline/models/original_crnn.keras"
preprocessor_path = "/Users/triductran/SpartanDev/my-work/datn/HarmonySeeker-ai/SongChordRecognizer_Pipeline/models/original_preprocessor.bin"

# Load model weights
model = CRNN_basic_WithStandardScaler()
model.load(model_path, preprocessor_path)

# ===== Preprocess dev set =====
_, n_frames, n_chromas, _ = dev_x.shape
dev_x_scaled = model.preprocessor.transform(dev_x.reshape((-1, n_chromas)))
dev_x_scaled = dev_x_scaled.reshape((-1, n_frames, n_chromas, 1))

# ======= Evaluate model =======
# Predict on dev set
predictions = model.model.predict(dev_x_scaled)  # (batch_size, time_steps, 25)
predicted_classes = np.argmax(predictions, axis=-1)

# Use majority vote
from scipy.stats import mode

majority_preds = mode(predicted_classes, axis=1)[0].squeeze()

# Compute accuracy
from sklearn.metrics import accuracy_score

dev_acc = accuracy_score(dev_y, majority_preds)
print(f"✅ Dev accuracy: {dev_acc * 100:.2f}%")

# ===== Display confusion matrix =====
print("📊 Generating confusion matrix...")
model.model.display_confusion_matrix(
    dev_x_scaled,
    dev_y,
    save_path="/Users/triductran/SpartanDev/my-work/datn/HarmonySeeker-ai/SongChordRecognizer_Pipeline/my_models/confusion_matrix_crnn_noFlat_nonNormC.png",
)
print("✅ Confusion matrix saved.")

"""
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 1000, 252, 16)     160       
_________________________________________________________________
batch_normalization (BatchNo (None, 1000, 252, 16)     64        
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 1000, 252, 16)     2320      
_________________________________________________________________
batch_normalization_1 (Batch (None, 1000, 252, 16)     64        
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 1000, 252, 16)     2320      
_________________________________________________________________
batch_normalization_2 (Batch (None, 1000, 252, 16)     64        
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 1000, 84, 16)      0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 1000, 84, 32)      4640      
_________________________________________________________________
batch_normalization_3 (Batch (None, 1000, 84, 32)      128       
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 1000, 84, 32)      9248      
_________________________________________________________________
batch_normalization_4 (Batch (None, 1000, 84, 32)      128       
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 1000, 84, 32)      9248      
_________________________________________________________________
batch_normalization_5 (Batch (None, 1000, 84, 32)      128       
_________________________________________________________________
reshape (Reshape)            (None, 1000, 2688)        0         
_________________________________________________________________
bidirectional (Bidirectional (None, 1000, 256)         2164224   
_________________________________________________________________
bidirectional_1 (Bidirection (None, 1000, 32)          26304     
_________________________________________________________________
dense (Dense)                (None, 1000, 25)          825       
=================================================================
Total params: 2,219,865
Trainable params: 2,219,577
Non-trainable params: 288

Dev accuracy:  68.12 % 
"""
