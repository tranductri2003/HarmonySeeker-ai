from SongChordRecognizer_Training.Models import CNN
from SongChordRecognizer_Training.Datasets import IsophonicsDataset
from joblib import load
import numpy as np

# Load dev set
x, y = IsophonicsDataset.load_preprocessed_dataset(
    "/Users/triductran/SpartanDev/my-work/datn/HarmonySeeker-ai/SongChordRecognizer_Training/PreprocessedDatasets/isophonics_cnn_sr22050_hop512_ws8_nf100_cqtspec_noFlat_nonNormC.ds"
)
n_samples, n_frames, n_chromas = x.shape

# Reshape
x = x.reshape((x.shape[0], x.shape[1], x.shape[2], 1))

# Split dev set (same as train script)
from sklearn.model_selection import train_test_split

_, dev_x, _, dev_y = train_test_split(x, y, test_size=0.2, random_state=42)

# Load normalization params
mean, std = load(
    "/Users/triductran/SpartanDev/my-work/datn/HarmonySeeker-ai/SongChordRecognizer_Pipeline/my_models/cnn_input_mean_std.bin"
)

# Normalize
dev_x = (dev_x - mean) / std

# Load model
input_shape = (n_frames, n_chromas, 1)
model = CNN(input_shape=input_shape)
model_path = "/Users/triductran/SpartanDev/my-work/datn/HarmonySeeker-ai/SongChordRecognizer_Pipeline/my_models/cnn_sr22050_hop512_ws8_nf100_cqtspec_noFlat_nonNormC.keras"
model.load(model_path)

# ======= Evaluate model =======
print("\n📊 Evaluating model on dev set...")
dev_acc = model.score(dev_x, dev_y)
print(f"✅ Dev accuracy: {dev_acc * 100:.2f}%")

# Display confusion matrix
model.display_confusion_matrix_cnn(
    dev_x,
    dev_y,
    save_path="/Users/triductran/SpartanDev/my-work/datn/HarmonySeeker-ai/SongChordRecognizer_Pipeline/my_confusion_matrix/cnn_sr22050_hop512_ws8_nf100_cqtspec_noFlat_nonNormC.png",
)

"""
Model: "sequential"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ conv2d (Conv2D)                      │ (None, 17, 252, 32)         │             320 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ max_pooling2d (MaxPooling2D)         │ (None, 9, 126, 32)          │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv2d_1 (Conv2D)                    │ (None, 9, 126, 64)          │          18,496 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ max_pooling2d_1 (MaxPooling2D)       │ (None, 5, 63, 64)           │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv2d_2 (Conv2D)                    │ (None, 5, 63, 64)           │          36,928 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ flatten (Flatten)                    │ (None, 20160)               │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense (Dense)                        │ (None, 64)                  │       1,290,304 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_1 (Dense)                      │ (None, 25)                  │           1,625 │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
 Total params: 1,347,673 (5.14 MB)
 Trainable params: 1,347,673 (5.14 MB)
 Non-trainable params: 0 (0.00 B)

📊 Evaluating model on dev set...
/Users/triductran/SpartanDev/my-work/datn/HarmonySeeker-ai/venv/lib/python3.12/site-packages/keras/src/backend/tensorflow/nn.py:717: UserWarning: "`sparse_categorical_crossentropy` received `from_logits=True`, but the `output` argument was produced by a Softmax activation and thus does not represent logits. Was this intended?
  output, from_logits = _get_logits(
1869/1869 - 14s - 8ms/step - accuracy: 0.5955 - loss: 2.3865
✅ Dev accuracy: 59.55%
1869/1869 ━━━━━━━━━━━━━━━━━━━━ 16s 9ms/step  
✅ Saved confusion matrix to /Users/triductran/SpartanDev/my-work/datn/HarmonySeeker-ai/SongChordRecognizer_Pipeline/my_confusion_matrix/confusion_matrix_cnn_noFlat_nonNormC.png
"""
