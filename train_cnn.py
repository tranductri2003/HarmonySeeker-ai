from SongChordRecognizer_Training.Datasets import IsophonicsDataset
from SongChordRecognizer_Training.Models import CNN
from sklearn.model_selection import train_test_split
import numpy as np
import time
import tensorflow as tf
from tqdm.keras import TqdmCallback
from joblib import dump

# ======= Load preprocessed data =======
print("📥 Loading dataset...")
x, y = IsophonicsDataset.load_preprocessed_dataset(
    "/Users/triductran/SpartanDev/my-work/datn/HarmonySeeker-ai/SongChordRecognizer_Training/PreprocessedDatasets/isophonics_cnn_sr22050_hop512_ws8_nf100_cqtspec_noFlat_nonNormC.ds"
)
print("✅ Loaded data:")
print("  Shape of x:", x.shape)
print("  Shape of y:", y.shape)

# ======= Prepare data =======
n_samples, n_frames, n_chromas = x.shape
x = x.reshape((n_samples, n_frames, n_chromas, 1))

# Train-dev split
train_x, dev_x, train_y, dev_y = train_test_split(x, y, test_size=0.2, random_state=42)

# ======= Normalize manually =======
print("📏 Normalizing input to 0-mean, unit-std (StandardScaler logic)...")
mean = train_x.mean()
std = train_x.std()
train_x = (train_x - mean) / std
dev_x = (dev_x - mean) / std
dump((mean, std), "cnn_input_mean_std.bin")

# ======= Create model =======
print("🧠 Initializing CNN model...")
input_shape = (n_frames, n_chromas, 1)
model = CNN(input_shape=input_shape)

# ======= Train model =======
print("\n🚀 Training model...")
start = time.time()
model.fit(train_x, train_y, dev_x, dev_y, epochs=20, batch_size=512)
end = time.time()
print(f"⏱️ Training completed in {end - start:.2f} seconds")

# ======= Evaluate model =======
print("\n📊 Evaluating model on dev set...")
dev_acc = model.score(dev_x, dev_y)
print(f"✅ Dev accuracy: {dev_acc * 100:.2f}%")

# ======= Confusion matrix =======
try:
    model.display_confusion_matrix_cnn(
        dev_x,
        dev_y,
        save_path="/Users/triductran/SpartanDev/my-work/datn/HarmonySeeker-ai/SongChordRecognizer_Pipeline/my_confusion_matrix/cnn_sr22050_hop512_ws8_nf100_cqtspec_noFlat_nonNormC.png",
    )
except Exception as e:
    print(f"⚠️ Error displaying confusion matrix: {e}")

# ======= Save model =======
print("\n💾 Saving model...")
model.save(
    model_path="/Users/triductran/SpartanDev/my-work/datn/HarmonySeeker-ai/SongChordRecognizer_Pipeline/my_models/cnn_sr22050_hop512_ws8_nf100_cqtspec_noFlat_nonNormC.keras"
)


"""
NormC
📥 Loading dataset...
[INFO] The Preprocessed Isophonics Dataset was loaded successfully.
✅ Loaded data:
  Shape of x: (299007, 17, 252)
  Shape of y: (299007,)
📏 Normalizing input to 0-mean, unit-std (StandardScaler logic)...
🧠 Initializing CNN model...
/Users/triductran/SpartanDev/my-work/datn/HarmonySeeker-ai/venv/lib/python3.12/site-packages/keras/src/layers/convolutional/base_conv.py:113: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.
  super().__init__(activity_regularizer=activity_regularizer, **kwargs)
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

🚀 Training model...
  0%|                                                                                      | 0/20 [00:00<?, ?epoch/sEpoch 1/20                                                                              | 0.00/468 [00:00<?, ?batch/s]
/Users/triductran/SpartanDev/my-work/datn/HarmonySeeker-ai/venv/lib/python3.12/site-packages/keras/src/backend/tensorflow/nn.py:717: UserWarning: "`sparse_categorical_crossentropy` received `from_logits=True`, but the `output` argument was produced by a Softmax activation and thus does not represent logits. Was this intended?
  output, from_logits = _get_logits(
468/468 ━━━━━━━━━━━━━━━━━━━━ 217s 462ms/step - accuracy: 0.3208 - loss: 2.1639 - val_accuracy: 0.4331 - val_loss: 1.7300%|█████████████████████████████████████████████████| 468/468 [03:21<00:00, 3.14batch/s, accuracy=0.365, loss=1.98]
100%|█████████████████████████████████████████████████| 468/468 [03:36<00:00, 2.16batch/s, accuracy=0.365, loss=1.98]
                                                                                                                    Epoch 2/20                                                                              | 0.00/468 [00:00<?, ?batch/s]
468/468 ━━━━━━━━━━━━━━━━━━━━ 213s 454ms/step - accuracy: 0.4664 - loss: 1.6301 - val_accuracy: 0.4948 - val_loss: 1.5171%|████████████████████████████████████████████████▉| 467/468 [03:18<00:00, 2.23batch/s, accuracy=0.482, loss=1.58]
100%|█████████████████████████████████████████████████| 468/468 [03:32<00:00, 2.20batch/s, accuracy=0.482, loss=1.58]
                                                                                                                    Epoch 3/20                                                                              | 0.00/468 [00:00<?, ?batch/s]
468/468 ━━━━━━━━━━━━━━━━━━━━ 219s 468ms/step - accuracy: 0.5535 - loss: 1.3353 - val_accuracy: 0.5422 - val_loss: 1.3723%|█████████████████████████████████████████████████▉| 467/468 [03:24<00:00, 2.23batch/s, accuracy=0.56, loss=1.32]
100%|██████████████████████████████████████████████████| 468/468 [03:39<00:00, 2.14batch/s, accuracy=0.56, loss=1.31]
                                                                                                                    Epoch 4/20                                                                              | 0.00/468 [00:00<?, ?batch/s]
468/468 ━━━━━━━━━━━━━━━━━━━━ 216s 462ms/step - accuracy: 0.6205 - loss: 1.1238 - val_accuracy: 0.5616 - val_loss: 1.3173%|█████████████████████████████████████████████████▉| 467/468 [03:21<00:00, 2.17batch/s, accuracy=0.62, loss=1.13]
100%|██████████████████████████████████████████████████| 468/468 [03:36<00:00, 2.17batch/s, accuracy=0.62, loss=1.13]
                                                                                                                    Epoch 5/20                                                                              | 0.00/468 [00:00<?, ?batch/s]
468/468 ━━━━━━━━━━━━━━━━━━━━ 215s 459ms/step - accuracy: 0.6704 - loss: 0.9710 - val_accuracy: 0.5834 - val_loss: 1.2591%|███████████████████████████████████████████████▉| 467/468 [03:20<00:00, 2.43batch/s, accuracy=0.665, loss=0.982]
100%|████████████████████████████████████████████████| 468/468 [03:34<00:00, 2.18batch/s, accuracy=0.665, loss=0.982]
                                                                                                                    Epoch 6/20                                                                              | 0.00/468 [00:00<?, ?batch/s]
468/468 ━━━━━━━━━━━━━━━━━━━━ 210s 449ms/step - accuracy: 0.7082 - loss: 0.8479 - val_accuracy: 0.5845 - val_loss: 1.2670%|████████████████████████████████████████████████| 468/468 [03:16<00:00, 3.15batch/s, accuracy=0.701, loss=0.868]
100%|████████████████████████████████████████████████| 468/468 [03:30<00:00, 2.23batch/s, accuracy=0.701, loss=0.868]
                                                                                                                    Epoch 7/20                                                                              | 0.00/468 [00:00<?, ?batch/s]
468/468 ━━━━━━━━━━━━━━━━━━━━ 207s 443ms/step - accuracy: 0.7434 - loss: 0.7445 - val_accuracy: 0.5928 - val_loss: 1.2573%|███████████████████████████████████████████████▉| 467/468 [03:13<00:00, 2.41batch/s, accuracy=0.734, loss=0.768]
100%|████████████████████████████████████████████████| 468/468 [03:27<00:00, 2.26batch/s, accuracy=0.734, loss=0.768]
                                                                                                                    Epoch 8/20                                                                              | 0.00/468 [00:00<?, ?batch/s]
468/468 ━━━━━━━━━━━━━━━━━━━━ 207s 442ms/step - accuracy: 0.7734 - loss: 0.6554 - val_accuracy: 0.5986 - val_loss: 1.3186%|███████████████████████████████████████████████▉| 467/468 [03:12<00:00, 2.41batch/s, accuracy=0.762, loss=0.685]
100%|████████████████████████████████████████████████| 468/468 [03:26<00:00, 2.26batch/s, accuracy=0.762, loss=0.685]
                                                                                                                    Epoch 9/20                                                                              | 0.00/468 [00:00<?, ?batch/s]
468/468 ━━━━━━━━━━━━━━━━━━━━ 207s 442ms/step - accuracy: 0.7974 - loss: 0.5810 - val_accuracy: 0.5942 - val_loss: 1.3430%|███████████████████████████████████████████████▉| 467/468 [03:13<00:00, 2.41batch/s, accuracy=0.786, loss=0.612]
100%|████████████████████████████████████████████████| 468/468 [03:27<00:00, 2.26batch/s, accuracy=0.786, loss=0.612]
                                                                                                                    Epoch 10/20                                                                             | 0.00/468 [00:00<?, ?batch/s]
468/468 ━━━━━━━━━━━━━━━━━━━━ 206s 441ms/step - accuracy: 0.8187 - loss: 0.5198 - val_accuracy: 0.5984 - val_loss: 1.4178%|████████████████████████████████████████████████▉| 467/468 [03:12<00:00, 2.40batch/s, accuracy=0.807, loss=0.55]
100%|█████████████████████████████████████████████████| 468/468 [03:26<00:00, 2.27batch/s, accuracy=0.807, loss=0.55]
                                                                                                                    Epoch 11/20                                                                             | 0.00/468 [00:00<?, ?batch/s]
468/468 ━━━━━━━━━━━━━━━━━━━━ 206s 441ms/step - accuracy: 0.8383 - loss: 0.4641 - val_accuracy: 0.5917 - val_loss: 1.5367%|███████████████████████████████████████████████▉| 467/468 [03:12<00:00, 2.36batch/s, accuracy=0.826, loss=0.496]
100%|████████████████████████████████████████████████| 468/468 [03:26<00:00, 2.27batch/s, accuracy=0.826, loss=0.496]
                                                                                                                    Epoch 12/20                                                                             | 0.00/468 [00:00<?, ?batch/s]
468/468 ━━━━━━━━━━━━━━━━━━━━ 206s 441ms/step - accuracy: 0.8536 - loss: 0.4205 - val_accuracy: 0.5941 - val_loss: 1.5454%|███████████████████████████████████████████████▉| 467/468 [03:12<00:00, 2.45batch/s, accuracy=0.842, loss=0.448]
100%|████████████████████████████████████████████████| 468/468 [03:26<00:00, 2.27batch/s, accuracy=0.842, loss=0.448]
                                                                                                                    Epoch 13/20                                                                             | 0.00/468 [00:00<?, ?batch/s]
468/468 ━━━━━━━━━━━━━━━━━━━━ 218s 465ms/step - accuracy: 0.8677 - loss: 0.3782 - val_accuracy: 0.5956 - val_loss: 1.6627%|████████████████████████████████████████████████| 468/468 [03:23<00:00, 3.04batch/s, accuracy=0.855, loss=0.407]
100%|████████████████████████████████████████████████| 468/468 [03:37<00:00, 2.15batch/s, accuracy=0.855, loss=0.407]
                                                                                                                    Epoch 14/20                                                                             | 0.00/468 [00:00<?, ?batch/s]
468/468 ━━━━━━━━━━━━━━━━━━━━ 217s 463ms/step - accuracy: 0.8812 - loss: 0.3401 - val_accuracy: 0.5962 - val_loss: 1.7251%|███████████████████████████████████████████████▉| 467/468 [03:22<00:00, 2.35batch/s, accuracy=0.869, loss=0.369]
100%|████████████████████████████████████████████████| 468/468 [03:36<00:00, 2.16batch/s, accuracy=0.869, loss=0.369]
                                                                                                                    Epoch 15/20                                                                             | 0.00/468 [00:00<?, ?batch/s]
468/468 ━━━━━━━━━━━━━━━━━━━━ 216s 463ms/step - accuracy: 0.8914 - loss: 0.3105 - val_accuracy: 0.5951 - val_loss: 1.8229%|███████████████████████████████████████████████▉| 467/468 [03:22<00:00, 2.28batch/s, accuracy=0.881, loss=0.336]
100%|████████████████████████████████████████████████| 468/468 [03:36<00:00, 2.16batch/s, accuracy=0.881, loss=0.336]
                                                                                                                    Epoch 16/20                                                                             | 0.00/468 [00:00<?, ?batch/s]
468/468 ━━━━━━━━━━━━━━━━━━━━ 221s 472ms/step - accuracy: 0.8995 - loss: 0.2875 - val_accuracy: 0.5909 - val_loss: 1.9840%|█████████████████████████████████████████████████| 468/468 [03:26<00:00, 2.98batch/s, accuracy=0.89, loss=0.311]
100%|█████████████████████████████████████████████████| 468/468 [03:40<00:00, 2.12batch/s, accuracy=0.89, loss=0.311]
                                                                                                                    Epoch 17/20                                                                             | 0.00/468 [00:00<?, ?batch/s]
468/468 ━━━━━━━━━━━━━━━━━━━━ 221s 473ms/step - accuracy: 0.9078 - loss: 0.2611 - val_accuracy: 0.5925 - val_loss: 2.0543%|████████████████████████████████████████████████| 468/468 [03:26<00:00, 2.87batch/s, accuracy=0.897, loss=0.288]
100%|████████████████████████████████████████████████| 468/468 [03:41<00:00, 2.11batch/s, accuracy=0.897, loss=0.288]
                                                                                                                    Epoch 18/20                                                                             | 0.00/468 [00:00<?, ?batch/s]
468/468 ━━━━━━━━━━━━━━━━━━━━ 223s 476ms/step - accuracy: 0.9187 - loss: 0.2339 - val_accuracy: 0.5947 - val_loss: 2.1669%|███████████████████████████████████████████████▉| 467/468 [03:28<00:00, 2.39batch/s, accuracy=0.908, loss=0.259]
100%|████████████████████████████████████████████████| 468/468 [03:42<00:00, 2.10batch/s, accuracy=0.908, loss=0.259]
                                                                                                                    Epoch 19/20                                                                             | 0.00/468 [00:00<?, ?batch/s]
468/468 ━━━━━━━━━━━━━━━━━━━━ 216s 462ms/step - accuracy: 0.9210 - loss: 0.2241 - val_accuracy: 0.5964 - val_loss: 2.2553%|███████████████████████████████████████████████▉| 467/468 [03:21<00:00, 2.39batch/s, accuracy=0.912, loss=0.247]
100%|████████████████████████████████████████████████| 468/468 [03:36<00:00, 2.17batch/s, accuracy=0.912, loss=0.247]
                                                                                                                    Epoch 20/20                                                                             | 0.00/468 [00:00<?, ?batch/s]
468/468 ━━━━━━━━━━━━━━━━━━━━ 219s 467ms/step - accuracy: 0.9292 - loss: 0.2032 - val_accuracy: 0.5955 - val_loss: 2.3865%|█████████████████████████████████████████████████| 468/468 [03:23<00:00, 2.94batch/s, accuracy=0.92, loss=0.226]
100%|█████████████████████████████████████████████████| 468/468 [03:38<00:00, 2.14batch/s, accuracy=0.92, loss=0.226]
100%|████████████| 20/20 [1:11:20<00:00, 214.00s/epoch, accuracy=0.92, loss=0.226, val_accuracy=0.595, val_loss=2.39]
[INFO] The CNN model was successfully trained.
⏱️ Training completed in 4284.75 seconds

📊 Evaluating model on dev set...
1869/1869 - 18s - 10ms/step - accuracy: 0.5955 - loss: 2.3865
✅ Dev accuracy: 59.55%
1869/1869 ━━━━━━━━━━━━━━━━━━━━ 17s 9ms/step  
⚠️ Error displaying confusion matrix: not enough values to unpack (expected 3, got 2)

💾 Saving model...
[INFO] The CNN model was saved successfully
"""


"""
nonNormC
📥 Loading dataset...
[INFO] The Preprocessed Isophonics Dataset was loaded successfully.
✅ Loaded data:
  Shape of x: (299007, 17, 252)
  Shape of y: (299007,)
📏 Normalizing input to 0-mean, unit-std (StandardScaler logic)...
🧠 Initializing CNN model...
/Users/triductran/SpartanDev/my-work/datn/HarmonySeeker-ai/venv/lib/python3.12/site-packages/keras/src/layers/convolutional/base_conv.py:113: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.
  super().__init__(activity_regularizer=activity_regularizer, **kwargs)
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

🚀 Training model...
  0%|                                                                                                          | 0/20 [00:00<?, ?epoch/sEpoch 1/20                                                                                                  | 0.00/468 [00:00<?, ?batch/s]
/Users/triductran/SpartanDev/my-work/datn/HarmonySeeker-ai/venv/lib/python3.12/site-packages/keras/src/backend/tensorflow/nn.py:717: UserWarning: "`sparse_categorical_crossentropy` received `from_logits=True`, but the `output` argument was produced by a Softmax activation and thus does not represent logits. Was this intended?
  output, from_logits = _get_logits(
468/468 ━━━━━━━━━━━━━━━━━━━━ 206s 439ms/step - accuracy: 0.2731 - loss: 2.3699 - val_accuracy: 0.4461 - val_loss: 1.75886, val_loss=1.76]
100%|██████████████████████████████████████████████████████████████████████| 468/468 [03:25<00:00, 2.27batch/s, accuracy=0.345, loss=2.1]
                                                                                                                                        Epoch 2/20                                                                                                  | 0.00/468 [00:00<?, ?batch/s]
468/468 ━━━━━━━━━━━━━━━━━━━━ 205s 438ms/step - accuracy: 0.4954 - loss: 1.5909 - val_accuracy: 0.5340 - val_loss: 1.45774, val_loss=1.46]
100%|█████████████████████████████████████████████████████████████████████| 468/468 [03:24<00:00, 2.28batch/s, accuracy=0.512, loss=1.53]
                                                                                                                                        Epoch 3/20                                                                                                  | 0.00/468 [00:00<?, ?batch/s]
468/468 ━━━━━━━━━━━━━━━━━━━━ 204s 436ms/step - accuracy: 0.5999 - loss: 1.2424 - val_accuracy: 0.5708 - val_loss: 1.33281, val_loss=1.33]
100%|█████████████████████████████████████████████████████████████████████| 468/468 [03:23<00:00, 2.29batch/s, accuracy=0.602, loss=1.23]
                                                                                                                                        Epoch 4/20                                                                                                  | 0.00/468 [00:00<?, ?batch/s]
468/468 ━━━━━━━━━━━━━━━━━━━━ 204s 435ms/step - accuracy: 0.6662 - loss: 1.0236 - val_accuracy: 0.5953 - val_loss: 1.26265, val_loss=1.26]
100%|█████████████████████████████████████████████████████████████████████| 468/468 [03:23<00:00, 2.30batch/s, accuracy=0.664, loss=1.03]
                                                                                                                                        Epoch 5/20                                                                                                  | 0.00/468 [00:00<?, ?batch/s]
468/468 ━━━━━━━━━━━━━━━━━━━━ 204s 436ms/step - accuracy: 0.7183 - loss: 0.8552 - val_accuracy: 0.6053 - val_loss: 1.22605, val_loss=1.23]
100%|█████████████████████████████████████████████████████████████████████| 468/468 [03:23<00:00, 2.30batch/s, accuracy=0.71, loss=0.878]
                                                                                                                                        Epoch 6/20                                                                                                  | 0.00/468 [00:00<?, ?batch/s]
468/468 ━━━━━━━━━━━━━━━━━━━━ 204s 436ms/step - accuracy: 0.7573 - loss: 0.7337 - val_accuracy: 0.6126 - val_loss: 1.24763, val_loss=1.25]
100%|████████████████████████████████████████████████████████████████████| 468/468 [03:23<00:00, 2.29batch/s, accuracy=0.747, loss=0.759]
                                                                                                                                        Epoch 7/20                                                                                                  | 0.00/468 [00:00<?, ?batch/s]
468/468 ━━━━━━━━━━━━━━━━━━━━ 204s 436ms/step - accuracy: 0.7910 - loss: 0.6301 - val_accuracy: 0.6213 - val_loss: 1.25911, val_loss=1.26]
100%|█████████████████████████████████████████████████████████████████████| 468/468 [03:23<00:00, 2.30batch/s, accuracy=0.78, loss=0.659]
                                                                                                                                        Epoch 8/20                                                                                                  | 0.00/468 [00:00<?, ?batch/s]
468/468 ━━━━━━━━━━━━━━━━━━━━ 204s 435ms/step - accuracy: 0.8150 - loss: 0.5471 - val_accuracy: 0.6206 - val_loss: 1.297321, val_loss=1.3]
100%|████████████████████████████████████████████████████████████████████| 468/468 [03:23<00:00, 2.30batch/s, accuracy=0.804, loss=0.578]
                                                                                                                                        Epoch 9/20                                                                                                  | 0.00/468 [00:00<?, ?batch/s]
468/468 ━━━━━━━━━━━━━━━━━━━━ 204s 436ms/step - accuracy: 0.8400 - loss: 0.4735 - val_accuracy: 0.6229 - val_loss: 1.37843, val_loss=1.38]
100%|████████████████████████████████████████████████████████████████████| 468/468 [03:23<00:00, 2.30batch/s, accuracy=0.827, loss=0.508]
                                                                                                                                        Epoch 10/20                                                                                                 | 0.00/468 [00:00<?, ?batch/s]
468/468 ━━━━━━━━━━━━━━━━━━━━ 205s 438ms/step - accuracy: 0.8582 - loss: 0.4174 - val_accuracy: 0.6218 - val_loss: 1.395122, val_loss=1.4]
100%|████████████████████████████████████████████████████████████████████| 468/468 [03:25<00:00, 2.28batch/s, accuracy=0.847, loss=0.447]
                                                                                                                                        Epoch 11/20                                                                                                 | 0.00/468 [00:00<?, ?batch/s]
468/468 ━━━━━━━━━━━━━━━━━━━━ 206s 440ms/step - accuracy: 0.8740 - loss: 0.3712 - val_accuracy: 0.6254 - val_loss: 1.498925, val_loss=1.5]
100%|██████████████████████████████████████████████████████████████████████| 468/468 [03:25<00:00, 2.27batch/s, accuracy=0.862, loss=0.4]
                                                                                                                                        Epoch 12/20                                                                                                 | 0.00/468 [00:00<?, ?batch/s]
468/468 ━━━━━━━━━━━━━━━━━━━━ 204s 435ms/step - accuracy: 0.8891 - loss: 0.3256 - val_accuracy: 0.6230 - val_loss: 1.57153, val_loss=1.57]
100%|████████████████████████████████████████████████████████████████████| 468/468 [03:23<00:00, 2.30batch/s, accuracy=0.877, loss=0.356]
                                                                                                                                        Epoch 13/20                                                                                                 | 0.00/468 [00:00<?, ?batch/s]
468/468 ━━━━━━━━━━━━━━━━━━━━ 204s 437ms/step - accuracy: 0.9002 - loss: 0.2929 - val_accuracy: 0.6244 - val_loss: 1.67864, val_loss=1.68]
100%|█████████████████████████████████████████████████████████████████████| 468/468 [03:24<00:00, 2.29batch/s, accuracy=0.889, loss=0.32]
                                                                                                                                        Epoch 14/20                                                                                                 | 0.00/468 [00:00<?, ?batch/s]
468/468 ━━━━━━━━━━━━━━━━━━━━ 9640s 21s/step - accuracy: 0.9108 - loss: 0.2584 - val_accuracy: 0.6267 - val_loss: 1.809227, val_loss=1.81]
100%|████████████████████████████████████████████████████████████████████| 468/468 [2:40:40<00:00, 20.6s/batch, accuracy=0.9, loss=0.287]
                                                                                                                                        Epoch 15/20                                                                                                 | 0.00/468 [00:00<?, ?batch/s]
468/468 ━━━━━━━━━━━━━━━━━━━━ 22202s 48s/step - accuracy: 0.9202 - loss: 0.2332 - val_accuracy: 0.6271 - val_loss: 1.85727, val_loss=1.86]
100%|██████████████████████████████████████████████████████████████████| 468/468 [6:10:02<00:00, 47.4s/batch, accuracy=0.911, loss=0.257]
                                                                                                                                        Epoch 16/20                                                                                                 | 0.00/468 [00:00<?, ?batch/s]
468/468 ━━━━━━━━━━━━━━━━━━━━ 204s 436ms/step - accuracy: 0.9256 - loss: 0.2187 - val_accuracy: 0.6231 - val_loss: 1.9951.623, val_loss=2]
100%|████████████████████████████████████████████████████████████████████| 468/468 [03:24<00:00, 2.29batch/s, accuracy=0.918, loss=0.237]
                                                                                                                                        Epoch 17/20                                                                                                 | 0.00/468 [00:00<?, ?batch/s]
468/468 ━━━━━━━━━━━━━━━━━━━━ 204s 436ms/step - accuracy: 0.9317 - loss: 0.1984 - val_accuracy: 0.6271 - val_loss: 2.02877, val_loss=2.03]
100%|█████████████████████████████████████████████████████████████████████| 468/468 [03:24<00:00, 2.29batch/s, accuracy=0.924, loss=0.22]
                                                                                                                                        Epoch 18/20                                                                                                 | 0.00/468 [00:00<?, ?batch/s]
468/468 ━━━━━━━━━━━━━━━━━━━━ 204s 436ms/step - accuracy: 0.9383 - loss: 0.1793 - val_accuracy: 0.6228 - val_loss: 2.102423, val_loss=2.1]
100%|█████████████████████████████████████████████████████████████████████| 468/468 [03:24<00:00, 2.29batch/s, accuracy=0.93, loss=0.201]
                                                                                                                                        Epoch 19/20                                                                                                 | 0.00/468 [00:00<?, ?batch/s]
468/468 ━━━━━━━━━━━━━━━━━━━━ 204s 437ms/step - accuracy: 0.9397 - loss: 0.1747 - val_accuracy: 0.6248 - val_loss: 2.22975, val_loss=2.23]
100%|████████████████████████████████████████████████████████████████████| 468/468 [03:24<00:00, 2.29batch/s, accuracy=0.934, loss=0.191]
                                                                                                                                        Epoch 20/20                                                                                                 | 0.00/468 [00:00<?, ?batch/s]
468/468 ━━━━━━━━━━━━━━━━━━━━ 211s 450ms/step - accuracy: 0.9480 - loss: 0.1527 - val_accuracy: 0.6197 - val_loss: 2.33722, val_loss=2.34]
100%|█████████████████████████████████████████████████████████████████████| 468/468 [03:30<00:00, 2.22batch/s, accuracy=0.94, loss=0.173]
100%|████████████████████████████████| 20/20 [9:52:06<00:00, 1776.34s/epoch, accuracy=0.94, loss=0.173, val_accuracy=0.62, val_loss=2.34]
[INFO] The CNN model was successfully trained.
⏱️ Training completed in 35530.50 seconds

📊 Evaluating model on dev set...
1869/1869 - 16s - 9ms/step - accuracy: 0.6197 - loss: 2.3372
✅ Dev accuracy: 61.97%
1869/1869 ━━━━━━━━━━━━━━━━━━━━ 16s 8ms/step  
✅ Saved confusion matrix to /Users/triductran/SpartanDev/my-work/datn/HarmonySeeker-ai/SongChordRecognizer_Pipeline/my_confusion_matrix/cnn_sr22050_hop512_ws8_nf100_cqtspec_noFlat_nonNormC.png

💾 Saving model...
[INFO] The CNN model was saved successfully
"""
