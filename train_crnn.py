from SongChordRecognizer_Training.Datasets import IsophonicsDataset
from SongChordRecognizer_Training.Models import CRNN_basic_WithStandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import time
import tensorflow as tf

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

# ======= Create model =======
print("🧠 Initializing CRNN model with StandardScaler...")
model = CRNN_basic_WithStandardScaler(
    input_shape=(n_frames, n_chromas, 1), output_classes=25, init=True
)

# ======= Train model =======
print("\n🚀 Training model...")
start = time.time()
model.fit(train_x, train_y, dev_x, dev_y, epochs=30, batch_size=512)
end = time.time()
print(f"⏱️ Training completed in {end - start:.2f} seconds")

# ======= Evaluate model =======
print("\n📊 Evaluating model on dev set...")
dev_acc = model.score(dev_x, dev_y)
print(f"✅ Dev accuracy: {dev_acc * 100:.2f}%")

# ======= Confusion matrix =======
try:
    model.model.display_confusion_matrix(
        dev_x,
        dev_y,
        save_path="/Users/triductran/SpartanDev/my-work/datn/HarmonySeeker-ai/SongChordRecognizer_Pipeline/my_models/confusion_matrix_sr22050_hop512_ws8_nf100_cqtspec_noFlat_nonNormC.png",
    )
except Exception as e:
    print(f"⚠️ Error displaying confusion matrix: {e}")

# ======= Save model =======
print("\n💾 Saving model and scaler...")
model.save(
    model_path="/Users/triductran/SpartanDev/my-work/datn/HarmonySeeker-ai/SongChordRecognizer_Pipeline/my_models/crnn_sr22050_hop512_ws8_nf100_cqtspec_noFlat_nonNormC.keras",
    preprocessor_path="/Users/triductran/SpartanDev/my-work/datn/HarmonySeeker-ai/SongChordRecognizer_Pipeline/my_models/preprocessor__sr22050_hop512_ws8_nf100_cqtspec_noFlat_nonNormC.bin",
)

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
_________________________________________________________________
[[INFO] The CRNN model was successfully created.
Epoch 1/75
40/40 [==============================] - 33s 617ms/step - loss: 2.4952 - accuracy: 0.2601 - val_loss: 3.3328 - val_accuracy: 0.1507
Epoch 2/75
40/40 [==============================] - 22s 539ms/step - loss: 2.2048 - accuracy: 0.3213 - val_loss: 3.3264 - val_accuracy: 0.1507
Epoch 3/75
40/40 [==============================] - 22s 540ms/step - loss: 2.1659 - accuracy: 0.3270 - val_loss: 4.4865 - val_accuracy: 0.1507
Epoch 4/75
40/40 [==============================] - 22s 541ms/step - loss: 2.1441 - accuracy: 0.3334 - val_loss: 3.5667 - val_accuracy: 0.1507
Epoch 5/75
40/40 [==============================] - 22s 539ms/step - loss: 2.0922 - accuracy: 0.3485 - val_loss: 2.5361 - val_accuracy: 0.1651
Epoch 6/75
40/40 [==============================] - 22s 539ms/step - loss: 1.9520 - accuracy: 0.4005 - val_loss: 2.2939 - val_accuracy: 0.2709
Epoch 7/75
40/40 [==============================] - 22s 541ms/step - loss: 1.8042 - accuracy: 0.4478 - val_loss: 2.7215 - val_accuracy: 0.1800
Epoch 8/75
40/40 [==============================] - 22s 541ms/step - loss: 1.7316 - accuracy: 0.4755 - val_loss: 2.4425 - val_accuracy: 0.2703
Epoch 9/75
40/40 [==============================] - 22s 541ms/step - loss: 1.5673 - accuracy: 0.5288 - val_loss: 2.1774 - val_accuracy: 0.3438
Epoch 10/75
40/40 [==============================] - 22s 540ms/step - loss: 1.4781 - accuracy: 0.5544 - val_loss: 1.9907 - val_accuracy: 0.4202
Epoch 11/75
40/40 [==============================] - 22s 540ms/step - loss: 1.3499 - accuracy: 0.5819 - val_loss: 1.7041 - val_accuracy: 0.4986
Epoch 12/75
40/40 [==============================] - 22s 540ms/step - loss: 1.3081 - accuracy: 0.5985 - val_loss: 1.5148 - val_accuracy: 0.5338
Epoch 13/75
40/40 [==============================] - 21s 539ms/step - loss: 1.1980 - accuracy: 0.6263 - val_loss: 1.4425 - val_accuracy: 0.5602
Epoch 14/75
40/40 [==============================] - 22s 541ms/step - loss: 1.1163 - accuracy: 0.6519 - val_loss: 1.3883 - val_accuracy: 0.5781
Epoch 15/75
40/40 [==============================] - 22s 540ms/step - loss: 1.0463 - accuracy: 0.6720 - val_loss: 1.3393 - val_accuracy: 0.5904
Epoch 16/75
40/40 [==============================] - 22s 541ms/step - loss: 0.9900 - accuracy: 0.6911 - val_loss: 1.2852 - val_accuracy: 0.6073
Epoch 17/75
40/40 [==============================] - 22s 539ms/step - loss: 0.9125 - accuracy: 0.7149 - val_loss: 1.2766 - val_accuracy: 0.6119
Epoch 18/75
40/40 [==============================] - 22s 542ms/step - loss: 0.8428 - accuracy: 0.7383 - val_loss: 1.2434 - val_accuracy: 0.6198
Epoch 19/75
40/40 [==============================] - 22s 539ms/step - loss: 0.7928 - accuracy: 0.7560 - val_loss: 1.3460 - val_accuracy: 0.6027
Epoch 20/75
40/40 [==============================] - 22s 542ms/step - loss: 0.7138 - accuracy: 0.7743 - val_loss: 1.2172 - val_accuracy: 0.6318
Epoch 21/75
40/40 [==============================] - 21s 538ms/step - loss: 0.6649 - accuracy: 0.7907 - val_loss: 1.2287 - val_accuracy: 0.6317
Epoch 22/75
40/40 [==============================] - 22s 539ms/step - loss: 0.6093 - accuracy: 0.8077 - val_loss: 1.2000 - val_accuracy: 0.6408
Epoch 23/75
40/40 [==============================] - 22s 540ms/step - loss: 0.5887 - accuracy: 0.8147 - val_loss: 1.2264 - val_accuracy: 0.6403
Epoch 24/75
40/40 [==============================] - 22s 540ms/step - loss: 0.5262 - accuracy: 0.8367 - val_loss: 1.2584 - val_accuracy: 0.6134
Epoch 25/75
40/40 [==============================] - 22s 539ms/step - loss: 0.4844 - accuracy: 0.8485 - val_loss: 1.1958 - val_accuracy: 0.6483
Epoch 26/75
40/40 [==============================] - 22s 540ms/step - loss: 0.4578 - accuracy: 0.8589 - val_loss: 1.2029 - val_accuracy: 0.6464
Epoch 27/75
40/40 [==============================] - 22s 542ms/step - loss: 0.4361 - accuracy: 0.8649 - val_loss: 1.2386 - val_accuracy: 0.6432
Epoch 28/75
40/40 [==============================] - 22s 540ms/step - loss: 0.4182 - accuracy: 0.8726 - val_loss: 1.2000 - val_accuracy: 0.6532
Epoch 29/75
40/40 [==============================] - 22s 540ms/step - loss: 0.4102 - accuracy: 0.8717 - val_loss: 1.2262 - val_accuracy: 0.6544
Epoch 30/75
40/40 [==============================] - 21s 538ms/step - loss: 0.3573 - accuracy: 0.8897 - val_loss: 1.2025 - val_accuracy: 0.6588
Epoch 31/75
40/40 [==============================] - 22s 539ms/step - loss: 0.3169 - accuracy: 0.9026 - val_loss: 1.1944 - val_accuracy: 0.6661
Epoch 32/75
40/40 [==============================] - 21s 539ms/step - loss: 0.2845 - accuracy: 0.9122 - val_loss: 1.2112 - val_accuracy: 0.6569
Epoch 33/75
40/40 [==============================] - 22s 539ms/step - loss: 0.2645 - accuracy: 0.9195 - val_loss: 1.2178 - val_accuracy: 0.6660
Epoch 34/75
40/40 [==============================] - 21s 539ms/step - loss: 0.2411 - accuracy: 0.9268 - val_loss: 1.2136 - val_accuracy: 0.6694
Epoch 35/75
40/40 [==============================] - 22s 542ms/step - loss: 0.2307 - accuracy: 0.9309 - val_loss: 1.2414 - val_accuracy: 0.6645
Epoch 36/75
40/40 [==============================] - 22s 540ms/step - loss: 0.2081 - accuracy: 0.9374 - val_loss: 1.2404 - val_accuracy: 0.6697
Epoch 37/75
40/40 [==============================] - 22s 542ms/step - loss: 0.1857 - accuracy: 0.9448 - val_loss: 1.2491 - val_accuracy: 0.6699
Epoch 38/75
40/40 [==============================] - 22s 540ms/step - loss: 0.1843 - accuracy: 0.9449 - val_loss: 1.2442 - val_accuracy: 0.6713
Epoch 39/75
40/40 [==============================] - 22s 539ms/step - loss: 0.1613 - accuracy: 0.9522 - val_loss: 1.2648 - val_accuracy: 0.6694
Epoch 40/75
40/40 [==============================] - 22s 540ms/step - loss: 0.1594 - accuracy: 0.9527 - val_loss: 1.2768 - val_accuracy: 0.6663
Epoch 41/75
40/40 [==============================] - 22s 542ms/step - loss: 0.1524 - accuracy: 0.9545 - val_loss: 1.2721 - val_accuracy: 0.6732
Epoch 42/75
40/40 [==============================] - 22s 541ms/step - loss: 0.1354 - accuracy: 0.9595 - val_loss: 1.2956 - val_accuracy: 0.6712
Epoch 43/75
40/40 [==============================] - 21s 539ms/step - loss: 0.1314 - accuracy: 0.9610 - val_loss: 1.3085 - val_accuracy: 0.6706
Epoch 44/75
40/40 [==============================] - 22s 540ms/step - loss: 0.1315 - accuracy: 0.9605 - val_loss: 1.2998 - val_accuracy: 0.6770
Epoch 45/75
40/40 [==============================] - 22s 540ms/step - loss: 0.1224 - accuracy: 0.9632 - val_loss: 1.3057 - val_accuracy: 0.6728
Epoch 46/75
40/40 [==============================] - 21s 538ms/step - loss: 0.1121 - accuracy: 0.9661 - val_loss: 1.3098 - val_accuracy: 0.6729
Epoch 47/75
40/40 [==============================] - 22s 539ms/step - loss: 0.1094 - accuracy: 0.9675 - val_loss: 1.3434 - val_accuracy: 0.6742
Epoch 48/75
40/40 [==============================] - 22s 540ms/step - loss: 0.1018 - accuracy: 0.9699 - val_loss: 1.3369 - val_accuracy: 0.6738
Epoch 49/75
40/40 [==============================] - 22s 539ms/step - loss: 0.0973 - accuracy: 0.9706 - val_loss: 1.3282 - val_accuracy: 0.6759
Epoch 50/75
40/40 [==============================] - 22s 542ms/step - loss: 0.0999 - accuracy: 0.9692 - val_loss: 1.3454 - val_accuracy: 0.6771
Epoch 51/75
40/40 [==============================] - 22s 541ms/step - loss: 0.0956 - accuracy: 0.9708 - val_loss: 1.3524 - val_accuracy: 0.6773
Epoch 52/75
40/40 [==============================] - 22s 540ms/step - loss: 0.0898 - accuracy: 0.9728 - val_loss: 1.3595 - val_accuracy: 0.6781
Epoch 53/75
40/40 [==============================] - 22s 543ms/step - loss: 0.0831 - accuracy: 0.9751 - val_loss: 1.3832 - val_accuracy: 0.6776
Epoch 54/75
40/40 [==============================] - 22s 539ms/step - loss: 0.0812 - accuracy: 0.9751 - val_loss: 1.4162 - val_accuracy: 0.6770
Epoch 55/75
40/40 [==============================] - 21s 538ms/step - loss: 0.0820 - accuracy: 0.9752 - val_loss: 1.4136 - val_accuracy: 0.6765
Epoch 56/75
40/40 [==============================] - 22s 542ms/step - loss: 0.0810 - accuracy: 0.9751 - val_loss: 1.4040 - val_accuracy: 0.6751
Epoch 57/75
40/40 [==============================] - 22s 540ms/step - loss: 0.0990 - accuracy: 0.9685 - val_loss: 1.4571 - val_accuracy: 0.6618
Epoch 58/75
40/40 [==============================] - 22s 540ms/step - loss: 0.0936 - accuracy: 0.9703 - val_loss: 1.4326 - val_accuracy: 0.6706
Epoch 59/75
40/40 [==============================] - 22s 540ms/step - loss: 0.0884 - accuracy: 0.9717 - val_loss: 1.4221 - val_accuracy: 0.6697
Epoch 60/75
40/40 [==============================] - 22s 539ms/step - loss: 0.0843 - accuracy: 0.9731 - val_loss: 1.4305 - val_accuracy: 0.6701
Epoch 61/75
40/40 [==============================] - 21s 539ms/step - loss: 0.0815 - accuracy: 0.9740 - val_loss: 1.4540 - val_accuracy: 0.6705
Epoch 62/75
40/40 [==============================] - 22s 540ms/step - loss: 0.0803 - accuracy: 0.9744 - val_loss: 1.4304 - val_accuracy: 0.6746
Epoch 63/75
40/40 [==============================] - 22s 541ms/step - loss: 0.1045 - accuracy: 0.9659 - val_loss: 1.5596 - val_accuracy: 0.6325
Epoch 64/75
40/40 [==============================] - 22s 541ms/step - loss: 0.2388 - accuracy: 0.9181 - val_loss: 1.5609 - val_accuracy: 0.6124
Epoch 65/75
40/40 [==============================] - 22s 541ms/step - loss: 0.3161 - accuracy: 0.8926 - val_loss: 1.6199 - val_accuracy: 0.5928
Epoch 66/75
40/40 [==============================] - 22s 539ms/step - loss: 0.3001 - accuracy: 0.8976 - val_loss: 1.4718 - val_accuracy: 0.6316
Epoch 67/75
40/40 [==============================] - 22s 539ms/step - loss: 0.2298 - accuracy: 0.9233 - val_loss: 1.4776 - val_accuracy: 0.6400
Epoch 68/75
40/40 [==============================] - 22s 539ms/step - loss: 0.1764 - accuracy: 0.9419 - val_loss: 1.3913 - val_accuracy: 0.6630
Epoch 69/75
40/40 [==============================] - 21s 538ms/step - loss: 0.1237 - accuracy: 0.9599 - val_loss: 1.4280 - val_accuracy: 0.6687
Epoch 70/75
40/40 [==============================] - 21s 538ms/step - loss: 0.0962 - accuracy: 0.9689 - val_loss: 1.4189 - val_accuracy: 0.6738
Epoch 71/75
40/40 [==============================] - 22s 539ms/step - loss: 0.0798 - accuracy: 0.9751 - val_loss: 1.4202 - val_accuracy: 0.6800
Epoch 72/75
40/40 [==============================] - 22s 540ms/step - loss: 0.0713 - accuracy: 0.9780 - val_loss: 1.4364 - val_accuracy: 0.6807
Epoch 73/75
40/40 [==============================] - 21s 538ms/step - loss: 0.0618 - accuracy: 0.9807 - val_loss: 1.4482 - val_accuracy: 0.6831
Epoch 74/75
40/40 [==============================] - 22s 541ms/step - loss: 0.0550 - accuracy: 0.9835 - val_loss: 1.4644 - val_accuracy: 0.6814
Epoch 75/75
40/40 [==============================] - 22s 539ms/step - loss: 0.0501 - accuracy: 0.9849 - val_loss: 1.4805 - val_accuracy: 0.6812
[INFO] The CRNN model was successfully trained.

17/17 - 2s - loss: 1.4805 - accuracy: 0.6812

 Dev accuracy:  68.12 % 
"""
