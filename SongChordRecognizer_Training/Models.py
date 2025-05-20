#!/usr/bin/env python3
import mir_eval
from sklearn.metrics._plot.confusion_matrix import ConfusionMatrixDisplay
import tensorflow
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
import numpy as np
import sklearn.pipeline
import sklearn.preprocessing
from SongChordRecognizer_Training.SegmentationModels import SegmentationCRNN
from SongChordRecognizer_Training.efficient_net import EfficientNetB0
import lzma
import os
import pickle
from sklearn.model_selection import cross_val_score
import sklearn_crfsuite
from sklearn_crfsuite import CRF
from joblib import dump, load
from tensorflow.keras.layers import GRU, Bidirectional


class MLP:
    """
    Very simple MLP model with one 100 units layer.
    """

    def __init__(self, max_iter=500, random_state=1):
        self.model = MLPClassifier(max_iter=max_iter, random_state=random_state)
        print("[INFO] The MLP model was successfully created.")

    def fit(self, data, targets):
        self.model.fit(data, targets)
        print("[INFO] The MLP model was successfully trained.")

    def predict(self, data):
        return self.model.predict(data)

    def score(self, data, targets):
        return self.model.score(data, targets)

    def display_confusion_matrix(
        self,
        data,
        targets,
        save_path="/Users/triductran/SpartanDev/my-work/key-finding/SongChordsRecognizer/ACR_Pipeline/my_confusion_matrix",
    ):
        try:
            import matplotlib.pyplot as plt
            from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
            import numpy as np
            import os
            from datetime import datetime

            # Labels for confusion matrix
            display_labels = np.array(
                [
                    "N",
                    "C",
                    "C:min",
                    "C#",
                    "C#:min",
                    "D",
                    "D:min",
                    "D#",
                    "D#:min",
                    "E",
                    "E:min",
                    "F",
                    "F:min",
                    "F#",
                    "F#:min",
                    "G",
                    "G:min",
                    "G#",
                    "G#:min",
                    "A",
                    "A:min",
                    "A#",
                    "A#:min",
                    "B",
                    "B:min",
                ]
            )

            # Get predictions from model
            predictions = self.model.predict(data)

            # Create confusion matrix
            cm = confusion_matrix(targets, predictions)

            # Display matrix
            disp = ConfusionMatrixDisplay(
                confusion_matrix=cm, display_labels=display_labels
            )
            fig, ax = plt.subplots(figsize=(12, 10))
            disp.plot(
                ax=ax, xticks_rotation="vertical", include_values=False, cmap="viridis"
            )
            plt.title("MLP Model Confusion Matrix")
            plt.tight_layout()

            # Create directory if it doesn't exist
            os.makedirs(save_path, exist_ok=True)

            # Create filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(save_path, f"confusion_matrix_{timestamp}.png")

            # Save image
            plt.savefig(filename, dpi=300, bbox_inches="tight")
            print(f"✅ Confusion matrix saved to: {filename}")

            # Display (optional)
            plt.show()

        except Exception as e:
            print(f"⚠️ Warning: Could not display or save confusion matrix: {e}")
            print("Continuing with training process...")

    def save(self, model_path="./model.model"):
        # Save this model.
        with lzma.open(model_path, "wb") as model_file:
            pickle.dump(self, model_file)
        print("[INFO] The MLP model was saved successfully")

    @staticmethod
    def load(model_path="./model.model") -> "MLP":
        # Load MLP model
        with lzma.open(model_path, "rb") as model_file:
            model = pickle.load(model_file)
        return model

    def display_cross_val_score(self, X, y):
        score = cross_val_score(self.model, X, y)
        print(
            "Cross validation score: ",
            100 * score,
            " | Average score: ",
            100 * np.mean(score),
        )

    @staticmethod
    def mir_score(x, y):
        (ref_intervals, ref_labels) = mir_eval.io.load_labeled_intervals(x)
        (est_intervals, est_labels) = mir_eval.io.load_labeled_intervals(y)
        est_intervals, est_labels = mir_eval.util.adjust_intervals(
            est_intervals,
            est_labels,
            ref_intervals.min(),
            ref_intervals.max(),
            mir_eval.chord.NO_CHORD,
            mir_eval.chord.NO_CHORD,
        )
        (intervals, ref_labels, est_labels) = mir_eval.util.merge_labeled_intervals(
            ref_intervals, ref_labels, est_intervals, est_labels
        )
        durations = mir_eval.util.intervals_to_durations(intervals)
        comparisons = mir_eval.chord.triads(ref_labels, est_labels)
        score = mir_eval.chord.weighted_accuracy(comparisons, durations)
        return score


class MLP_scalered(MLP):
    def __init__(self, max_iter=500, random_state=1):
        scalered_mlp = sklearn.pipeline.Pipeline(
            [
                (
                    "scaler",
                    sklearn.preprocessing.StandardScaler(with_mean=True, with_std=True),
                ),
                (
                    "estimator",
                    MLPClassifier(max_iter=max_iter, random_state=random_state),
                ),
            ]
        )
        self.model = scalered_mlp
        print(
            "[INFO] The MLP model with scaler preprocessing was successfully created."
        )


class CRNN:
    """
    Very basic CRNN model, maybe not working, who knows.
    """

    def __init__(
        self,
    ):
        input_shape = (1000, 252, 1)
        output_classes = 25
        # Create model
        model = tensorflow.keras.models.Sequential()

        # Feature Extractor
        model.add(
            tensorflow.keras.layers.Conv2D(
                32, (3, 3), activation="relu", input_shape=input_shape, padding="same"
            )
        )
        model.add(tensorflow.keras.layers.MaxPooling2D((2, 2), padding="same"))
        model.add(
            tensorflow.keras.layers.Conv2D(
                64, (3, 3), activation="relu", padding="same"
            )
        )
        model.add(tensorflow.keras.layers.MaxPooling2D((2, 2), padding="same"))
        model.add(
            tensorflow.keras.layers.Conv2D(
                64, (3, 3), activation="relu", padding="same"
            )
        )

        # Classifier - RNN
        model.add(tensorflow.keras.layers.Flatten())
        model.add(tensorflow.keras.layers.Dense(64, activation="relu"))
        model.add(tensorflow.keras.layers.Dense(output_classes, activation="softmax"))

        # Compile model
        model.compile(
            optimizer=tensorflow.keras.optimizers.Adam(learning_rate=0.1),
            loss=tensorflow.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True
            ),
            metrics=["accuracy"],
        )

        self.model = model

    def fit(self, data, targets, dev_data=[], dev_targets=[], epochs=50):
        if dev_data == [] or dev_targets == []:
            validation_data = None
        else:
            validation_data = (dev_data, dev_targets)

        # Train model
        self.history = self.model.fit(
            data, targets, epochs=epochs, validation_data=validation_data
        )
        print("[INFO] The CRNN model was successfully trained.")

    def score(self, data, targets):
        _, test_acc = self.model.evaluate(data, targets, verbose=2)
        return test_acc

    def predict(self, data):
        return self.model.predict(data)

    def display_training_progress(self):
        plt.plot(self.history.history["accuracy"], label="accuracy")
        plt.plot(self.history.history["val_accuracy"], label="val_accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.ylim([0.0, 1])
        plt.legend(loc="lower right")
        plt.show()

    def save(self, model_path="./model.h5"):
        # Save this model.
        self.model.save(model_path)
        print("[INFO] The CRNN model was saved successfully")

    def load(self, model_path="./model.h5"):
        # Load tensorflow model
        self.model = tensorflow.keras.models.load_model(model_path)

    def display_confusion_matrix(self, data, targets):
        # Define labels
        display_labels = np.array(
            [
                "N",
                "C",
                "C:min",
                "C#",
                "C#:min",
                "D",
                "D:min",
                "D#",
                "D#:min",
                "E",
                "E:min",
                "F",
                "F:min",
                "F#",
                "F#:min",
                "G",
                "G:min",
                "G#",
                "G#:min",
                "A",
                "A:min",
                "A#",
                "A#:min",
                "B",
                "B:min",
            ]
        )
        labels = np.array([i for i in range(len(display_labels))])

        # Generate predictions and targets
        predictions = self.model.predict(data)
        a1, a2, a3 = predictions.shape
        predictions = predictions.reshape((a1 * a2, a3))
        predictions = tensorflow.argmax(predictions, axis=1)
        a1, a2 = targets.shape
        targets = targets.reshape((a1 * a2))

        # Set and display confusion matrix
        disp = ConfusionMatrixDisplay(
            confusion_matrix=confusion_matrix(
                targets, predictions, labels=labels, normalize="all"
            ),
            display_labels=display_labels,
        )
        disp.plot(xticks_rotation="vertical", include_values=False)


class CRNN_efficient(CRNN):
    """
    Efficient net based CRNN
    """

    def __init__(self, input_shape, output_classes, random_seed=7):
        # Set tensorflow random seed
        np.random.seed(random_seed)
        # Create model
        n_frames, n_features, chanells = input_shape

        efficent_model = EfficientNetB0(
            include_top=False, input_shape=input_shape, weights=None, classes=25
        )

        inputs = tensorflow.keras.layers.Input(shape=input_shape)
        hidden = efficent_model(inputs)
        x = tensorflow.keras.layers.Reshape((n_frames, -1))(hidden[-1])
        x = tensorflow.keras.layers.Bidirectional(
            tensorflow.keras.layers.GRU(512, return_sequences=True, dropout=0.5)
        )(x)
        x = tensorflow.keras.layers.Bidirectional(
            tensorflow.keras.layers.GRU(32, return_sequences=True, dropout=0.5)
        )(x)
        outputs = tensorflow.keras.layers.Dense(output_classes, activation="softmax")(x)

        model = tensorflow.keras.Model(inputs=inputs, outputs=outputs)

        # Compile model
        model.compile(
            optimizer=tensorflow.keras.optimizers.Adam(),
            loss=tensorflow.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True
            ),
            metrics=["accuracy"],
        )

        model.summary()
        self.model = model
        print("[INFO] The CRNN model was successfully created.")


class CRNN_basic(CRNN):
    """
    CRNN model inspired by Junyan Jiang, Ke Chen, Wei li, Gus Xia, 2019.
    """

    def __init__(self, input_shape, output_classes):

        # Create model
        model = tensorflow.keras.models.Sequential()
        # Feature Extractor
        model.add(
            tensorflow.keras.layers.Conv2D(
                16, (3, 3), activation="relu", input_shape=input_shape, padding="same"
            )
        )
        model.add(tensorflow.keras.layers.BatchNormalization())
        model.add(
            tensorflow.keras.layers.Conv2D(
                16, (3, 3), activation="relu", padding="same"
            )
        )
        model.add(tensorflow.keras.layers.BatchNormalization())
        model.add(
            tensorflow.keras.layers.Conv2D(
                16, (3, 3), activation="relu", padding="same"
            )
        )
        model.add(tensorflow.keras.layers.BatchNormalization())
        model.add(tensorflow.keras.layers.MaxPooling2D((1, 3), padding="same"))
        model.add(
            tensorflow.keras.layers.Conv2D(
                32, (3, 3), activation="relu", padding="same"
            )
        )
        model.add(tensorflow.keras.layers.BatchNormalization())
        model.add(
            tensorflow.keras.layers.Conv2D(
                32, (3, 3), activation="relu", padding="same"
            )
        )
        model.add(tensorflow.keras.layers.BatchNormalization())
        model.add(
            tensorflow.keras.layers.Conv2D(
                32, (3, 3), activation="relu", padding="same"
            )
        )
        model.add(tensorflow.keras.layers.BatchNormalization())
        # Doesn't improve the result
        # model.add(tensorflow.keras.layers.MaxPooling2D((1,3),padding='same'))
        # model.add(tensorflow.keras.layers.Conv2D(64, (3,3), activation='relu',padding='same'))
        # model.add(tensorflow.keras.layers.BatchNormalization())
        # model.add(tensorflow.keras.layers.Conv2D(64, (3,3), activation='relu',padding='same'))
        # model.add(tensorflow.keras.layers.BatchNormalization())
        # model.add(tensorflow.keras.layers.MaxPooling2D((1,4),padding='same'))
        # model.add(tensorflow.keras.layers.Conv2D(80, (3,3), activation='relu',padding='same'))
        # model.add(tensorflow.keras.layers.BatchNormalization())
        # model.add(tensorflow.keras.layers.Conv2D(80, (3,3), activation='relu',padding='same'))
        # model.add(tensorflow.keras.layers.BatchNormalization())

        _, n_frames, _, _ = model.output_shape

        # Classifier - RNN
        model.add(tensorflow.keras.layers.Reshape((n_frames, -1)))

        model.add(
            tensorflow.keras.layers.Bidirectional(
                tensorflow.keras.layers.GRU(128, return_sequences=True, dropout=0.5)
            )
        )
        model.add(
            tensorflow.keras.layers.Bidirectional(
                tensorflow.keras.layers.GRU(16, return_sequences=True, dropout=0.5)
            )
        )

        model.add(tensorflow.keras.layers.Dense(output_classes, activation="softmax"))

        # Compile model
        model.compile(
            optimizer=tensorflow.keras.optimizers.Adam(),
            loss=tensorflow.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True
            ),
            metrics=["accuracy"],
        )

        model.summary()
        self.model = model
        print("[INFO] The CRNN model was successfully created.")


class CRNN_basic_WithStandardScaler:
    """
    CRNN basic that stores also the StandardScaler
    """

    def __init__(self, input_shape=(1000, 252, 1), output_classes=25, init=False):
        if init:
            self.model = CRNN_basic(input_shape, output_classes)
            self.preprocessor = sklearn.preprocessing.StandardScaler()
            print(
                "[INFO] The Baisc CRNN with Standar Scaler was successfully initialized"
            )

    def fit(self, train_x, train_y, dev_data=[], dev_targets=[], epochs=50):
        _, n_frames, n_chromas, _ = train_x.shape
        self.preprocessor.fit(train_x.reshape((-1, 252)))
        print("[INFO] The preprocessor was successfully trained")

        train_x = self.preprocessor.transform(train_x.reshape((-1, 252))).reshape(
            (-1, n_frames, n_chromas, 1)
        )
        self.model.fit(train_x, train_y, dev_data, dev_targets, epochs=epochs)
        print("[INFO] The CRNN model was successfully trained.")

    def score(self, data, targets):
        _, n_frames, n_chromas, _ = data.shape
        transformed_data = self.preprocessor.transform(data.reshape((-1, 252))).reshape(
            (-1, n_frames, n_chromas, 1)
        )
        _, test_acc = self.model.evaluate(transformed_data, targets, verbose=2)
        return test_acc

    def predict(self, data):
        _, n_frames, n_chromas, _ = data.shape
        transformed_data = self.preprocessor.transform(data.reshape((-1, 252))).reshape(
            (-1, n_frames, n_chromas, 1)
        )
        return self.model.predict(transformed_data)

    def save(self, model_path="./model.h5", preprocessor_path="./preprocessor.bin"):
        # Save this model with its scaler.
        self.model.save(model_path)
        print("[INFO] The CRNN model was saved successfully")
        dump(self.preprocessor, preprocessor_path, compress=True)
        print("[INFO] The Preprocessor was saved successfully")

    def load(self, model_path="./model.h5", preprocessor_path="./preprocessor.bin"):
        # Load tensorflow model and scaler
        self.model = tensorflow.keras.models.load_model(model_path)
        self.preprocessor = load(preprocessor_path)


class CRNN_CRF(CRNN):
    """
    CRNN model inspired by Junyan Jiang, Ke Chen, Wei li, Gus Xia, 2019.
    """

    def __init__(self, input_shape, output_classes):
        # Create model
        n_frames, _, _ = input_shape
        input_layer = tensorflow.keras.Input(shape=input_shape)
        x = tensorflow.keras.layers.Conv2D(
            16, (3, 3), activation="relu", input_shape=input_shape, padding="same"
        )(input_layer)
        x = tensorflow.keras.layers.BatchNormalization()(x)
        x = tensorflow.keras.layers.Conv2D(
            16, (3, 3), activation="relu", padding="same"
        )(x)
        x = tensorflow.keras.layers.BatchNormalization()(x)
        x = tensorflow.keras.layers.Conv2D(
            16, (3, 3), activation="relu", padding="same"
        )(x)
        x = tensorflow.keras.layers.BatchNormalization()(x)
        x = tensorflow.keras.layers.MaxPooling2D((1, 3), padding="same")(x)
        x = tensorflow.keras.layers.Conv2D(
            32, (3, 3), activation="relu", padding="same"
        )(x)
        x = tensorflow.keras.layers.BatchNormalization()(x)
        x = tensorflow.keras.layers.Conv2D(
            32, (3, 3), activation="relu", padding="same"
        )(x)
        x = tensorflow.keras.layers.BatchNormalization()(x)
        x = tensorflow.keras.layers.Conv2D(
            32, (3, 3), activation="relu", padding="same"
        )(x)
        x = tensorflow.keras.layers.BatchNormalization()(x)
        x = tensorflow.keras.layers.Reshape((n_frames, -1))(x)
        x = tensorflow.keras.layers.Bidirectional(
            tensorflow.keras.layers.GRU(128, return_sequences=True, dropout=0.5)
        )(x)
        x = tensorflow.keras.layers.Dense(512)(x)
        output = tensorflow.keras.layers.Dense(output_classes, activation="softmax")(x)
        base_model = tensorflow.keras.models.Model(input_layer, output)
        self.model = base_model
        self.model.compile(optimizer="adam")
        base_model.summary()

        print("[INFO] The CRNN model with CRF was successfully created.")

    def fit(self, data, targets, dev_data=[], dev_targets=[], epochs=50):
        if dev_data == [] or dev_targets == []:
            validation_data = None
        else:
            validation_data = (dev_data, dev_targets)

        # Train model
        self.history = None
        self.model.fit(data, targets, epochs=epochs, validation_data=validation_data)
        print("[INFO] The CRNN model was successfully trained.")

    def display_confusion_matrix(self, data, targets):
        # Define labels
        display_labels = np.array(
            [
                "N",
                "C",
                "C:min",
                "C#",
                "C#:min",
                "D",
                "D:min",
                "D#",
                "D#:min",
                "E",
                "E:min",
                "F",
                "F:min",
                "F#",
                "F#:min",
                "G",
                "G:min",
                "G#",
                "G#:min",
                "A",
                "A:min",
                "A#",
                "A#:min",
                "B",
                "B:min",
            ]
        )
        labels = np.array([i for i in range(len(display_labels))])

        # Generate predictions and targets
        predictions = self.model.predict(data)
        a1, a2 = predictions.shape
        predictions = predictions.reshape((a1 * a2))
        a1, a2 = targets.shape
        targets = targets.reshape((a1 * a2))

        # Set and display confusion matrix
        disp = ConfusionMatrixDisplay(
            confusion_matrix=confusion_matrix(
                targets, predictions, labels=labels, normalize="all"
            ),
            display_labels=display_labels,
        )
        disp.plot(xticks_rotation="vertical", include_values=False)


class MLP2RNN:
    """
    sklearn scalered MLP -> tensorflow RNN
    """

    _window_size = 5
    _test_size = 0.3

    def __init__(self, input_shape, output_classes, max_iter=500, random_state=7):
        _, n_frames, _ = input_shape
        # ACOUSTIC model

        # Create Pipeline
        scalered_mlp = sklearn.pipeline.Pipeline(
            [
                (
                    "scaler",
                    sklearn.preprocessing.StandardScaler(with_mean=True, with_std=True),
                ),
                (
                    "estimator",
                    MLPClassifier(max_iter=max_iter, random_state=random_state),
                ),
            ]
        )

        self._acoustic_model = scalered_mlp
        print(
            "[INFO] The MLP model with scaler preprocessing was successfully created."
        )

        # LINGUISTIC model

        # Create model
        model = tensorflow.keras.models.Sequential()

        model.add(
            tensorflow.keras.layers.Bidirectional(
                tensorflow.keras.layers.LSTM(64, return_sequences=True),
                input_shape=(n_frames, output_classes),
            )
        )
        model.add(tensorflow.keras.layers.Dense(32, activation="relu"))
        model.add(tensorflow.keras.layers.Dense(output_classes, activation="softmax"))

        # Compile model
        model.compile(
            optimizer=tensorflow.keras.optimizers.Adam(),
            loss=tensorflow.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True
            ),
            metrics=["accuracy"],
        )

        model.summary()
        self._linguistic_model = model
        print("[INFO] The RNN model was successfully created.")

    def fit(self, data, targets, dev_data, dev_targets, epochs=50):
        test_size = 0.7
        random_state = 42
        train_x, dev_x, train_y, dev_y = sklearn.model_selection.train_test_split(
            data, targets, test_size=test_size, random_state=random_state
        )

        # Train ACOUSTIC model

        # Preprocess acoustic data
        acoustic_data, acoustic_targets = self.preprocess_acoustic(train_x, train_y)
        dev_acoustic_data, dev_acousting_targets = self.preprocess_acoustic(
            dev_x, dev_y
        )
        # Fit model
        self._acoustic_model.fit(acoustic_data, acoustic_targets)
        print(
            "[INFO] The acoustic model was successfully trained with dev accuracy {:.2f}".format(
                100
                * self._acoustic_model.score(dev_acoustic_data, dev_acousting_targets)
            ),
            "%",
        )
        # Display results
        self.display_acoustic_confusion_matrix(dev_acoustic_data, dev_acousting_targets)

        # Train LINGUISTIC model

        # Preprocess linguistic data
        linguistic_data, linguistic_target = self.preprocess_linguistic(dev_x, dev_y)
        dev_linguistic_data, dev_linguistic_targets = self.preprocess_linguistic(
            dev_data, dev_targets
        )
        # Fit model
        self._linguistic_history = self._linguistic_model.fit(
            linguistic_data,
            linguistic_target,
            epochs=epochs,
            validation_data=(dev_linguistic_data, dev_linguistic_targets),
        )
        print(
            "[INFO] The linguistic model was successfully trained with dev accuracy {:.2f}".format(
                100
                * self._linguistic_model.evaluate(
                    dev_linguistic_data, dev_linguistic_targets, verbose=2
                )[1]
            ),
            "%",
        )
        # Display results
        self.display_linguistic_training_progress()
        self.display_linguistic_confusion_matrix(
            dev_linguistic_data, dev_linguistic_targets
        )

    def display_score(self, data, targets):
        window_size = self._window_size
        random_state = None
        linguistic_data, linguistic_targets = [], targets

        # Preprocess data
        _, n_frames, n_features = data.shape
        for sequence in data:
            predictable_sequence = []
            for i in range(n_frames):
                predictable_sequence.append(
                    np.concatenate(
                        (
                            np.zeros((abs(min(0, i - window_size)), n_features)),
                            np.array(
                                sequence[
                                    max(0, i - window_size) : min(
                                        i + window_size + 1, n_frames
                                    ),
                                    :,
                                ]
                            ),
                            np.zeros(
                                (
                                    abs(min(0, (n_frames) - (i + window_size + 1))),
                                    n_features,
                                )
                            ),
                        ),
                        axis=0,
                    ).flatten()
                )
            linguistic_data.append(
                self._acoustic_model.predict_proba(predictable_sequence)
            )

        linguistic_data = np.array(linguistic_data)
        linguistic_targets = np.array(linguistic_targets)

        print(
            "[INFO] The linguistic model was successfully trained with dev accuracy {:.2f}".format(
                100
                * self._linguistic_model.evaluate(
                    linguistic_data, linguistic_targets, verbose=2
                )[1]
            ),
            "%",
        )
        # Display results
        self.display_linguistic_confusion_matrix(linguistic_data, linguistic_targets)

    def preprocess_acoustic(self, data, targets):
        window_size = self._window_size
        test_size = self._test_size
        random_state = None
        acoustic_data, acoustic_targets = [], []

        # Preprocess data and targets
        _, n_frames, n_features = data.shape
        for data_sequence, chord_sequence in zip(data, targets):
            for i in range(n_frames):
                acoustic_data.append(
                    np.concatenate(
                        (
                            np.zeros((abs(min(0, i - window_size)), n_features)),
                            np.array(
                                data_sequence[
                                    max(0, i - window_size) : min(
                                        i + window_size + 1, n_frames
                                    ),
                                    :,
                                ]
                            ),
                            np.zeros(
                                (
                                    abs(min(0, (n_frames) - (i + window_size + 1))),
                                    n_features,
                                )
                            ),
                        ),
                        axis=0,
                    ).flatten()
                )
                acoustic_targets.append(chord_sequence[i])

        # Divide dataset to training and developing sets
        return np.array(acoustic_data), np.array(acoustic_targets)

    def preprocess_linguistic(self, data, targets):
        window_size = self._window_size
        test_size = self._test_size
        random_state = None
        linguistic_data, linguistic_targets = [], targets

        # Preprocess data
        _, n_frames, n_features = data.shape
        for sequence in data:
            predictable_sequence = []
            for i in range(n_frames):
                predictable_sequence.append(
                    np.concatenate(
                        (
                            np.zeros((abs(min(0, i - window_size)), n_features)),
                            np.array(
                                sequence[
                                    max(0, i - window_size) : min(
                                        i + window_size + 1, n_frames
                                    ),
                                    :,
                                ]
                            ),
                            np.zeros(
                                (
                                    abs(min(0, (n_frames) - (i + window_size + 1))),
                                    n_features,
                                )
                            ),
                        ),
                        axis=0,
                    ).flatten()
                )
            linguistic_data.append(
                self._acoustic_model.predict_proba(predictable_sequence)
            )

        return np.array(linguistic_data), np.array(linguistic_targets)

    def display_linguistic_training_progress(self):
        plt.plot(self._linguistic_history.history["accuracy"], label="accuracy")
        plt.plot(self._linguistic_history.history["val_accuracy"], label="val_accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.ylim([0.0, 1])
        plt.legend(loc="lower right")
        plt.show()

    def display_acoustic_confusion_matrix(self, data, targets):
        # Define labels
        display_labels = np.array(
            [
                "N",
                "C",
                "C:min",
                "C#",
                "C#:min",
                "D",
                "D:min",
                "D#",
                "D#:min",
                "E",
                "E:min",
                "F",
                "F:min",
                "F#",
                "F#:min",
                "G",
                "G:min",
                "G#",
                "G#:min",
                "A",
                "A:min",
                "A#",
                "A#:min",
                "B",
                "B:min",
            ]
        )
        labels = np.array([i for i in range(len(display_labels))])

        # Generate predictions
        predictions = self._acoustic_model.predict(data)

        # Set and display confusion matrix
        disp = ConfusionMatrixDisplay(
            confusion_matrix=confusion_matrix(
                targets, predictions, labels=labels, normalize="all"
            ),
            display_labels=display_labels,
        )
        disp.plot(xticks_rotation="vertical", include_values=False)
        plt.show()

    def display_linguistic_confusion_matrix(self, data, targets):
        # Define labels
        display_labels = np.array(
            [
                "N",
                "C",
                "C:min",
                "C#",
                "C#:min",
                "D",
                "D:min",
                "D#",
                "D#:min",
                "E",
                "E:min",
                "F",
                "F:min",
                "F#",
                "F#:min",
                "G",
                "G:min",
                "G#",
                "G#:min",
                "A",
                "A:min",
                "A#",
                "A#:min",
                "B",
                "B:min",
            ]
        )
        labels = np.array([i for i in range(len(display_labels))])

        # Generate predictions and targets
        predictions = self._linguistic_model.predict(data)
        a1, a2, a3 = predictions.shape
        predictions = predictions.reshape((a1 * a2, a3))
        predictions = tensorflow.argmax(predictions, axis=1)
        a1, a2 = targets.shape
        targets = targets.reshape((a1 * a2))

        # Set and display confusion matrix
        disp = ConfusionMatrixDisplay(
            confusion_matrix=confusion_matrix(
                targets, predictions, labels=labels, normalize="all"
            ),
            display_labels=display_labels,
        )
        disp.plot(xticks_rotation="vertical", include_values=False)
        plt.show()


class BassVsThird1:
    """
    Two CRNN models.
    The first one is to recognize major/minor chord, another one is to recognize the bass of the chord.
    """

    def __init__(self, input_shape, output_classes):
        # Create model
        n_frames, _, _ = input_shape
        # Create model
        model = tensorflow.keras.models.Sequential()
        # Feature Extractor
        model.add(
            tensorflow.keras.layers.Conv2D(
                16, (3, 3), activation="relu", input_shape=input_shape, padding="same"
            )
        )
        model.add(tensorflow.keras.layers.BatchNormalization())
        model.add(
            tensorflow.keras.layers.Conv2D(
                16, (3, 3), activation="relu", padding="same"
            )
        )
        model.add(tensorflow.keras.layers.BatchNormalization())
        model.add(
            tensorflow.keras.layers.Conv2D(
                16, (3, 3), activation="relu", padding="same"
            )
        )
        model.add(tensorflow.keras.layers.BatchNormalization())
        model.add(tensorflow.keras.layers.MaxPooling2D((1, 3), padding="same"))
        model.add(
            tensorflow.keras.layers.Conv2D(
                32, (3, 3), activation="relu", padding="same"
            )
        )
        model.add(tensorflow.keras.layers.BatchNormalization())
        model.add(
            tensorflow.keras.layers.Conv2D(
                32, (3, 3), activation="relu", padding="same"
            )
        )
        model.add(tensorflow.keras.layers.BatchNormalization())
        model.add(
            tensorflow.keras.layers.Conv2D(
                32, (3, 3), activation="relu", padding="same"
            )
        )
        model.add(tensorflow.keras.layers.BatchNormalization())

        _, n_frames, _, _ = model.output_shape

        # Classifier - RNN
        model.add(tensorflow.keras.layers.Reshape((n_frames, -1)))
        model.add(
            tensorflow.keras.layers.Bidirectional(
                tensorflow.keras.layers.GRU(128, return_sequences=True, dropout=0.5)
            )
        )
        model.add(
            tensorflow.keras.layers.Bidirectional(
                tensorflow.keras.layers.GRU(16, return_sequences=True, dropout=0.5)
            )
        )
        model.add(tensorflow.keras.layers.Dense(output_classes, activation="softmax"))

        model.summary()

        self._bass_model = tensorflow.keras.models.clone_model(model)
        self._third_model = tensorflow.keras.models.clone_model(model)
        # Compile model
        self._bass_model.compile(
            optimizer=tensorflow.keras.optimizers.Adam(),
            loss=tensorflow.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True
            ),
            metrics=["accuracy"],
        )
        self._third_model.compile(
            optimizer=tensorflow.keras.optimizers.Adam(),
            loss=tensorflow.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True
            ),
            metrics=["accuracy"],
        )

        print("[INFO] The CRNN bass-third model was successfully created.")

    def fit(
        self,
        data,
        targets,
        dev_data=[],
        dev_targets=[],
        bass_epochs=50,
        third_epochs=50,
    ):
        if dev_data == [] or dev_targets == []:
            bass_validation_data, third_validation_data = None, None
        else:
            dev_x, val_bass_targets, val_third_targets = [], [], []
            for x, y in zip(dev_data, dev_targets):
                prep_x, prep_bass_targets, prep_third_targets = (
                    self.preprocess_datataset(x, y)
                )
                dev_x.append(prep_x)
                val_bass_targets.append(prep_bass_targets)
                val_third_targets.append(prep_third_targets)
            bass_validation_data, third_validation_data = (
                np.array(dev_x),
                np.array(val_bass_targets),
            ), (np.array(dev_x), np.array(val_third_targets))

        # Prepare training data
        train_x, bass_targets, third_targets = [], [], []
        for x, y in zip(data, targets):
            prep_x, prep_bass_targets, prep_third_targets = self.preprocess_datataset(
                x, y
            )
            train_x.append(prep_x)
            bass_targets.append(prep_bass_targets)
            third_targets.append(prep_third_targets)

        self._bass_history = self._bass_model.fit(
            np.array(train_x),
            np.array(bass_targets),
            epochs=bass_epochs,
            validation_data=bass_validation_data,
        )
        self._third_history = self._third_model.fit(
            np.array(train_x),
            np.array(third_targets),
            epochs=third_epochs,
            validation_data=third_validation_data,
        )
        print("[INFO] The MLP model was successfully trained.")

    def display_training_progress(self):
        plt.plot(self._bass_history.history["accuracy"], label="accuracy")
        plt.plot(self._bass_history.history["val_accuracy"], label="val_accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.ylim([0.0, 1])
        plt.legend(loc="lower right")
        plt.show()

        plt.plot(self._third_history.history["accuracy"], label="accuracy")
        plt.plot(self._third_history.history["val_accuracy"], label="val_accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.ylim([0.0, 1])
        plt.legend(loc="lower right")
        plt.show()

    def predict(self, data, targets):
        # Preprocess data
        test_x, bass_targets, third_targets = [], [], []
        for x, y in zip(data, targets):
            prep_x, prep_bass_targets, prep_third_targets = self.preprocess_datataset(
                x, y
            )
            test_x.append(prep_x)
            bass_targets.append(prep_bass_targets)
            third_targets.append(prep_third_targets)

        # Predict thirds and basses
        basses = self._bass_model.predict(data)
        thirds = self._third_model.predict(data)

        # Collect thirds and basses to chords
        predictions = []
        for x, y in zip(basses, thirds):
            predictions.append(self.postprocess_targets(x, y))

        return np.array(predictions)

    def score(self, data, targets):
        # Predict targets
        predictions = self.predict(data, targets)

        flatten_predictions = predictions.reshape(-1)
        flatten_targets = (np.array(targets)).reshape(-1)

        return sklearn.metrics.accuracy_score(
            np.array(flatten_predictions), np.array(flatten_targets)
        )

    def preprocess_datataset(self, data, targets):
        bass_targets = []
        third_targets = []
        for chord in targets:
            if chord == 0:
                bass_targets.append(0)
                third_targets.append(0)
            else:
                bass_targets.append((int)((chord + 1) / 2))
                third_targets.append((int)((chord + 1) % 2) + 1)

        return np.array(data), np.array(bass_targets), np.array(third_targets)

    def postprocess_targets(self, bass_targets, third_targets):
        targets = []
        for bass, third in zip(bass_targets, third_targets):
            bass = np.argmax(bass)
            third = np.argmax(third)
            if bass == 0 or third == 0:
                targets.append(0)
            else:
                targets.append((bass - 1) * 2 + third)

        return np.array(targets)

    def save_bass(self, model_path="./bass.h5"):
        # Save this model.
        self._bass_model.save(model_path)
        print("[INFO] The BASS CRNN model was saved successfully")

    def load_bass(self, model_path="./bass.h5"):
        # Load tensorflow model
        self._bass_model = tensorflow.keras.models.load_model(model_path)

    def save_third(self, model_path="./third.h5"):
        # Save this model.
        self._third_model.save(model_path)
        print("[INFO] The THIRD CRNN model was saved successfully")

    def load_third(self, model_path="./third.h5"):
        # Load tensorflow model
        self._third_model = tensorflow.keras.models.load_model(model_path)

    def display_confusion_matrix(self, data, targets):
        # Define labels
        display_labels = np.array(
            [
                "N",
                "C",
                "C:min",
                "C#",
                "C#:min",
                "D",
                "D:min",
                "D#",
                "D#:min",
                "E",
                "E:min",
                "F",
                "F:min",
                "F#",
                "F#:min",
                "G",
                "G:min",
                "G#",
                "G#:min",
                "A",
                "A:min",
                "A#",
                "A#:min",
                "B",
                "B:min",
            ]
        )
        labels = np.array([i for i in range(len(display_labels))])

        # Generate predictions
        predictions = self.predict(data, targets)
        predictions = predictions.reshape((-1))
        targets = (np.array(targets)).reshape((-1))

        # Set and display confusion matrix
        disp = ConfusionMatrixDisplay(
            confusion_matrix=confusion_matrix(
                targets, predictions, labels=labels, normalize="all"
            ),
            display_labels=display_labels,
        )
        disp.plot(xticks_rotation="vertical", include_values=False)
        plt.show()


class SegmentationVoting:
    """
    Two models, one predicting chords, one predicting harmony segments.
    After the training single models, chord predictions of some predicted segments are taken and the most frequent chord is choosed for whole segment.
    """

    def __init__(self, input_shape):
        self.model = CRNN_basic(input_shape=input_shape, output_classes=25)
        self.segmentation_model = SegmentationCRNN(input_shape=input_shape)
        print("[INFO] The SegmentationVoting model was successfully created.")

    def fit(self, data, targets, dev_data, dev_targets):
        self.model.fit(data, targets)
        self.segmentation_model.fit(data, targets, dev_data, dev_targets)
        print("[INFO] The SegmentationVoting model was successfully trained.")

    @staticmethod
    def vote(segmentations, chords):
        chords_to_vote = np.zeros(25)
        voted_chords = []
        start = 0
        # Iterate over all time stamps
        for ind, change, chord in enumerate(zip(segmentations, chords)):
            # If there is a chord change
            if change == 1:
                winner = np.argmax(voted_chords)
                for _ in range(start, ind):
                    voted_chords.append(winner)
                chords_to_vote = np.zeros(25)
                start = ind
            # Add chord occurence
            chords_to_vote[chord] = chords_to_vote[chord] + 1

        return voted_chords

    def predict(self, data, targets):
        # Predict chords and segmentations and basses
        chords = self.model.predict(data)
        segmentations = self.segmentation_model.predict(data)

        # Change chords in segments as a most common segment's chord
        predictions = SegmentationVoting.vote(
            np.concatenate(chords), np.concatenate(segmentations)
        )

        return np.array(predictions)

    def score(self, data, targets):
        # Predict targets
        predictions = self.predict(data, targets)

        return sklearn.metrics.accuracy_score(np.concatenate(targets), predictions)

    def display_confusion_matrix(self, data, targets):
        # Define labels
        display_labels = np.array(
            [
                "N",
                "C",
                "C:min",
                "C#",
                "C#:min",
                "D",
                "D:min",
                "D#",
                "D#:min",
                "E",
                "E:min",
                "F",
                "F:min",
                "F#",
                "F#:min",
                "G",
                "G:min",
                "G#",
                "G#:min",
                "A",
                "A:min",
                "A#",
                "A#:min",
                "B",
                "B:min",
            ]
        )
        labels = np.array([i for i in range(len(display_labels))])

        # Generate predictions
        predictions = self.predict(data, targets)

        # Set and display confusion matrix
        disp = ConfusionMatrixDisplay(
            confusion_matrix=confusion_matrix(
                np.concatenate(targets), predictions, labels=labels, normalize="all"
            ),
            display_labels=display_labels,
        )
        disp.plot(xticks_rotation="vertical", include_values=False)
        plt.show()
