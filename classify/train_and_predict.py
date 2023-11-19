"""Script used to train and test the performance
GZTAN dataset on a defined CNN architecture"""
import argparse
import os
from datetime import datetime
from typing import Dict

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model

from classify.dataset import Dataset
from visualize.plot import output_training_history

DATASET_DIR = "data/genres_original"
DATA_EXT = ".wav"


def load_dataset(
    dataset_directory: str, file_prefix: str, file_suffix: str
) -> Dataset:
    """Loads the saved dataset from disk
    Args:
        dataset_directory: location where dataset files are stored
        file_prefix: Prefix of the dataset files
        file_suffix: Suffix of the dataset files
    Returns:
        A dataset object that contains the filenames, mel spectrograms,
        and labels
    """
    filenames_path = os.path.join(
        dataset_directory, f"{file_prefix}-filenames-{file_suffix}.npz"
    )
    spectrograms_path = os.path.join(
        dataset_directory, f"{file_prefix}-spectrograms-{file_suffix}.npz"
    )
    labels_path = os.path.join(
        dataset_directory, f"{file_prefix}-labels-{file_suffix}.npz"
    )
    with np.load(filenames_path) as npz_filenames, np.load(
        spectrograms_path
    ) as npz_spectrograms, np.load(labels_path) as npz_labels:
        filenames = npz_filenames["filenames"]
        spectrograms = npz_spectrograms["spectrograms"]
        spectrograms = spectrograms.reshape(
            spectrograms.shape[0],
            spectrograms.shape[1],
            spectrograms.shape[2],
            1,
        )
        labels = npz_labels["labels"]
        labels = tf.keras.utils.to_categorical(
            labels, num_classes=10, dtype="float32"
        )
        dataset = Dataset(
            filenames=filenames, spectrograms=spectrograms, labels=labels
        )
    return dataset


def setup_network_architecture(output_directory: str) -> tf.keras.Sequential:
    """Sets up the network architecture and outputs architecture plot
    Args:
        output_directory: location to save the plots
    Returns:
        A Keras Sequential model with 5 2D convolutional layers, max pooling,
        dropout, and 3 fully-connected layers
    """
    model = tf.keras.Sequential(
        [
            Conv2D(
                8,
                (3, 3),
                activation="relu",
                input_shape=(128, 1249, 1),
                padding="same",
            ),
            MaxPooling2D((4, 4), padding="same"),
            Conv2D(16, (5, 5), activation="relu", padding="same"),
            Dropout(0.2),
            MaxPooling2D((4, 4), padding="same"),
            Conv2D(32, (5, 5), activation="relu", padding="same"),
            Dropout(0.2),
            MaxPooling2D((4, 4), padding="same"),
            Conv2D(64, (3, 3), activation="relu", padding="same"),
            Dropout(0.2),
            MaxPooling2D((4, 4), padding="same"),
            Conv2D(64, (3, 3), activation="relu", padding="same"),
            Dropout(0.2),
            MaxPooling2D((4, 4), padding="same"),
            Flatten(),
            Dense(128, activation="relu"),
            Dropout(0.5),
            Dense(64, activation="relu"),
            Dropout(0.5),
            Dense(10, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="Adam", loss="binary_crossentropy", metrics=["accuracy"]
    )
    model.summary()
    plot_model(
        model,
        to_file=os.path.join(output_directory, "model_plot.png"),
        show_shapes=True,
        show_layer_names=True,
    )
    return model


def main(args: Dict) -> None:
    """Main entry point for the script.  Loads the dataset, sets up the network
        architecture, trains and tests the model
    Args:
        args: arguments passed to argparse
    """
    dataset = load_dataset(
        args.dataset_directory, args.dataset_prefix, args.dataset_suffix
    )
    model = setup_network_architecture(args.output_directory)
    not_test_data, test_data, not_test_labels, test_labels = train_test_split(
        dataset.spectrograms,
        dataset.labels,
        test_size=150,
        shuffle=True,
        random_state=719,
    )
    train_data, eval_data, train_labels, eval_labels = train_test_split(
        not_test_data,
        not_test_labels,
        test_size=150,
        shuffle=True,
        random_state=719,
    )

    start_timestamp = datetime.now().strftime("%Y-%m-%d-%H.%M")
    best_model_filename = os.path.join(
        args.output_directory,
        f"{start_timestamp}-{args.dataset_prefix}-{args.dataset_suffix}.tf",
    )
    checkpoint = ModelCheckpoint(
        best_model_filename,
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        save_freq="epoch",
    )
    history = model.fit(
        train_data,
        train_labels,
        validation_data=(eval_data, eval_labels),
        epochs=args.epochs,
        batch_size=32,
        callbacks=[checkpoint],
    )
    output_training_history(
        history,
        args.dataset_prefix,
        args.dataset_suffix,
        args.output_directory,
    )
    best_model = load_model(best_model_filename)
    loss, accuracy = best_model.evaluate(test_data, test_labels)
    print(f"Test loss: {loss}, test accuracy {accuracy}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=200, type=int)
    parser.add_argument("--output-directory", default="results")
    parser.add_argument("--dataset-directory", default="data/prepared")
    parser.add_argument("--dataset-prefix", default="gztan-dataset")
    parser.add_argument("--dataset-suffix", default="2023-11-17-22.00")

    main(parser.parse_args())
