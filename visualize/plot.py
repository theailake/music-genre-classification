import os

import librosa
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.callbacks import History


def plot_spectrogram(
    mel_spectrogram: np.ndarray, sampling_rate: int = 22500
) -> None:
    """Plots the spectrogram to an image for debug purposes
    Args:
        mel_spectrogram: the mel spectrogram to plot
        sampling_rate: the sampling rate of the audio
    """
    fig, ax = plt.subplots()
    S_dB = librosa.power_to_db(mel_spectrogram, ref=np.max)
    img = librosa.display.specshow(
        S_dB,
        x_axis="time",
        y_axis="mel",
        sr=sampling_rate,
        fmax=8000,
        ax=ax,
    )
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set(title="Mel-frequency spectrogram")


def output_training_history(
    history: History,
    dataset_prefix: str,
    dataset_suffix: str,
    output_directory: str,
    do_show: bool = False,
) -> None:
    """Plots and saves training history of a network
    Args:
        history: the history of the network
        dataset_prefix: the prefix of the prepared dataset
        dataset_suffix: the suffix of the prepared dataset
        output_directory: the directory to write the images to
        do_show: if true, then the plots are shown in a GUI
    """
    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.title("Model Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["train", "eval"], loc="upper left")
    if do_show:
        plt.show()
    plt.savefig(
        os.path.join(
            output_directory, f"accuracy-{dataset_prefix}-{dataset_suffix}.png"
        )
    )
    plt.clf()
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("Model Loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(["train", "eval"], loc="upper left")
    if do_show:
        plt.show()
    plt.savefig(
        os.path.join(
            output_directory, f"loss-{dataset_prefix}-{dataset_suffix}.png"
        )
    )
