"""Script used to prepare the GZTAN dataset"""

import argparse
import glob
import os
from datetime import datetime
from typing import Dict

import librosa
import numpy as np
from audioread.exceptions import NoBackendError
from soundfile import LibsndfileError

from classify.dataset import Dataset

DATA_EXT = ".wav"


def create_directory(directory: str) -> None:
    """Creates the required directory and subdirectories as needed
    Args:
        directory: The relative or absolute path of the required directory
    """
    if not os.path.exists(directory):
        os.makedirs(directory)


def calculate_mel_spectrogram(
    filename: str, duration_in_sec: int = 29
) -> np.ndarray:
    """Loads the first 29 seconds of a .wav file and
    converts to a Mel Spectrogram
    Args:
        filename: .wav file name
        duration_in_sec: the duration of the file to load.
            This is primarily used as a mechnanism to keep
            all data the same length
    Returns:
        A mel spectrogram where the first dimension is 128
    """
    audio, sampling_rate = librosa.load(filename, duration=duration_in_sec)
    mel_spectrogram = librosa.feature.melspectrogram(
        y=audio, sr=sampling_rate, n_mels=128, n_fft=1024, hop_length=512
    )
    return mel_spectrogram


def load_dataset(data_directory: str) -> Dataset:
    """Loads a dataset from given data_directory. The expected
        format is that each subdirectory is a label and within the
        subdirectory there are the .wav files which correspond to
        the label
    Args:
        data_directory: the directory of the data files
    Returns:
        A dataset object that contains the filenames, mel spectrograms,
        and labels
    """
    genres = next(os.walk(data_directory))[1]
    filenames = []
    labels = []
    genre_num = 0
    logging_num = 0
    spectrograms = []
    for genre in genres:
        files_in_dir = glob.glob(
            os.path.join(data_directory, genre, f"*{DATA_EXT}")
        )
        good_files = []
        for file in files_in_dir:
            print(f"Processing file {logging_num}")
            logging_num += 1
            try:
                spectrogram = calculate_mel_spectrogram(file)
                good_files.append(file)
                spectrograms.append(spectrogram)
                assert spectrogram.shape == (128, 1249)
            except (LibsndfileError, NoBackendError) as e:
                """Note: there was some issues with 2 files,
                one was corrupted and another was another size"""
                print(f"for file {file} an error occured: {e}")
        filenames.extend(good_files)
        labels.extend([genre_num] * len(good_files))
        genre_num += 1
    dataset = Dataset(
        filenames=filenames, spectrograms=spectrograms, labels=labels
    )
    return dataset


def save_dataset(
    output_directory: str, file_prefix: str, dataset: Dataset
) -> None:
    """Saves the dataset as npz files
    Args:
        output_directory: the directory where to save the npz files
        file_prefix: the prefix for the npz files
        dataset: the dataset containing all the information needed
    Returns:
        A mel spectrogram where the first dimension is 128
    """
    creation_timestamp = datetime.now().strftime("%Y-%m-%d-%H.%M")
    np.savez(
        os.path.join(
            output_directory,
            f"{file_prefix}-filenames-{creation_timestamp}.npz",
        ),
        dataset.filenames,
    )

    np.savez(
        os.path.join(
            output_directory,
            f"{file_prefix}-spectrograms-{creation_timestamp}.npz",
        ),
        dataset.spectrograms,
    )
    np.savez(
        os.path.join(
            output_directory, f"{file_prefix}-labels-{creation_timestamp}.npz"
        ),
        dataset.labels,
    )


def main(args: Dict) -> None:
    """Main entry point for the script.  Prepares the dataset
    Args:
        args: arguments passed to argparse
    """
    create_directory(args.output_directory)
    dataset = load_dataset(args.dataset_directory)
    save_dataset(args.output_directory, args.output_file_name, dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-directory",
        help="Parent directory for all raw data",
        default="data/genres_original",
    )
    parser.add_argument(
        "--output-file-name",
        help="Name for the output file without file extention",
        default="gztan-dataset",
    )
    parser.add_argument(
        "--output-directory",
        help="Name for the output folder for the prepared datasets",
        default="data/prepared",
    )
    main(parser.parse_args())
