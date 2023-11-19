from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class Dataset:
    """A simple data class to keep track of dataset data
    Attributes:
        filenames: a list of the names of the raw files
        spectrograms: a list of the mel-spectrograms
        labels: a list of the class labels
    """

    filenames: List[str]
    spectrograms: List[np.ndarray]
    labels: List[int]
