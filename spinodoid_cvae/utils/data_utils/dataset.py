# utils/data_utils/dataset.py

import torch
from torch.utils.data import Dataset
from utils.data_utils.load_data import load_dataset

class SpinodoidDataset(Dataset):
    """
    PyTorch-compatible Dataset class for spinodoid data.

    Each sample is a (P, S) pair:
    - P ∈ ℝ⁹: 9 target elastic tensor components
    - S ∈ ℝ⁴: 4 structure parameters
    """
    
    def __init__(self, path_csv):
        # load full dataset using my preprocessing function
        self.P, self.S = load_dataset(path_csv)

    def __len__(self):
        # return total number of samples
        return len(self.P)

    def __getitem__(self, idx):
        # index into the dataset to return (P, S) pair
        return self.P[idx], self.S[idx]


