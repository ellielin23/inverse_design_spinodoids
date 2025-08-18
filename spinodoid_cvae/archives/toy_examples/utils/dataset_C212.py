# utils/dataset.py

import torch
from torch.utils.data import Dataset
from utils.load_data import load_dataset

class SpinodoidC212Dataset(Dataset):
    """
    PyTorch Dataset for loading structure parameters S ∈ ℝ⁴
    and target material property C212 ∈ ℝ¹.
    """

    def __init__(self, path_csv):
        """
        Args:
            path_csv (str): Path to the CSV file containing the dataset.
        """
        P_full, S = load_dataset(path_csv)
        self.S = S                            # shape: (N, 4)
        self.P = P_full[:, 6:7]               # shape: (N, 1) ← only C212

    def __len__(self):
        return len(self.S)

    def __getitem__(self, idx):
        return self.P[idx], self.S[idx]