# utils/dataset.py

import torch
from torch.utils.data import Dataset
from utils.data_utils.load_data import load_dataset  # returns (P, S, C_tensor)

class SpinodoidDataset(Dataset):
    """
    Minimal dataset: yields (S, P) pairs for training.
    """

    def __init__(self, path_csv):
        P, S, _ = load_dataset(path_csv)  # both are torch.Tensors
        self.S = S
        self.P = P

    def __len__(self):
        return len(self.S)

    def __getitem__(self, idx):
        return self.S[idx], self.P[idx]
