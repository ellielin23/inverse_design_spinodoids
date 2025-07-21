import numpy as np
import torch
from collections import defaultdict
def full_C_from_C_flat_21(C_flat_21):
    """
    Converts flattened 21D Mandel representation to full 3x3x3x3 tensors.

    Args:
        C_flat_21 (np.ndarray): shape (N, 21), each row is a flattened 6x6 upper triangle

    Returns:
        C_tensor4 (np.ndarray): shape (N, 3, 3, 3, 3), full rank-4 elasticity tensor
    """
    # reconstruct symmetric 6x6 matrix from 21 elements
    C_flat_36 = np.concatenate([
        C_flat_21[:, 0:1],   # (0,0)
        C_flat_21[:, 1:2],   # (0,1)
        C_flat_21[:, 2:3],   # (0,2)
        C_flat_21[:, 3:4],   # (0,3)
        C_flat_21[:, 4:5],   # (0,4)
        C_flat_21[:, 5:6],   # (0,5)
        C_flat_21[:, 1:2],   # (1,0)
        C_flat_21[:, 6:7],   # (1,1)
        C_flat_21[:, 7:8],   # (1,2)
        C_flat_21[:, 8:9],   # (1,3)
        C_flat_21[:, 9:10],  # (1,4)
        C_flat_21[:,10:11],  # (1,5)
        C_flat_21[:, 2:3],   # (2,0)
        C_flat_21[:, 7:8],   # (2,1)
        C_flat_21[:,11:12],  # (2,2)
        C_flat_21[:,12:13],  # (2,3)
        C_flat_21[:,13:14],  # (2,4)
        C_flat_21[:,14:15],  # (2,5)
        C_flat_21[:, 3:4],   # (3,0)
        C_flat_21[:, 8:9],   # (3,1)
        C_flat_21[:,12:13],  # (3,2)
        C_flat_21[:,15:16],  # (3,3)
        C_flat_21[:,16:17],  # (3,4)
        C_flat_21[:,17:18],  # (3,5)
        C_flat_21[:, 4:5],   # (4,0)
        C_flat_21[:, 9:10],  # (4,1)
        C_flat_21[:,13:14],  # (4,2)
        C_flat_21[:,16:17],  # (4,3)
        C_flat_21[:,18:19],  # (4,4)
        C_flat_21[:,19:20],  # (4,5)
        C_flat_21[:, 5:6],   # (5,0)
        C_flat_21[:,10:11],  # (5,1)
        C_flat_21[:,14:15],  # (5,2)
        C_flat_21[:,17:18],  # (5,3)
        C_flat_21[:,19:20],  # (5,4)
        C_flat_21[:,20:21],  # (5,5)
    ], axis=-1)

    # reshape to symmetric 6x6 matrix form for each sample
    C_km = C_flat_36.reshape(-1, 6, 6)

    # convert each 6x6 matrix to full 3x3x3x3 tensor
    C_tensor4 = np.stack([mandel_to_tensor4_numpy(C) for C in C_km], axis=0)
    return C_tensor4  # shape: (N, 3, 3, 3, 3)


def mandel_to_tensor4_numpy(C):
    """
    Convert a 6x6 matrix in Mandel notation to a 3x3x3x3 elasticity tensor.
    
    Args:
        C (np.ndarray): shape (6, 6)

    Returns:
        T (np.ndarray): shape (3, 3, 3, 3)
    """
    voigt_to_tensor = {
        0: (0, 0),
        1: (1, 1),
        2: (2, 2),
        3: (1, 2),
        4: (0, 2),
        5: (0, 1),
    }

    T = np.zeros((3, 3, 3, 3))
    for i in range(6):
        for j in range(6):
            a, b = voigt_to_tensor[i]
            c, d = voigt_to_tensor[j]
            T[a, b, c, d] = C[i, j]
            T[b, a, c, d] = C[i, j]
            T[a, b, d, c] = C[i, j]
            T[b, a, d, c] = C[i, j]
    return T


def extract_target_properties(C_tensor):
    """
    Extract 9 components: 1111, 1122, 1133, 2222, 2233, 3333, 1212, 1313, 2323

    Args:
        C_tensor (np.ndarray): shape (N, 3, 3, 3, 3)

    Returns:
        (N, 9) array where each row is:
        [C_1111, C_1122, C_1133, C_2222, C_2233, C_3333, C_1212, C_1313, C_2323]
    """
    idxs = [
        (0,0,0,0), (0,0,1,1), (0,0,2,2),
        (1,1,1,1), (1,1,2,2), (2,2,2,2),
        (0,1,0,1), (0,2,0,2), (1,2,1,2),
    ]
    return np.stack([
        np.array([C[i,j,k,l] for (i,j,k,l) in idxs])
        for C in C_tensor
    ])


def load_dataset(path_csv):
    """
    Loads a spinodoid dataset CSV, extracts structure parameters and elastic properties.

    Args:
        path_csv (str): Path to CSV with 25 columns (ID, S1-S4, C_flat_21)

    Returns:
        P ∈ ℝ⁹ (torch.Tensor): shape (N, 9) — 9 target elastic components
        S ∈ ℝ⁴ (torch.Tensor): shape (N, 4) — structure parameters
    """
    data = np.genfromtxt(path_csv, delimiter=',')[:, 1:]  # skip ID column
    S = np.concatenate([data[:, 1:4], data[:, 0:1]], axis=-1)  # S: [2:5] + [1]
    C_flat_21 = data[:, 4:]

    C_tensor = full_C_from_C_flat_21(C_flat_21)
    P = extract_target_properties(C_tensor)

    # convert to pytorch tensors
    S = torch.tensor(S, dtype=torch.float32)
    P = torch.tensor(P, dtype=torch.float32)
    return P, S


def load_distributional_dataset(path_csv):
    """
    Loads a CSV where multiple P samples are given per S, grouped by identical S.

    Returns:
        dict: {S_key (tuple): P_samples (np.ndarray of shape [M, 9])}
    """
    data = np.genfromtxt(path_csv, delimiter=',')[:, 1:]  # skip ID

    # reorder structure: S = [S2, S3, S4, S1]
    S_all = np.concatenate([data[:, 1:4], data[:, 0:1]], axis=-1)  # shape (N, 4)

    # convert C_flat_21 to elasticity tensors
    C_flat_21 = data[:, 4:]
    C_tensor = full_C_from_C_flat_21(C_flat_21)
    P_all = extract_target_properties(C_tensor)  # shape (N, 9)

    # group all Ps under them
    empirical_data = defaultdict(list)
    for S_vec, P_vec in zip(S_all, P_all):
        S_key = tuple(S_vec)
        empirical_data[S_key].append(P_vec)

    # stack lists into arrays
    empirical_data = {k: np.vstack(v) for k, v in empirical_data.items()}
    return empirical_data
