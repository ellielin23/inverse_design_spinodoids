# utils/data_utils/load_data.py

import numpy as np
import torch

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


# def mandel_to_tensor4_numpy(C):
#     """
#     Convert a 6x6 matrix in Mandel notation to a 3x3x3x3 elasticity tensor.
    
#     Args:
#         C (np.ndarray): shape (6, 6)

#     Returns:
#         T (np.ndarray): shape (3, 3, 3, 3)
#     """
#     voigt_to_tensor = {
#         0: (0, 0),
#         1: (1, 1),
#         2: (2, 2),
#         3: (0, 1),
#         4: (0, 2),
#         5: (1, 2),
#     }

#     T = np.zeros((3, 3, 3, 3))
#     for i in range(6):
#         for j in range(6):
#             a, b = voigt_to_tensor[i]
#             c, d = voigt_to_tensor[j]
#             T[a, b, c, d] = C[i, j]
#             T[b, a, c, d] = C[i, j]
#             T[a, b, d, c] = C[i, j]
#             T[b, a, d, c] = C[i, j]
#     return T

def mandel_to_tensor4_numpy(T_M):
    i = index_map
    T1111 = T_M[...,i['11'],i['11']]
    T1122 = T_M[...,i['11'],i['22']]
    T1133 = T_M[...,i['11'],i['33']]
    T1123 = T_M[...,i['11'],i['23']]/(2**0.5)
    T1113 = T_M[...,i['11'],i['13']]/(2**0.5)
    T1112 = T_M[...,i['11'],i['12']]/(2**0.5)
    T2222 = T_M[...,i['22'],i['22']]
    T2233 = T_M[...,i['22'],i['33']]
    T2223 = T_M[...,i['22'],i['23']]/(2**0.5)
    T2213 = T_M[...,i['22'],i['13']]/(2**0.5)
    T2212 = T_M[...,i['22'],i['12']]/(2**0.5)
    T3333 = T_M[...,i['33'],i['33']]
    T3323 = T_M[...,i['33'],i['23']]/(2**0.5)
    T3313 = T_M[...,i['33'],i['13']]/(2**0.5)
    T3312 = T_M[...,i['33'],i['12']]/(2**0.5)
    T2323 = T_M[...,i['23'],i['23']]/2
    T2313 = T_M[...,i['23'],i['13']]/2
    T2312 = T_M[...,i['23'],i['12']]/2
    T1313 = T_M[...,i['13'],i['13']]/2
    T1312 = T_M[...,i['13'],i['12']]/2
    T1212 = T_M[...,i['12'],i['12']]/2
    T1211 = T1112
    T1213 = T1312
    T1222 = T2212
    T1223 = T2312
    T1233 = T3312
    T1311 = T1113
    T1322 = T2213
    T1323 = T2313
    T1333 = T3313
    T2111 = T1112
    T2112 = T1212
    T2113 = T1312
    T2122 = T2212
    T2123 = T2312
    T2133 = T3312
    T2211 = T1122
    T2311 = T1123
    T2322 = T2223
    T2333 = T3323
    T3111 = T1113
    T3112 = T1312
    T3113 = T1313
    T3122 = T2213
    T3123 = T2313
    T3133 = T3313
    T3211 = T1123
    T3212 = T2312
    T3213 = T2313
    T3222 = T2223
    T3223 = T2323
    T3233 = T3323
    T3311 = T1133
    T3322 = T2233
    return np.einsum('ijkl...->...ijkl', np.array(
                            [[[[T1111, T1112, T1113],
                               [T1112, T1122, T1123],
                               [T1113, T1123, T1133]],
                              [[T1211, T1212, T1213],
                               [T1212, T1222, T1223],
                               [T1213, T1223, T1233]],
                              [[T1311, T1312, T1313],
                               [T1312, T1322, T1323],
                               [T1313, T1323, T1333]]],

                             [[[T2111, T2112, T2113],
                               [T2112, T2122, T2123],
                               [T2113, T2123, T2133]],
                              [[T2211, T2212, T2213],
                               [T2212, T2222, T2223],
                               [T2213, T2223, T2233]],
                              [[T2311, T2312, T2313],
                               [T2312, T2322, T2323],
                               [T2313, T2323, T2333]]],

                             [[[T3111, T3112, T3113],
                               [T3112, T3122, T3123],
                               [T3113, T3123, T3133]],
                              [[T3211, T3212, T3213],
                               [T3212, T3222, T3223],
                               [T3213, T3223, T3233]],
                              [[T3311, T3312, T3313],
                               [T3312, T3322, T3323],
                               [T3313, T3323, T3333]]]],
                            ))


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
    Loads a spinodoid dataset CSV, extracts structure parameters, elastic properties, and full C tensors.

    Args:
        path_csv (str): Path to CSV with 25 columns (ID, S1-S4, C_flat_21)

    Returns:
        P ∈ ℝ⁹ (torch.Tensor): shape (N, 9)
        S ∈ ℝ⁴ (torch.Tensor): shape (N, 4)
        C_tensor ∈ ℝ^{N×3×3×3×3} (np.ndarray): full fourth-order tensors
    """
    data = np.genfromtxt(path_csv, delimiter=',')[:, 1:]  # skip ID column
    S = np.concatenate([data[:, 1:4], data[:, 0:1]], axis=-1)
    C_flat_21 = data[:, 4:]

    C_tensor = full_C_from_C_flat_21(C_flat_21)
    P = extract_target_properties(C_tensor)

    # convert to pytorch tensors
    S = torch.tensor(S, dtype=torch.float32)
    P = torch.tensor(P, dtype=torch.float32)
    return P, S, C_tensor

index_map = {'11': 0,
             '22': 1,
             '33': 2,
             '12': 3,
             '13': 4,
             '23': 5,
             '1' : (0,0),
             '2' : (1,1),
             '3' : (2,2),
             '4' : (0,1),
             '5' : (0,2),
             '6' : (1,2)}