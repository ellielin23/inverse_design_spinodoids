# utils/transform_S_C_pair.py

import numpy as np

def get_Qs():
    return np.array([
        [[1,0,0 ],
         [0,1,0 ],
         [0,0,1.]],
        [[0,1,0 ],
         [0,0,1 ],
         [1,0,0.]],
        [[0,0,1 ],
         [1,0,0 ],
         [0,1,0.]],
        [[0,0,1 ],
         [0,1,0 ],
         [1,0,0.]],
        [[1,0,0 ],
         [0,0,1 ],
         [0,1,0.]],
        [[0,1,0 ],
         [1,0,0 ],
         [0,0,1.]]
         ])

def get_permuted_S_C_pairs(S, C):
    thetas = S[:3]
    N_nonzero_thetas = np.count_nonzero(thetas)
    Qs = get_Qs()
    if N_nonzero_thetas == 1: Qs = Qs[:3]

    permuted_S_C_pairs = []
    for Q in Qs:
       thetas_permuted = np.einsum('ik,...k->...i', Q, thetas)
       S_permuted      = np.concatenate([thetas_permuted, S[-1:]],-1)
       C_permuted      = np.einsum('im,jn,ko,lp,...mnop->...ijkl', Q,Q,Q,Q,C)
       permuted_S_C_pairs.append([S_permuted, C_permuted])
    return permuted_S_C_pairs
