"""Copyright 2024-2025 Max Rosenkranz (Dresden University of Technology)
Licensed under the MIT License. See LICENSE file in the project root for full license text or visit
https://mit-license.org/
"""

import tensorflow as tf
from itertools import permutations

I  = tf.eye(3)
e1 = tf.constant([1, 0, 0], dtype=tf.float32)
e2 = tf.constant([0, 1, 0], dtype=tf.float32)
e3 = tf.constant([0, 0, 1], dtype=tf.float32)
e = tf.constant([[[ 0, 0, 0],
                  [ 0, 0, 1],
                  [ 0,-1, 0]],
                 [[ 0, 0,-1],
                  [ 0, 0, 0],
                  [ 1, 0, 0]],
                 [[ 0, 1, 0],
                  [-1, 0, 0],
                  [ 0, 0, 0]]], dtype=tf.float32)

def dyad(*tensors):
    def dyad_2tensors(t1, t2):
        return tf.tensordot(t1, t2, axes=0)
    res = tensors[0]
    for tensor in tensors[1:]:
        res = dyad_2tensors(res, tensor)
    return res

def mandel_to_tensor2(T_M):
    i = index_map
    T11 = T_M[...,i['11']]
    T22 = T_M[...,i['22']]
    T33 = T_M[...,i['33']]
    T12 = T_M[...,i['12']]/(2**0.5)
    T13 = T_M[...,i['13']]/(2**0.5)
    T23 = T_M[...,i['23']]/(2**0.5)
    return tf.einsum('ij...->...ij',(tf.convert_to_tensor(
                            [[T11, T12, T13],
                             [T12, T22, T23],
                             [T13, T23, T33]],
                            )))

def mandel_to_tensor4(T_M):
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
    return tf.einsum('ijkl...->...ijkl', tf.convert_to_tensor(
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

def tensor2_to_mandel(T):
    i   = index_map
    m1 = T[...,*i['1']]
    m2 = T[...,*i['2']]
    m3 = T[...,*i['3']]
    m4 = T[...,*i['4']]*2**(0.5)
    m5 = T[...,*i['5']]*2**(0.5)
    m6 = T[...,*i['6']]*2**(0.5)
    return tf.einsum('i...->...i', tf.convert_to_tensor([m1,m2,m3,m4,m5,m6]))

def tensor4_to_mandel(T):
    i   = index_map
    k11 = T[...,*i['1'],*i['1']]
    k12 = T[...,*i['1'],*i['2']]
    k13 = T[...,*i['1'],*i['3']]
    k14 = T[...,*i['1'],*i['4']]*2**(0.5)
    k15 = T[...,*i['1'],*i['5']]*2**(0.5)
    k16 = T[...,*i['1'],*i['6']]*2**(0.5)
    k22 = T[...,*i['2'],*i['2']]
    k23 = T[...,*i['2'],*i['3']]
    k24 = T[...,*i['2'],*i['4']]*2**(0.5)
    k25 = T[...,*i['2'],*i['5']]*2**(0.5)
    k26 = T[...,*i['2'],*i['6']]*2**(0.5)
    k33 = T[...,*i['3'],*i['3']]
    k34 = T[...,*i['3'],*i['4']]*2**(0.5)
    k35 = T[...,*i['3'],*i['5']]*2**(0.5)
    k36 = T[...,*i['3'],*i['6']]*2**(0.5)
    k44 = T[...,*i['4'],*i['4']]*2
    k45 = T[...,*i['4'],*i['5']]*2
    k46 = T[...,*i['4'],*i['6']]*2
    k55 = T[...,*i['5'],*i['5']]*2
    k56 = T[...,*i['5'],*i['6']]*2
    k66 = T[...,*i['6'],*i['6']]*2

    return tf.einsum('ij...->...ij', tf.convert_to_tensor(
        [[k11, k12, k13, k14, k15, k16],
         [k12, k22, k23, k24, k25, k26],
         [k13, k23, k33, k34, k35, k36],
         [k14, k24, k34, k44, k45, k46],
         [k15, k25, k35, k45, k55, k56],
         [k16, k26, k36, k46, k56, k66]]))

def euclidian_norm(T):
    rnk = len(tf.shape(T))
    reduction_idcs = [i for i in range(rnk)]
    T_T = tf.tensordot(T,T, [reduction_idcs, reduction_idcs])
    return tf.sqrt(T_T)

def euclidian_distance(T1, T2, rnk=None):
    return euclidian_norm(T1-T2)

def log4(T):
    """Only for rank(T)=4."""
    T_mandel = tensor4_to_mandel(T)
    eigvals, eigvecs = tf.linalg.eigh(T_mandel)
    log_eigvals = tf.math.log(eigvals)
    log_T_mandel = apply_rotation(tf.linalg.diag(log_eigvals), eigvecs)
    return mandel_to_tensor4(log_T_mandel)

def root_of_tensor4(T):
    T_mandel = tensor4_to_mandel(T)
    eigvals, eigvecs = tf.linalg.eigh(T_mandel)
    T_sqrt_mandel = tf.einsum('...ik,...kl,...lj->...ij', eigvecs, tf.linalg.diag(eigvals), tf.einsum("...ij->...ji", eigvecs))
    return mandel_to_tensor4(T_sqrt_mandel)

def apply_rotation(T, Q):
    rnk = len(tf.shape(T))
    QxQ = dyad(*rnk*[Q])
    reduction_idcs_QxQ = [(2*i)+1 for i in range(rnk)]
    reduction_idcs_T   = [i       for i in range(rnk)]
    return tf.tensordot(QxQ, T, [reduction_idcs_QxQ, reduction_idcs_T])

def is_positive_semidefinite(matrix):
    eigvals = tf.linalg.eigvalsh(matrix)
    return tf.reduce_all(eigvals>=0)

def get_symmetric_indices(ind_tuple, sym_type):
    if sym_type == None:
        return [ind_tuple]
    elif sym_type == 'minor':
        assert len(ind_tuple) == 4
        raise NotImplementedError
    elif sym_type == 'major':
        assert len(ind_tuple) == 4
        raise NotImplementedError
    elif sym_type == 'minor+major':
        assert len(ind_tuple) == 4
        perms = [[0,1,2,3],[0,1,3,2],[1,0,2,3],[1,0,3,2],
                 [2,3,0,1],[3,2,0,1],[2,3,1,0],[3,2,1,0]]
        return [tuple([ind_tuple[i] for i in perm]) for perm in perms]
    elif sym_type == 'full':
        return list(permutations(ind_tuple))

def rodrigues_angles_to_rotation_matrix(rodrigues_angles):
    theta, phi, alpha = rodrigues_angles[0], rodrigues_angles[1], rodrigues_angles[2]
    n = tf.stack([
            tf.cos(theta)*tf.sin(phi), 
            tf.sin(theta)*tf.sin(phi),  
                          tf.cos(phi)
            ], 0)
    return dyad(n,n) + tf.cos(alpha)*(I-dyad(n,n)) + tf.sin(alpha)*(tf.einsum('ipq,p,qj->ij',e,n,I))

def angles_to_rotation_matrix(angles):
    return rodrigues_angles_to_rotation_matrix(angles)

def calc_stiffness_in_direction(C, direction):
    dd          = tf.einsum("i,j->ij", direction, direction)
    dd_mandel   = tensor2_to_mandel(dd)
    C_inv_KM    = tf.linalg.inv(tensor4_to_mandel(C))
    E_dir       = 1 / (tf.einsum("L,L",tf.einsum("K,KL->L", dd_mandel, C_inv_KM), dd_mandel))
    return E_dir

# Index map
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
