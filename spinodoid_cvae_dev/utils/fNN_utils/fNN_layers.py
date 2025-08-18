# utils/fNN_utils/fNN_layers.py

# === max's original code ===

"""Copyright 2024-2025 Max Rosenkranz (Dresden University of Technology)
Licensed under the MIT License. See LICENSE file in the project root for full license text or visit
https://mit-license.org/

Includes the layers the surrogate model is composed of.
"""

import tensorflow as tf
from functools import partial
from itertools import permutations, product
from utils.fNN_utils.mathops import dyad

@tf.keras.utils.register_keras_serializable()
class PermutationEquivariantLayer(tf.keras.layers.Layer):
    """Implementation of Ravanbakhsh et al. (2017):
    'Equivariance through parameter sharing'.
    """
    def __init__(self, dim, rank_input, rank_output, neurons, activation,
                 N_non_pe_inputs=0, N_non_pe_outputs=0, sym_input=None, sym_output=None,
                 no_orb_for_in_inds=[], no_orb_for_out_inds=[],
                 **kwargs):
        super().__init__(**kwargs)
        self.dim                 = dim
        self.rnk_out             = rank_output
        self.neurons_output      = neurons
        self.activation          = activation
        self.N_non_pe_inputs     = N_non_pe_inputs
        self.N_non_pe_outputs    = N_non_pe_outputs
        self.sym_input           = sym_input
        self.sym_output          = sym_output
        self.no_orb_for_in_inds  = no_orb_for_in_inds
        self.no_orb_for_out_inds = no_orb_for_out_inds
        self.activation_fn       = tf.keras.layers.Activation(self.activation)
        self.initializer         = tf.keras.initializers.GlorotUniform(5)
        self.inds_perm           = range(self.dim)
        self.inds_in             = range(self.dim+self.N_non_pe_inputs)
        self.inds_out            = range(self.dim+self.N_non_pe_outputs)
    
    def build(self, input_shape):
        self.neurons_input  = input_shape[1] # 0 is batch dimension
        self.rnk_in         = len(input_shape) - 2
        orbits              = self.get_orbits(self.rnk_in, self.rnk_out, self.sym_input, self.sym_output, self.no_orb_for_in_inds, self.no_orb_for_out_inds)
        orbits_out          = self.get_orbits(0, self.rnk_out, None, self.sym_output, [], self.no_orb_for_out_inds)
        self.orbit_matrices = self.get_matrices_from_orbits(orbits,  [*(self.rnk_out-1)*[self.dim]+[self.dim+self.N_non_pe_outputs],
                                                                      *(self.rnk_in-1)*[self.dim]+[self.dim+self.N_non_pe_inputs]])
        self.orbit_matrices_out = self.get_matrices_from_orbits(orbits_out,  [*(self.rnk_out-1)*[self.dim]+[self.dim+self.N_non_pe_outputs]])
        self.orbit_weights    = self.add_weight(shape=(self.neurons_output, self.neurons_input, len(orbits)),
                                                 initializer=self.initializer,
                                                 dtype=tf.float32,
                                                 trainable=True)
        self.orbit_biases = self.add_weight(shape=(self.neurons_output, len(orbits_out)),
                                                 initializer=self.initializer,
                                                 dtype=tf.float32,
                                                 trainable=True)
        
    def get_orbits(self, rnk_N, rnk_M, sym_input, sym_output,
                   no_orbit_for_in_inds=[], no_orbit_for_out_inds=[]):
        G       = frozenset([lambda i, perm=perm: (perm[i[0]] if i[0] in self.inds_perm else i[0],) for perm in permutations(self.inds_perm)])
        N       = frozenset(product(self.inds_in, repeat=rnk_N))
        M       = frozenset(product(self.inds_out, repeat=rnk_M))
        gamma_N = lambda g,n: tuple(g([n[i]])[0] for i in range(rnk_N))
        gamma_M = lambda g,m: tuple(g([m[i]])[0] for i in range(rnk_M))

        G_NM       = frozenset([(partial(gamma_N, g=g), partial(gamma_M, g=g)) for g in G])
        orbit_nm   = lambda n,m: frozenset([(g_N(n=n), g_M(m=m)) for g_N, g_M in G_NM])
        orbits     = frozenset([orbit_nm(n,m) for n in N for m in M])

        orbits_sym = frozenset([self.symmetrize_orbit(orbit, sym_input, sym_output) for orbit in orbits])
        no_orbit_for_inds = [(tuple(n),m) for m in M for n in no_orbit_for_in_inds] + [(n,tuple(m)) for n in N for m in no_orbit_for_out_inds]
        orbits_red = frozenset([orbit for orbit in orbits_sym if not any(ind in orbit for ind in no_orbit_for_inds)])
        return orbits_red
    
    def symmetrize_orbit(self, orbit, sym_input, sym_output):
        orbit_sym = []
        for input_inds, output_inds in orbit:
            input_inds_sym  = self.get_idcs_sym(input_inds,  sym_input)
            output_inds_sym = self.get_idcs_sym(output_inds, sym_output)
            orbit_sym.extend([(input_ind, output_ind) for input_ind in input_inds_sym for output_ind in output_inds_sym])
        return frozenset(orbit_sym)
    
    def get_idcs_sym(self, inds, sym_type):
        if sym_type == None:
            return [inds]
        elif sym_type == 'minor':
            assert len(inds) == 4
            perms = [[0,1,2,3],[0,1,3,2],[1,0,2,3],[1,0,3,2],]
            return [tuple([inds[i] for i in perm]) for perm in perms]
        elif sym_type == 'major':
            assert len(inds) == 4
            perms = [[0,1,2,3],[2,3,0,1]]
            return [tuple([inds[i] for i in perm]) for perm in perms]
        elif sym_type == 'minor+major':
            assert len(inds) == 4
            perms = [[0,1,2,3],[0,1,3,2],[1,0,2,3],[1,0,3,2],
                     [2,3,0,1],[3,2,0,1],[2,3,1,0],[3,2,1,0]]
            return [tuple([inds[i] for i in perm]) for perm in perms]
        elif sym_type == 'full':
            return list(permutations(inds))
    
    def get_matrices_from_orbits(self, orbits, shape):
        orbit_matrices = []
        for orbit in orbits:
            inds = sorted([(*m,*n) for n,m in orbit])
            W_i_sparse  = tf.SparseTensor(indices=inds, values=tf.ones((len(inds))), dense_shape=shape)
            orbit_matrices.append(tf.sparse.to_dense(W_i_sparse))
        return tf.stack(orbit_matrices, -1)
    
    def call(self, input):
        """y = A(W*x), where W is the weighted sum of all
        orbit matrices for the given input/output ranks.
        """
        inds_N = 'ijkl'[:self.rnk_in]
        inds_M = 'mnop'[:self.rnk_out]
        stacked_orbit_matrices = tf.stack(self.neurons_output*[tf.stack(self.neurons_input*[self.orbit_matrices],0)],0)
        stacked_orbit_matrices_out = tf.stack(self.neurons_output*[self.orbit_matrices_out],0)
        # a  bit of reshaping is necesary here to assamble the weight matrix and bias. The weight matrix has the shape
        # neurons_out x neurons_in x {idcs_out} x {idcs_in}
        W_reshape = tf.reduce_sum((tf.einsum('OI...k->...OIk', stacked_orbit_matrices) * self.orbit_weights), -1)
        B_reshape = tf.reduce_sum((tf.einsum('O...k->...Ok', stacked_orbit_matrices_out) * self.orbit_biases), -1)
        W = tf.einsum('...OI->OI...', W_reshape)
        B = tf.einsum('...O->O...', B_reshape)
        return self.activation_fn(tf.einsum(f'OI{inds_M}{inds_N},...I{inds_N}->...O{inds_M}', W,input) + B)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "dim": self.dim,
            "rank_input": self.rnk_in,
            "rank_output": self.rnk_out,
            "neurons": self.neurons_output,
            "activation": self.activation,
            "N_non_pe_inputs": self.N_non_pe_inputs,
            "N_non_pe_outputs": self.N_non_pe_outputs,
            "sym_input": self.sym_input,
            "sym_output": self.sym_output,
            "no_orb_for_in_inds" : self.no_orb_for_in_inds,
            "no_orb_for_out_inds": self.no_orb_for_out_inds,
            })
        return config

@tf.keras.utils.register_keras_serializable()
class DoubleContractionLayer(tf.keras.layers.Layer):
    """Takes fourth order tensor T and return T:T, i.e., T double contracted with itself.
    Thus, the output is positive semi-definite.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, input):
        return tf.einsum("...ijkl,...klmn->...ijmn", input, input)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            })
        return config

@tf.keras.utils.register_keras_serializable()
class EnforceIsotropyLayer(tf.keras.layers.Layer):
    """Takes structure parameters S and fourth order tensor t, decomposes t and
    filters out anisotropic part depending on S.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        I2 = tf.eye(3)
        I4 = 1/2*(tf.einsum('ik,jl->ijkl',I2,I2) + tf.einsum('il,jk->ijkl',I2,I2))
        P1 = 1/3*dyad(I2,I2)
        P2 = I4 - P1
        self.P_iso = dyad(P1,P1) + 1/5*dyad(P2,P2)
        self.rho_max = 1.
        self.theta_max = 90.
        self.m = 1

    def call(self, input):
        S, C = input
        kappa = self.kappa(S)
        C_iso = tf.einsum("ijklmnop,...mnop->...ijkl", self.P_iso, C)
        C_aniso = C-C_iso
        return C_iso + tf.reshape(kappa, [-1,1,1,1,1,1])*C_aniso
    
    def kappa(self, S):
        theta_1, theta_2, theta_3, rho = S[...,0], S[...,1], S[...,2], S[...,3]
        return self.m*(1-theta_1/self.theta_max)*(1-theta_2/self.theta_max)*(1-theta_3/self.theta_max)*(1-rho/self.rho_max)

    def get_config(self):
        config = super().get_config()
        config.update({
            })
        return config

@tf.keras.utils.register_keras_serializable()
class NormalizationLayer(tf.keras.layers.Layer):
    """Normalizes inputs or outputs to the range [-1,1].
    """
    def __init__(self, direction, **kwargs):
        super().__init__(**kwargs)
        assert (direction=='in' or direction=='out')
        self.direction = direction
    
    def build(self, input_shape):
        input_shape_no_nones = [d for d in input_shape if d is not None]
        self.m               = self.add_weight(shape=input_shape_no_nones, trainable=False, dtype=tf.float32, name='m')
        self.s               = self.add_weight(shape=input_shape_no_nones, trainable=False, dtype=tf.float32, name='s')

    def adapt(self, data):
        m = tf.convert_to_tensor([1/2*(tf.reduce_max(data[...,i])+tf.reduce_min(data[...,i])) for i in range(data.shape[-1])])
        s = tf.convert_to_tensor([1/2*(tf.reduce_max(data[...,i])-tf.reduce_min(data[...,i])) for i in range(data.shape[-1])])
        self.m.assign(m)
        self.s.assign(s)

    def call(self, input):
        if self.direction == 'in':
            return (input - self.m) / self.s
        else:
            return  input * self.s + self.m

    def get_config(self):
        config = super().get_config()
        config.update({
            "direction": self.direction
            })
        return config
