# -*- coding: utf-8 -*-
import numpy as np
from scipy.linalg import svd, LinAlgError

from time import perf_counter

def sign(x):
    return 2 * (x >= 0) - 1

def random_unitaries(shape, strength=1e-2, real=False):
    unitaries = np.random.normal(0, strength, size=shape)
    if not real:
        unitaries = unitaries.astype(np.complex128)\
            + 1j * np.random.normal(0, strength, size=shape)
    unitaries += np.eye(shape[-2], shape[-1])
    unitaries, rs = np.linalg.qr(unitaries)
    unitaries *= sign(np.einsum('...ii->...i', rs))[...,None,:]
    return unitaries

def polar_projection(A, SO=False):
    # these matrices are usually 4x4, so no need for a stable SVD
    X, S, Y = np.linalg.svd(A)
    U = X @ Y
    if SO:
        # flip signs to keep determinants positive
        dets = np.linalg.det(U)
        Y[..., -1, :] *= dets[..., :, None]
        U = X @ Y
    return U

@np.vectorize(signature='(m,n)->(m,p),(p),(p,n)')
def stable_svd(A):
    try:
        U, S, V = svd(A, full_matrices=False, lapack_driver='gesdd')
    except LinAlgError:
        U, S, V = svd(A, full_matrices=False, lapack_driver='gesvd')
    return U, S, V

@np.vectorize(signature='(m,n)->(p)')
def stable_svdvals(A):
    try:
        S = svd(A, compute_uv=False, lapack_driver='gesdd')
    except LinAlgError:
        S = svd(A, compute_uv=False, lapack_driver='gesvd')
    return S

class sequential_circuit:
    
    def __init__(self, L, layers, batchsize=1, Lc=None):
        # system size
        self.L = L
        self.d = 2
        self.layers = layers
        self.batch = batchsize
        # canonical center of sequential circuit
        if Lc is None:
            self.Lc = self.L//2-1
        else:
            self.Lc = Lc
        # parameters for contraction
        self.svd_min = 1e-8
        self.chi_min = 2
        self.chi_max = 256
    
    def left_canonical(self, Bs):
        As = []
        R = np.ones((self.batch, 1, 1))
        for B in Bs:
            _, chil, d, chir = B.shape
            A = np.einsum('iab,ibjc->iajc', R, B)\
                .reshape(self.batch, chil * d, chir)
            Q, R = np.linalg.qr(A)
            As.append(Q.reshape(self.batch, chil, d, chir))
        return As
    
    def right_canonical(self, As):
        Bs = []
        Lt = np.ones((self.batch, 1, 1))
        for A in As[::-1]:
            _, chil, d, chir = A.shape
            A = np.einsum('iajb,icb->iajc', A, Lt)\
                .reshape(self.batch, chil, d * chir)
            Qt, Lt = np.linalg.qr(A.transpose(0,2,1))
            Bs.append(Qt.transpose(0,2,1).reshape(self.batch, chil, d, chir))
        return Bs[::-1]
    
    def apply_gate(self, gate, Lambdal, Bl, Br):
        # reshape gate
        gate = gate.reshape(self.batch, *(self.d,)*4)
        # calculate theta tensor and bring into right shape
        theta = np.einsum('ijklm,ia,ialb,ibmc->iajkc', gate, Lambdal, Bl, Br)
        _, chil, _, _, chir = theta.shape
        theta = theta.reshape(self.batch, chil * self.d, self.d * chir)
        # SVD decomposition and truncation
        U, S, V = stable_svd(theta)
        chi_new = min(S.shape[1],
                      max(self.chi_min, max(np.sum(S > self.svd_min, axis=1))),
                      self.chi_max)
        # new tensors
        Lambdar = S[:, :chi_new] / np.linalg.norm(S, axis=1)[:, None]
        Br = V[:, :chi_new, :].reshape(self.batch, chi_new, self.d, chir)
        Bl = (1/Lambdal)[:, :, None, None]\
            * (U[:, :, :chi_new] * Lambdar[:, None, :])\
            .reshape(self.batch, chil, self.d, chi_new)
        return Bl, Br, Lambdar
    
    def get_gates(self, q_start, q_end, layer):
        raise NotImplementedError(
            'This function should be implemented by the inheriting class.')
    
    def _calculate_upper_environments(self, target_Bs, target_Lambdas):
        upper_env_Bs = []
        upper_env_Lambdas = []
        Bs = target_Bs.copy()
        Lambdas = target_Lambdas.copy()
        # loop through the layers from top to bottom
        for layer in range(self.layers-1, -1, -1):
            # first half sweep from right to left
            gates = self.get_gates(q_start=None, q_end=self.Lc, layer=layer)
            for i, gate in enumerate(gates):
                idx = self.L-1-i
                Bl, Br, Lambdar = self.apply_gate(gate.transpose(0,2,1),
                                                  Lambdas[idx-1],
                                                  Bs[idx-1],
                                                  Bs[idx])
                Bs[idx-1] = Bl
                Bs[idx] = Br
                Lambdas[idx] = Lambdar
            # second half sweep from left to right
            gates = self.get_gates(q_start=0, q_end=self.Lc+1, layer=layer)
            for i, gate in enumerate(gates):
                Bl, Br, Lambdar = self.apply_gate(gate.transpose(0,2,1),
                                                  Lambdas[i],
                                                  Bs[i],
                                                  Bs[i+1])
                Bs[i] = Bl
                Bs[i+1] = Br
                Lambdas[i+1] = Lambdar
            upper_env_Bs.append(Bs.copy())
            upper_env_Lambdas.append(Lambdas.copy())
        # reorder upper_envs before returning
        upper_env_Bs = upper_env_Bs[::-1]
        upper_env_Lambdas = upper_env_Lambdas[::-1]
        return upper_env_Bs, upper_env_Lambdas
    
    def _calculate_right_environments(self, Bs, upper_env):
        right_env = np.ones((self.batch, 1, 1))
        right_envs = [right_env]
        # sweep right to left
        for B, Benv in zip(Bs[:self.Lc+1:-1], upper_env[:self.Lc+1:-1]):
            right_env = np.einsum('iajb,ibc->iajc', Benv, right_env)
            right_env = np.einsum('iajb,icjb->iac', right_env, B)
            right_envs.append(right_env)
        # reorder right_envs before returning
        right_envs = right_envs[::-1]
        return right_envs
    
    def _calculate_left_environment(self, Bs, upper_env):
        left_env = np.ones((self.batch, 1, 1))
        # sweep left to right
        for B, Benv  in zip(Bs[:self.Lc], upper_env[:self.Lc]):
            left_env = np.einsum('iab,iajc->ibjc', left_env, Benv)
            left_env = np.einsum('iajb,iajc->ibc', left_env, B)
        left_env = np.einsum('iab,iajc->ibjc', left_env, upper_env[self.Lc])
        left_env = np.einsum('iajb,iakc->ijbkc', left_env, Bs[self.Lc])
        return left_env
    
    def _calculate_left_environments(self, Bs, upper_env):
        left_env = np.ones((self.batch, 1, 1))
        left_envs = [left_env]
        # sweep left to right
        for B, Benv in zip(Bs[:self.Lc-1], upper_env[:self.Lc-1]):
            left_env = np.einsum('iab,iajc->ibjc', left_env, Benv)
            left_env = np.einsum('iajb,iajc->ibc', left_env, B)
            left_envs.append(left_env)
        # reorder right_envs before returning
        left_envs = left_envs[::-1]
        return left_envs
    
    def _calculate_right_environment(self, Bs, upper_env):
        right_env = np.ones((self.batch, 1, 1))
        # sweep right to left
        for B, Benv in zip(Bs[:self.Lc:-1], upper_env[:self.Lc:-1]):
            right_env = np.einsum('iajb,ibc->iajc', Benv, right_env)
            right_env = np.einsum('iajb,icjb->iac', right_env, B)
        right_env = np.einsum('iajb,ibc->iajc', upper_env[self.Lc], right_env)
        right_env = np.einsum('iajb,ickb->iajck', right_env, Bs[self.Lc])
        return right_env
    
    def apply_one_layer_right(self, layer, Bs, Lambdas):
        # assumes the given Bs are in right-canonical form
        # apply gates on right half
        gates = self.get_gates(q_start=self.Lc, q_end=self.L, layer=layer)
        for i, gate in enumerate(gates):
            idx = self.Lc + i
            Bl, Br, Lambdar = self.apply_gate(gate,
                                              Lambdas[idx],
                                              Bs[idx],
                                              Bs[idx+1])
            Bs[idx] = Bl
            Bs[idx+1] = Br
            Lambdas[idx+1] = Lambdar
        return Bs, Lambdas
    
    def apply_one_layer_right_conj(self, layer, Bs, Lambdas):
        # assumes the given Bs are in right-canonical form
        # apply gates on right half
        gates = self.get_gates(q_start=self.Lc, q_end=self.L, layer=layer)
        for i, gate in enumerate(gates):
            idx = self.Lc + i
            Bl, Br, Lambdar = self.apply_gate(gate.conj(),
                                              Lambdas[idx],
                                              Bs[idx],
                                              Bs[idx+1])
            Bs[idx] = Bl
            Bs[idx+1] = Br
            Lambdas[idx+1] = Lambdar
        return Bs, Lambdas
    
    def apply_one_layer_left(self, layer, Bs, Lambdas):
        # assumes the given Bs are in right-canonical form
        # apply gates on left half
        gates = self.get_gates(q_start=self.Lc-1, q_end=None, layer=layer)
        for i, gate in enumerate(gates):
            idx = self.Lc-1 - i
            Bl, Br, Lambdar = self.apply_gate(gate,
                                              Lambdas[idx],
                                              Bs[idx],
                                              Bs[idx+1])
            Bs[idx] = Bl
            Bs[idx+1] = Br
            Lambdas[idx+1] = Lambdar
        return Bs, Lambdas
    
    def _overlap(self, Bs, target_Bs):
        # assumes target_Bs is already complex conjugated
        overlap = np.ones((self.batch, 1, 1))
        for B, tB in zip(Bs, target_Bs):
            overlap = np.einsum('iab,iajc->ibjc', overlap, tB)
            overlap = np.einsum('ibja,ibjc->iac', overlap, B)
        return np.squeeze(overlap)
    
    def initial_state(self):
        raise NotImplementedError(
            'This function should be implemented by the inheriting class.')
    
    def fully_contract(self):
        # prepare initial state
        Bs, Lambdas = self.initial_state()
        # apply layers
        for layer in range(self.layers):
            Bs, Lambdas = self.apply_one_layer_right(layer, Bs, Lambdas)
            Bs, Lambdas = self.apply_one_layer_left(layer, Bs, Lambdas)
        return Bs, Lambdas
    
    def compute_Lambdas(self, Bs, right_canonical=False):
        # bring Bs into right-canonical form
        if not right_canonical:
            Bs = self.right_canonical(Bs)
        # iterate over bonds to fill up Lambdas
        Lambdas = [np.ones(Bs[0].shape[:2])]
        for i in range(self.L-1):
            theta = np.einsum('ia,iajb,ibkc->iajkc', Lambdas[-1], Bs[i], Bs[i+1])
            _, chil, _, _, chir = theta.shape
            theta = theta.reshape(self.batch, chil * self.d, self.d * chir)
            S = stable_svdvals(theta)
            S = S[:, :Bs[i].shape[-1]]
            Lambdas.append(S / np.linalg.norm(S, axis=1)[:, None])
        return Bs, Lambdas
    
    def truncate(self, As, chi_max=2, which='right_to_left'):
        if which == 'right_to_left':
            # sweeps right to left
            # assumes left-canonical form, returns right-canonical form
            for i in range(len(As)-1, 0, -1):
                # two-site tensor
                A0 = As[i-1]
                A1 = As[i]
                theta = np.einsum('iajb,ibkc->iajkc', A0, A1)
                _, chil, _, _, chir = theta.shape
                theta = theta.reshape(self.batch, chil * self.d, self.d * chir)
                # singular value decomposition
                X, S, Y = stable_svd(theta)
                # new bond dimension
                chi_new = min(S.shape[1],
                              max(self.chi_min, max(np.sum(S > self.svd_min, axis=1))),
                              chi_max)
                # normalize truncated state
                S = S[:, :chi_new]\
                    / np.sqrt(np.sum(S[:, :chi_new]**2, axis=1))[:, None]
                # truncate and save
                As[i-1] = (X[:, :, :chi_new] * S[:, None, :])\
                    .reshape(self.batch, -1, self.d, chi_new)
                As[i] = Y[:, :chi_new, :]\
                    .reshape(self.batch, chi_new, self.d, -1)
        elif which == 'left_to_right':
            # sweeps left to right
            # assumes right-canonical form, returns left-canonical form
            for i in range(len(As)-1):
                # two-site tensor
                A0 = As[i]
                A1 = As[i+1]
                theta = np.einsum('iajb,ibkc->iajkc', A0, A1)
                _, chil, _, _, chir = theta.shape
                theta = theta.reshape(self.batch, chil * self.d, self.d * chir)
                # singular value decomposition
                X, S, Y = stable_svd(theta)
                # new bond dimension
                chi_new = min(S.shape[1],
                              max(self.chi_min, max(np.sum(S > self.svd_min, axis=1))),
                              chi_max)
                # normalize truncated state
                S = S[:, :chi_new]\
                    / np.sqrt(np.sum(S[:, :chi_new]**2, axis=1))[:, None]
                # truncate and save
                As[i] = X[:, :, :chi_new]\
                    .reshape(self.batch, -1, self.d, chi_new)
                As[i+1] = (S[:, :, None] * Y[:, :chi_new, :])\
                    .reshape(self.batch, chi_new, self.d, -1)
        else:
            raise ValueError("Wrong argument 'which' given for 'truncate'.")
        return As
    
    def truncate_to_chi(self, Bs, chi_final, factor=1.5):
        # could truncate in one go, but intermediate steps give better result
        chi_max = max([B.shape[1] for B in Bs])
        while chi_max > chi_final: # exponentially reduce chi until chi=1
            # set chi_max and truncate left to right, gives left-canonical MPS
            chi_max = max(chi_final, int(chi_max/factor))
            Bs = self.truncate(Bs, chi_max=chi_max, which='left_to_right')
            # set chi_max and truncate right to left, gives right-canonical MPS
            chi_max = max(chi_final, int(chi_max/factor))
            Bs = self.truncate(Bs, chi_max=chi_max, which='right_to_left')
        return Bs
    
    def apply_conjugate_circuit(self, Bs, Lambdas=None):
        # get Bs and Lambdas
        Bs = Bs.copy()
        if Lambdas is None:
            Bs, Lambdas = self.compute_Lambdas(Bs)
        else:
            Lambdas = Lambdas.copy()
        # loop through the layers from top to bottom
        for layer in range(self.layers-1, -1, -1):
            # first half sweep from right to left
            gates = self.get_gates(q_start=None, q_end=self.Lc, layer=layer)
            for i, gate in enumerate(gates):
                idx = self.L-1-i
                Bl, Br, Lambdar = self.apply_gate(gate.transpose(0,2,1).conj(),
                                                  Lambdas[idx-1],
                                                  Bs[idx-1],
                                                  Bs[idx])
                Bs[idx-1] = Bl
                Bs[idx] = Br
                Lambdas[idx] = Lambdar
            # second half sweep from left to right
            gates = self.get_gates(q_start=0, q_end=self.Lc+1, layer=layer)
            for i, gate in enumerate(gates):
                Bl, Br, Lambdar = self.apply_gate(gate.transpose(0,2,1).conj(),
                                                  Lambdas[i],
                                                  Bs[i],
                                                  Bs[i+1])
                Bs[i] = Bl
                Bs[i+1] = Br
                Lambdas[i+1] = Lambdar
        return Bs, Lambdas

class sweeping_sequential_circuit(sequential_circuit):
    
    def __init__(self, L, layers, batchsize=1, Lc=None, initial_gates=None):
        # initialize via base class
        super().__init__(L, layers, batchsize, Lc)
        
        # initial gates
        self._set_initial_gates(initial_gates)
        
        # 'learning rate' for SVD update
        self.r = 1.0
    
    def _set_initial_gates(self, initial_gates=None):
        # add initial_gates
        strength = 1e-2
        # initialize random gates
        if initial_gates is None:
            gates = random_unitaries(
                shape=(self.batch, self.L-1, self.layers, self.d**2, self.d**2),
                strength=strength,
                real=self.real)
        # initialize with given gates
        else:
            gates = initial_gates[:self.batch, :self.L-1, :self.layers]
            batch, L, layers, d1, d2 = initial_gates.shape
            assert d1 == d2 and d2 == self.d**2
            
            if batch < self.batch:
                gates = np.append(
                    gates,
                    random_unitaries(
                        shape=(self.batch-batch, L, layers, d2, d2),
                        strength=strength,
                        real=self.real),
                    axis=0)
            
            if L < self.L-1:
                gates = np.append(
                    gates,
                    random_unitaries(
                        shape=(self.batch, self.L-1-L, layers, d2, d2),
                        strength=strength,
                        real=self.real),
                    axis=1)
            
            if layers < self.layers:
                gates = np.append(
                    gates,
                    random_unitaries(
                        shape=(self.batch, self.L-1, self.layers-layers, d2, d2),
                        strength=strength,
                        real=self.real),
                    axis=2)
                
        if self.SO:
            gates = polar_projection(gates, SO=True)
        
        self.gates = gates.astype(self.dtype)
    
    def get_gates(self, q_start, q_end, layer):
        if q_start is None:
            gates = self.gates[:, :q_end:-1, layer]
        elif q_end is None:
            gates = self.gates[:, q_start::-1, layer]
        elif q_start < q_end:
            gates = self.gates[:, q_start:q_end, layer]
        else:
            gates = self.gates[:, q_start:q_end:-1, layer]
        return np.moveaxis(gates, 1, 0)
    
    def initial_state(self):
        B = np.zeros((self.batch, 1, self.d, 1), dtype=self.dtype)
        B[:, :, 0, :] = 1.
        Bs = [B for j in range(self.L)]
        Lambda = np.ones((self.batch, 1), dtype=self.dtype)
        Lambdas = [Lambda for j in range(self.L)]
        return Bs, Lambdas
    
    def optimize_initial_state(self, upper_env):
        pass
    
    def sweep(self, target_Bs, target_Lambdas):
        # calculate upper environments
        upper_env_Bs, upper_env_Lambdas =\
            self._calculate_upper_environments(target_Bs, target_Lambdas)
        # prepare initial state
        self.optimize_initial_state(upper_env_Bs[0])
        Bs, Lambdas = self.initial_state()
        # sweep through layers (and upper environments) from bottom to top
        for layer, upper_env in enumerate(upper_env_Bs):
            # first deal with right half of the system
            right_envs = self._calculate_right_environments(Bs, upper_env)
            left_env = self._calculate_left_environment(Bs, upper_env)
            # sweep through the gates from left to right
            gates = self.get_gates(q_start=self.Lc, q_end=self.L, layer=layer)
            updated_gates = np.zeros((self.batch, self.L-1-self.Lc,
                                      self.d**2, self.d**2),
                                     dtype=self.dtype)
            for i, (gate, right_env) in enumerate(zip(gates, right_envs)):
                idx = self.Lc + i
                # calculate environment and remove old two-qubit gate
                # we can reuse part of this calculation for updating left_env
                left_env = np.einsum('ijakb,ialc->ijlckb',
                                     left_env,
                                     upper_env[idx+1])
                left_env = np.einsum('ijklm,ilmanb->ijkanb',
                                     gate.conj()\
                                         .reshape(self.batch, *(self.d,)*4),
                                     left_env)
                left_env = np.einsum('ijkalb,ibmc->ijkalmc',
                                     left_env,
                                     Bs[idx+1])
                environment = np.einsum('ijkalmb,iab->ijklm',
                                        left_env,
                                        right_env)
                # update gate
                U = self._update_gate(environment, gate)
                updated_gates[:, i] = U
                # update left_env with new gate
                left_env = np.einsum('ijkalmb,ijnlm->ikanb',
                                     left_env,
                                     U.reshape(self.batch, *(self.d,)*4))
            # remove old gates from right half of upper_env
            upper_env, _ = self.apply_one_layer_right_conj(
                                                    layer,
                                                    upper_env,
                                                    upper_env_Lambdas[layer])
            # update old gates to new gates
            self.gates[:, self.Lc:, layer] = updated_gates
            # apply updated gates on the right half of MPS
            Bs, Lambdas = self.apply_one_layer_right(layer, Bs, Lambdas)
            # now deal with left half of the system
            left_envs = self._calculate_left_environments(Bs, upper_env)
            right_env = self._calculate_right_environment(Bs, upper_env)
            # sweep through the gates from right to left
            gates = self.get_gates(q_start=self.Lc-1, q_end=None, layer=layer)
            for i, (gate, left_env) in enumerate(zip(gates, left_envs)):
                idx = self.Lc-1 - i
                # calculate environment and remove old two-qubit gate
                # we can reuse part of this calculation for updating right_env
                right_env = np.einsum('iajb,ibkcl->iajkcl',
                                      upper_env[idx],
                                      right_env)
                right_env = np.einsum('ijklm,ialmbn->iajkbn',
                                      gate.conj()\
                                          .reshape(self.batch, *(self.d,)*4),
                                      right_env)
                right_env = np.einsum('iclb,iajkbm->iajkclm',
                                      Bs[idx],
                                      right_env)
                environment = np.einsum('iab,iajkblm->ijklm',
                                        left_env,
                                        right_env)
                # update gate
                U = self._update_gate(environment, gate)
                self.gates[:, idx, layer] = U
                # update right_env with new gate
                right_env = np.einsum('iajkblm,inklm->iajbn',
                                      right_env,
                                      U.reshape(self.batch, *(self.d,)*4))
            # update MPS tensors with new gates in left half
            Bs, Lambdas = self.apply_one_layer_left(layer, Bs, Lambdas)
        return Bs, Lambdas
    
    def optimize_circuit(self, target_Bs, target_Lambdas=None, iters=20):
        target_Bs = [B.conj() for B in target_Bs]
        if target_Lambdas is None:
            target_Bs, target_Lambdas = self.compute_Lambdas(target_Bs)
        Bs, _ = self.fully_contract()
        overlaps = [self._overlap(Bs, target_Bs)]
        # iterate
        time = []
        for i in range(iters):
            start = perf_counter()
            Bs, Lambdas = self.sweep(target_Bs, target_Lambdas)
            time.append(perf_counter() - start)
            overlaps.append(self._overlap(Bs, target_Bs))
        return overlaps, np.asarray(time), Bs 
    
    def add_one_layer(self, target_Bs, target_Lambdas=None):
        # contract conjugate of circuit to target state
        Bs, Lambdas = self.apply_conjugate_circuit(target_Bs, target_Lambdas)
        # truncate to bond dimension chi=d
        Bs = self.truncate_to_chi(Bs, chi_final=self.d)
        
        # bring into mixed canonical form with canonical center on site Lc
        # B_0 ... B_Lc B_Lc+1 ... B_L-1 ---> A_0 ... A_Lc C B_Lc+1 ... B_L-1
        R = np.ones((self.batch, 1, 1))
        for i in range(self.Lc+1):
            _, chil, d, chir = Bs[i].shape
            B = np.einsum('iab,ibjc->iajc', R, Bs[i])\
                .reshape(self.batch, chil * d, chir)
            Q, R = np.linalg.qr(B)
            Bs[i] = Q.reshape(self.batch, chil, d, chir)
        C = R
        
        # turn MPS into gate sequence
        new_gates = np.zeros((self.batch, self.L-1, d**2, d**2),
                             dtype=self.gates.dtype)
        U = np.zeros((self.batch, d**2, d, d),
                     dtype=self.gates.dtype)
        iso = np.einsum('iajb,ibkc->iajkc', Bs[0], Bs[1])\
                        .reshape(self.batch, d**2, d)
        U[:, :, 0, :] = iso
        U, R = np.linalg.qr(U.reshape(self.batch, d**2, d**2))
        U = U * sign(np.einsum('ijj->ij', R))[:, None, :]
        new_gates[:, 0, :, :] = U
        for i in range(2, self.Lc+1):
            U = np.zeros((self.batch, d**2, d, d),
                         dtype=self.gates.dtype)
            iso = Bs[i].reshape(self.batch, d**2, d)
            U[:, :, 0, :] = iso
            U, R = np.linalg.qr(U.reshape(self.batch, d**2, d**2))
            U = U * sign(np.einsum('ijj->ij', R))[:, None, :]
            new_gates[:, i-1, :, :] = U
        U = np.zeros((self.batch, d**2, d**2),
                     dtype=self.gates.dtype)
        iso = C.reshape(self.batch, d**2)
        U[:, :, 0] = iso
        U, R = np.linalg.qr(U)
        U = U * sign(np.einsum('ijj->ij', R))[:, None, :]
        new_gates[:, self.Lc, :, :] = U
        for i in range(self.Lc+1, self.L-2):
            U = np.zeros((self.batch, d**2, d, d),
                         dtype=self.gates.dtype)
            iso = Bs[i].transpose(0,2,3,1)\
                                 .reshape(self.batch, d**2, d)
            U[:, :, 0, :] = iso
            U, R = np.linalg.qr(U.reshape(self.batch, d**2, d**2))
            U = U * sign(np.einsum('ijj->ij', R))[:, None, :]
            U = U.reshape(self.batch, d**2, d, d)\
                 .transpose(0,1,3,2)\
                 .reshape(self.batch, d**2, d**2)
            new_gates[:, i, :, :] = U
        U = np.zeros((self.batch, d**2, d, d),
                     dtype=self.gates.dtype)
        iso = np.einsum('iajb,ibkc->ijkac', Bs[-2], Bs[-1])\
                        .reshape(self.batch, d**2, d)
        U[:, :, 0, :] = iso
        U, R = np.linalg.qr(U.reshape(self.batch, d**2, d**2))
        U = U * sign(np.einsum('ijj->ij', R))[:, None, :]
        U = U.reshape(self.batch, d**2, d, d)\
             .transpose(0,1,3,2)\
             .reshape(self.batch, d**2, d**2)
        new_gates[:, -1, :, :] = U
        
        if self.SO:
            # check that determinants are positive
            dets = np.linalg.det(new_gates)
            new_gates[:, :, :, -1] *= dets[:, :, None]
        
        # update internal parameters
        self.layers += 1
        self.gates = np.append(new_gates[:, :, None, :, :], self.gates, axis=2)

class sweeping_three_cnot_sequential_circuit(sweeping_sequential_circuit):
    
    def __init__(self, L, layers, batchsize=1, Lc=None, initial_gates=None,
                 dtype=np.float64):
        # dtype
        if dtype in [np.float64, 'real', 'float', float]:
            self.dtype = np.float64
            self.real = True
        else:
            self.dtype = np.complex128
            self.real = False
        self.SO = False
        # initialize via base class
        super().__init__(L, layers, batchsize, Lc, initial_gates)
    
    def _update_gate(self, environment, old_gate):
        # polar decomposition of environment
        U = polar_projection(environment\
                                .reshape(self.batch, self.d**2, self.d**2))
        U = U.conj()
        # rate of update
        if self.r < 1.:
            U = polar_projection((1-self.r) * old_gate + self.r * U)
        return U
    
    def check_form_of_gates(self):
        Ids = np.einsum('...ij,...kj->...ik', self.gates, self.gates.conj())
        Id_diffs = np.linalg.norm(Ids - np.eye(self.d**2), axis=(-2,-1))
        print('Maximal deviation from unitary:')
        print(np.max(Id_diffs))

class sweeping_two_cnot_sequential_circuit(sweeping_sequential_circuit):
    
    def __init__(self, L, layers, batchsize=1, Lc=None, initial_gates=None):
        # dtype
        self.dtype = np.float64
        self.real = True
        self.SO = True
        # initialize via base class
        super().__init__(L, layers, batchsize, Lc, initial_gates)
    
    def _update_gate(self, environment, old_gate):
        # polar decomposition of environment
        U = polar_projection(environment\
                                .reshape(self.batch, self.d**2, self.d**2),
                             SO=True)
        # rate of update
        if self.r < 1.:
            U = polar_projection((1-self.r) * old_gate + self.r * U, SO=True)
        return U
    
    def check_form_of_gates(self):
        assert np.linalg.norm(self.gates.imag) == 0
        dets = np.linalg.det(self.gates)
        det_diffs = np.abs(dets - 1)
        print('Maximal deviation from special orthogonal:')
        print(np.max(det_diffs))

class sweeping_single_cnot_sequential_circuit(sweeping_sequential_circuit):
    
    def __init__(self, L, layers, batchsize=1, Lc=None, initial_gates=None,
                 dtype=np.float64):
        # dtype
        if dtype in [np.float64, 'real', 'float', float]:
            self.dtype = np.float64
            self.real = True
        else:
            self.dtype = np.complex128
            self.real = False
        # define CNOT
        X = np.fliplr(np.eye(2))
        CNOT = np.eye(4)
        CNOT[-2:,-2:] = X
        self.CNOT = CNOT.reshape((2,)*4)
        # initialize via base class (without initial gates)
        super().__init__(L, layers, batchsize, Lc, initial_gates)
    
    def _set_initial_gates(self, initial_gates):
        # add initial_gates
        strength = 1e-2
        # initialize random gates
        if initial_gates is None:
            first_gates = random_unitaries(
                shape=(self.batch, self.L, self.d, self.d),
                strength=strength,
                real=self.real)
            self.first_gates = first_gates.astype(self.dtype)
            
            gates = random_unitaries(
                shape=(2, self.batch, self.L-1, self.layers, self.d, self.d),
                strength=strength,
                real=self.real)
            gates = np.einsum('...jk,...lm->...jlkm', *gates)
            gates = np.einsum('...jklm,lmno->...jkno', gates, self.CNOT)\
                    .reshape(*gates.shape[:-4], self.d**2, self.d**2)
            self.gates = gates.astype(self.dtype)
        # initialize with given gates
        else:
            first_gates = initial_gates[0][:self.batch, :self.L]
            batch, L, d1, d2 = first_gates.shape
            assert d1 == d2 and d2 == self.d
            
            gates = initial_gates[1][:self.batch, :self.L-1, :self.layers]
            if gates.shape[:-3] == (2, self.d, self.d):
                gates = np.einsum('...jklm,lmno->...jkno', gates, self.CNOT)
            batch, L1, layers, d1, d2 = gates.shape
            assert L-1 == L and d1 == d2 and d2 == self.d**2
            
            if batch < self.batch:
                new = random_unitaries(
                    shape=(self.batch-batch, L, self.d, self.d),
                    strength=strength,
                    real=self.real)
                first_gates = np.append(first_gates, new, axis=0)
                
                new = random_unitaries(
                    shape=(2, self.batch-batch, L-1, layers, self.d, self.d),
                    strength=strength,
                    real=self.real)
                new = np.einsum('...jk,...lm->...jlkm', *new)
                new = np.einsum('...jklm,lmno->...jkno', new, self.CNOT)\
                        .reshape(*new.shape[:-4], self.d**2, self.d**2)
                gates = np.append(gates, new, axis=0)
            
            if L < self.L:
                new = random_unitaries(
                    shape=(self.batch, self.L-L, self.d, self.d),
                    strength=strength,
                    real=self.real)
                first_gates = np.append(first_gates, new, axis=1)
                
                new = random_unitaries(
                    shape=(2, self.batch, self.L-L, layers, self.d, self.d),
                    strength=strength,
                    real=self.real)
                new = np.einsum('...jk,...lm->...jlkm', *new)
                new = np.einsum('...jklm,lmno->...jkno', new, self.CNOT)\
                        .reshape(*new.shape[:-4], self.d**2, self.d**2)
                gates = np.append(gates, new, axis=1)
            
            if layers < self.layers:
                new = random_unitaries(
                    shape=(2, self.batch, self.L-1,
                           self.layers-layers, self.d, self.d),
                    strength=strength,
                    real=self.real)
                new = np.einsum('...jk,...lm->...jlkm', *new)
                new = np.einsum('...jklm,lmno->...jkno', new, self.CNOT)\
                        .reshape(*new.shape[:-4], self.d**2, self.d**2)
                gates = np.append(gates, new, axis=2)
            
            self.first_gates = first_gates.astype(self.dtype)
            self.gates = gates.astype(self.dtype)
    
    def initial_state(self):
        B = np.zeros((self.batch, 1, self.d, 1), dtype=self.dtype)
        B[:, :, 0, :] = 1.
        Bs = [np.einsum('ijk,iakb->iajb',
                        self.first_gates[:,j], B) for j in range(self.L)]
        Lambda = np.ones((self.batch, 1), dtype=self.dtype)
        Lambdas = [Lambda for j in range(self.L)]
        return Bs, Lambdas
    
    def optimize_initial_state(self, upper_env):
        # get initial state
        Bs, _ = self.initial_state()
        # calculate left and right environments
        left_env = np.ones((self.batch, 1, 1), dtype=self.dtype)
        right_env = np.ones((self.batch, 1, 1), dtype=self.dtype)
        right_envs = [right_env]
        for B, Benv in zip(Bs[:0:-1], upper_env[:0:-1]):
            right_env = np.einsum('iajb,ibc->iajc', Benv, right_env)
            right_env = np.einsum('iajb,icjb->iac', right_env, B)
            right_envs.append(right_env)
        right_envs = right_envs[::-1]
        # sweep left to right through sites and update gates
        gates = np.moveaxis(self.first_gates, 1, 0)
        for i, (gate, right_env) in enumerate(zip(gates, right_envs)):
            # calculate environment (no need to remove old single-qubit gate)
            # we can reuse part of this calculation for updating left_env
            left_env = np.einsum('iab,iajc->ibjc', left_env, upper_env[i])
            left_env = np.einsum('iajb,iakc->ijbkc', left_env, Bs[i])
            environment = np.einsum('ijakb,iab->ijk', left_env, right_env)
            # update gate
            U = self._update_single_qubit_gate(environment)
            self.first_gates[:, i] = U @ gate
            # update left_env with new gate
            left_env = np.einsum('ijakb,ijk->iab', left_env, U)
        
    def _update_single_qubit_gate(self, environment):
        U = polar_projection(environment)
        U = U.conj()
        return U
    
    def _single_qubit_gates_from_two_qubit_gate(self, gate):
        rand = np.random.rand(self.d, self.d)
        U1 = np.einsum('...jklm,lmno,ko->...jn', gate, self.CNOT, rand)
        U1 = polar_projection(U1)
        U2 = np.einsum('...jklm,lmon,jo->...kn', gate, self.CNOT, rand)
        U2 = polar_projection(U2)
        return U1, U2
    
    def check_form_of_gates(self):
        assert self.first_gates.shape == (self.batch, self.L, self.d, self.d)
        U1, U2 = self._single_qubit_gates_from_two_qubit_gate(
                    self.gates.reshape(*self.gates.shape[:-2], *(self.d,)*4))
        gates = np.einsum('...jk,...lm->...jlkm', U1, U2)
        gates = np.einsum('...jklm,lmno->...jkno', gates, self.CNOT)\
                                                    .reshape(self.gates.shape)
        args1 = np.angle(self.gates[..., 0, 0] + self.gates[..., 0, 1])
        phases1 = np.exp(-1j * args1)[..., None, None]
        args2 = np.angle(gates[..., 0, 0] + gates[..., 0, 1])
        phases2 = np.exp(-1j * args2)[..., None, None]
        print('Maximal deviation from single-CNOT form:')
        norm_diffs = np.linalg.norm(gates / phases2 - self.gates / phases1,
                                    axis=(-2,-1))
        print(np.max(norm_diffs))
    
    def _update_gate(self, environment, old_gate):
        old_gate = old_gate.reshape(self.batch, *(self.d,)*4)
        # update environment to second qubit and polar decomposition
        temp_env = np.einsum('ijklm,ijnlm->ikn', environment, old_gate)
        U = self._update_single_qubit_gate(temp_env)
        gate = np.einsum('ijklm,ink->ijnlm', old_gate, U)
        # update environment to second qubit and polar decomposition
        temp_env = np.einsum('ijklm,inklm->ijn', environment, gate)
        U = self._update_single_qubit_gate(temp_env)
        gate = np.einsum('ijklm,inj->inklm', gate, U)
        # rate of update
        if self.r < 1.:
            # get single-qubit gates
            V1, V2 = self._single_qubit_gates_from_two_qubit_gate(old_gate)
            U1, U2 = self._single_qubit_gates_from_two_qubit_gate(gate)
            # learning rate update
            U1 = polar_projection((1-self.r) * V1 + self.r * U1)
            U2 = polar_projection((1-self.r) * V2 + self.r * U2)
            # combine to gate
            gates = np.einsum('ijk,ilm->ijlkm', U1, U2)
            gates = np.einsum('ijklm,lmno->ijkno', gates, self.CNOT)
        return gate.reshape(self.batch, self.d**2, self.d**2)
    
    def add_one_layer_fast(self, target_Bs, target_Lambdas=None):
        # construct new first layer from CNOTs and previous single qubit gates
        gates = np.zeros((2, self.batch, self.L-1, self.d, self.d),
                         dtype=self.dtype)
        gates[:, :, :] = np.eye(self.d)
        gates[0, :, 0] = self.first_gates[:, 0]
        gates[1, :, :self.Lc] = self.first_gates[:, 1:self.Lc+1]
        gates[0, :, self.Lc+1:] = self.first_gates[:, self.Lc+1:-1]
        gates[1, :, -1] = self.first_gates[:, -1]
        gates = np.einsum('...jk,...lm->...jlkm', *gates)
        gates = np.einsum('...jklm,lmno->...jkno', gates, self.CNOT)
        gates = gates.reshape(*gates.shape[:-4], self.d**2, self.d**2)
        self.layers += 1
        self.gates = np.append(gates[:, :, None, :, :], self.gates, axis=2)
        
        # contract conjugate of updated circuit to target state
        Bs, Lambdas = self.apply_conjugate_circuit(target_Bs, target_Lambdas)
        
        # truncate to product state, i.e., bond dimension chi=1
        Bs = self.truncate_to_chi(Bs, chi_final=1)
        
        # new first layer of single-qubit gates from product state
        new_gates = np.zeros((self.batch, self.L, self.d, self.d),
                             dtype=self.dtype)
        for i, B in enumerate(Bs):
            U = np.zeros((self.batch, self.d, self.d),
                         dtype=self.dtype)
            U[:, :, 0] = np.squeeze(B)
            U, R = np.linalg.qr(U)
            U = U * sign(np.einsum('ijj->ij', R))[:, None, :]
            new_gates[:, i, :, :] = U
        self.first_gates = new_gates

    def add_one_layer(self, target_Bs, target_Lambdas=None):
        # contract conjugate of updated circuit to target state
        Bs, Lambdas = self.apply_conjugate_circuit(target_Bs, target_Lambdas)

        # define array to store new gates
        new_gates = np.zeros((self.batch, self.L-1, self.d**2, self.d**2),
                             dtype=self.dtype)
        # iterate site 0 to site Lc-1, then iterate site L-2 to site Lc
        for i in [*range(self.Lc), *range(self.L-2, self.Lc-1, -1)]:
            # truncate target state to product state
            Ms = self.truncate_to_chi(Bs.copy(), chi_final=1)
            # get the two single-qubit gates on current bond
            U = np.zeros((self.batch, 2, self.d, self.d),
                          dtype=self.dtype)
            U[:, 0, :, 0] = np.squeeze(Ms[i])
            U[:, 1, :, 0] = np.squeeze(Ms[i+1])
            U, R = np.linalg.qr(U)
            U = U * sign(np.einsum('ijkk->ijk', R))[:, :, None, :]
            # construct new two-site gate by combining with CNOT
            gate = np.einsum('ijk,ilm->ijlkm', U[:,0], U[:,1])
            gate = np.einsum('ijklm,lmno->ijkno', gate, self.CNOT)
            new_gates[:, i] = gate.reshape(self.batch, self.d**2, self.d**2)
            # apply conjugate of new gate to target state
            gate_dagger = new_gates[:, i].transpose(0,2,1).conj()
            Bl, Br, Lambdar = self.apply_gate(gate_dagger,
                                              Lambdas[i],
                                              Bs[i],
                                              Bs[i+1])
            Bs[i] = Bl
            Bs[i+1] = Br
            Lambdas[i+1] = Lambdar
        # update first layer of single-qubit gates
        Ms = self.truncate_to_chi(Bs.copy(), chi_final=1)
        U = np.zeros((self.batch, self.L, self.d, self.d),
                      dtype=self.dtype)
        for i in range(self.L):
            U[:, i, :, 0] = np.squeeze(Ms[i])
        U, R = np.linalg.qr(U)
        U = U * sign(np.einsum('ijkk->ijk', R))[:, :, None, :]
        self.first_gates = U
        # update two-qubit gates
        self.layers += 1
        self.gates = np.append(new_gates[:, :, None, :, :], self.gates, axis=2)
    
    def add_one_layer_flip_CNOTs(self, target_Bs, target_Lambdas=None):
        # contract conjugate of updated circuit to target state
        Bs, Lambdas = self.apply_conjugate_circuit(target_Bs, target_Lambdas)

        # define array to store new gates
        new_gates = np.zeros((self.batch, self.L-1, self.d**2, self.d**2),
                             dtype=self.dtype)
        # iterate site 0 to site Lc-1
        for i in range(self.Lc):
            # truncate target state to product state
            Ms = self.truncate_to_chi(Bs.copy(), chi_final=1)
            # get the two single-qubit gates on current bond
            U = np.zeros((self.batch, 2, self.d, self.d),
                          dtype=self.dtype)
            U[:, 0, :, 0] = np.squeeze(Ms[i])
            U[:, 1, :, 0] = np.squeeze(Ms[i+1])
            U, R = np.linalg.qr(U)
            U = U * sign(np.einsum('ijkk->ijk', R))[:, :, None, :]
            # construct new two-site gate by combining with CNOT
            gate = np.einsum('ijk,ilm->ijlkm', U[:,0], U[:,1])
            gate = np.einsum('ijklm,lmno->ijkno', gate, self.CNOT)
            new_gates[:, i] = gate.reshape(self.batch, self.d**2, self.d**2)
            # apply conjugate of new gate to target state
            gate_dagger = new_gates[:, i].transpose(0,2,1).conj()
            Bl, Br, Lambdar = self.apply_gate(gate_dagger,
                                              Lambdas[i],
                                              Bs[i],
                                              Bs[i+1])
            Bs[i] = Bl
            Bs[i+1] = Br
            Lambdas[i+1] = Lambdar
        # iterate site L-2 to site Lc
        for i in range(self.L-2, self.Lc-1, -1):
            # truncate target state to product state
            Ms = self.truncate_to_chi(Bs.copy(), chi_final=1)
            # get the two single-qubit gates on current bond
            U = np.zeros((self.batch, 2, self.d, self.d),
                          dtype=self.dtype)
            U[:, 0, :, 0] = np.squeeze(Ms[i])
            U[:, 1, :, 0] = np.squeeze(Ms[i+1])
            U, R = np.linalg.qr(U)
            U = U * sign(np.einsum('ijkk->ijk', R))[:, :, None, :]
            # construct new two-site gate by combining with flipped CNOT
            gate = np.einsum('ijk,ilm->ijlkm', U[:,0], U[:,1])
            gate = np.einsum('ijklm,lmno->ijkno', gate, self.CNOT.transpose(1,0,3,2))
            new_gates[:, i] = gate.reshape(self.batch, self.d**2, self.d**2)
            # apply conjugate of new gate to target state
            gate_dagger = new_gates[:, i].transpose(0,2,1).conj()
            Bl, Br, Lambdar = self.apply_gate(gate_dagger,
                                              Lambdas[i],
                                              Bs[i],
                                              Bs[i+1])
            Bs[i] = Bl
            Bs[i+1] = Br
            Lambdas[i+1] = Lambdar
        # update first layer of single-qubit gates
        Ms = self.truncate_to_chi(Bs.copy(), chi_final=1)
        U = np.zeros((self.batch, self.L, self.d, self.d),
                      dtype=self.dtype)
        for i in range(self.L):
            U[:, i, :, 0] = np.squeeze(Ms[i])
        U, R = np.linalg.qr(U)
        U = U * sign(np.einsum('ijkk->ijk', R))[:, :, None, :]
        # flip CNOTs with Hadamards
        H = np.array([[1., 1.], [1.,-1.]])/np.sqrt(2)
        new_gates[:, self.Lc:] = np.einsum('ijkl,lm->ijkm',
                                           new_gates[:, self.Lc:],
                                           np.kron(H, H))
        new_gates[:, self.Lc:-1] = np.einsum('kl,ijlm->ijkm',
                                             np.kron(np.eye(self.d), H),
                                             new_gates[:, self.Lc:-1])
        U[:, self.Lc:] = np.einsum('kl,ijlm->ijkm', H, U[:, self.Lc:])
        # update two-qubit gates
        self.layers += 1
        self.first_gates = U
        self.gates = np.append(new_gates[:, :, None, :, :], self.gates, axis=2)

class gradient_sequential_circuit(sequential_circuit):
    
    def __init__(self, L, layers, batchsize=1, Lc=None, initial_params=None):
        # initialize via base class
        super().__init__(L, layers, batchsize, Lc)
        
        # initial parameters
        self._set_initial_parameters(initial_params)
        
        # define gates
        self.CNOT = np.array([[1., 0., 0., 0.],
                              [0., 1., 0., 0.],
                              [0., 0., 0., 1.],
                              [0., 0., 1., 0.]])
        self.rCNOT = np.array([[1., 0., 0., 0.],
                               [0., 0., 0., 1.],
                               [0., 0., 1., 0.],
                               [0., 1., 0., 0.]])
        self.Id = np.eye(2)
        self.X = np.array([[0., 1.], [1., 0.]])
        self.Y = np.array([[0., -1.j], [1.j, 0.]])
        self.Z = np.diag([1., -1.])
    
    def _set_initial_parameters(self, initial_params=None):
        loc = 0
        scale = 1e-2
        num_params = self.num_params
        # initialize random parameters
        if initial_params is None:
            self.initial_state_params = np.random.normal(
                loc=loc,
                scale=scale,
                size=(self.batch, self.L, 2))
            
            self.params = np.random.normal(
                loc=loc,
                scale=scale,
                size=(self.batch, self.L-1, self.layers, num_params))
        # initialize with given parameters
        else:
            initial_state_params = initial_params[0][:self.batch, :self.L]
            batch, L, num = initial_state_params.shape
            assert num == 2
            
            params = initial_params[1][:self.batch, :self.L-1, :self.layers]
            batch, L1, layers, num = initial_params.shape
            assert L-1 == L and num == num_params
            
            if batch < self.batch:
                initial_state_params = np.append(
                    initial_state_params,
                    np.random.normal(
                        loc=loc,
                        scale=scale,
                        size=(self.batch-batch, L1+1, 2)),
                    axis=0)
                params = np.append(
                    params,
                    np.random.normal(
                        loc=loc,
                        scale=scale,
                        size=(self.batch-batch, L1, layers, num_params)),
                    axis=0)
            
            if L1 < self.L-1:
                initial_state_params = np.append(
                    initial_state_params,
                    np.random.normal(
                        loc=loc,
                        scale=scale,
                        size=(self.batch, self.L-L1-1, 2)),
                    axis=0)
                params = np.append(
                    params,
                    np.random.normal(
                        loc=loc,
                        scale=scale,
                        size=(self.batch, self.L-1-L1, layers, num_params)),
                    axis=1)
            
            if layers < self.layers:
                params = np.append(
                    params,
                    np.random.normal(
                        loc=loc,
                        scale=scale,
                        size=(self.batch, self.L-1, self.layers-layers, num_params)),
                    axis=2)
            
            self.initial_state_params = initial_state_params
            self.params = params
    
    def get_gates(self, q_start, q_end, layer):
        if q_start is None:
            params = self.params[:, :q_end:-1, layer]
        elif q_end is None:
            params = self.params[:, q_start::-1, layer]
        elif q_start < q_end:
            params = self.params[:, q_start:q_end, layer]
        else:
            params = self.params[:, q_start:q_end:-1, layer]
        gates = self.calculate_gate(params)
        return np.moveaxis(gates, 1, 0)
    
    def calculate_gate(self, params):
        raise NotImplementedError(
            'This function should be implemented by the inheriting class.')
    
    def calculate_gate_with_gradient(self, params):
        raise NotImplementedError(
            'This function should be implemented by the inheriting class.')
        
    def initial_state(self):
        Bs = [np.array([np.exp(-1j * self.initial_state_params[:, j, 1] / 2)\
                      * np.cos(self.initial_state_params[:, j, 0] / 2),
                        np.exp(1j * self.initial_state_params[:, j, 1] / 2)\
                      * np.sin(self.initial_state_params[:, j, 0] / 2)])\
              .transpose(1, 0)[:, None, :, None]
              for j in range(self.L)]
        Lambdas = [np.ones(Bs[0].shape[:2]) for j in range(self.L)]
        return Bs, Lambdas
    
    def initial_state_with_gradient(self):
        # initial state
        Bs = [np.array([np.exp(-1j * self.initial_state_params[:, j, 1] / 2)\
                      * np.cos(self.initial_state_params[:, j, 0] / 2),
                        np.exp(1j * self.initial_state_params[:, j, 1] / 2)\
                      * np.sin(self.initial_state_params[:, j, 0] / 2)])\
              .transpose(1, 0)[:, None, :, None]
              for j in range(self.L)]
        Lambdas = [np.ones(Bs[0].shape[:2]) for j in range(self.L)]
        # gradients
        grad_Bs = [(-1/2) *
            np.array(
                [[np.exp(-1j * self.initial_state_params[:, j, 1] / 2)\
                      * np.sin(self.initial_state_params[:, j, 0] / 2),
                 - np.exp(1j * self.initial_state_params[:, j, 1] / 2)\
                      * np.cos(self.initial_state_params[:, j, 0] / 2)],
                 [1j * np.exp(-1j * self.initial_state_params[:, j, 1] / 2)\
                           * np.cos(self.initial_state_params[:, j, 0] / 2),
                 - 1j * np.exp(1j * self.initial_state_params[:, j, 1] / 2)\
                           * np.sin(self.initial_state_params[:, j, 0] / 2)]])\
                .transpose(2, 0, 1)[:, :, None, :, None]
            for j in range(self.L)] # (batch, param, a, j, b)
        return Bs, Lambdas, grad_Bs
    
    def gradient_initial_state(self, upper_env):
        # fill up gradient
        initial_state_gradient = np.zeros(self.initial_state_params.shape,
                                          dtype=np.complex128)
        # initial state with gradients
        Bs, Lambdas, grad_Bs = self.initial_state_with_gradient()
        # sweep left to right through first layer of single-qubit gates
        right_env = np.ones((self.batch, 1, 1))
        right_envs = [right_env]
        for B, Benv in zip(Bs[:0:-1], upper_env[:0:-1]):
            right_env = np.einsum('iajb,ibc->iajc', Benv, right_env)
            right_env = np.einsum('iajb,icjb->iac', right_env, B)
            right_envs.append(right_env)
        right_envs = right_envs[::-1]
        left_env = np.ones((self.batch, 1, 1))
        for i, right_env in enumerate(right_envs):
            # compute on-site environment
            environment = np.einsum('iab,iajc,icd->ibjd',
                                    left_env,
                                    upper_env[i],
                                    right_env)
            # gradient
            grad = np.einsum('iajb,ipajb->ip', environment, grad_Bs[i])
            initial_state_gradient[:, i, :] = grad
            # update left_env
            left_env = np.einsum('iab,iajc,ibjd->icd',
                                 left_env,
                                 upper_env[i],
                                 Bs[i])
        return initial_state_gradient
    
    def gradient(self, target_Bs, target_Lambdas):
        # fill up gradient
        gradient = np.zeros(self.params.shape,
                            dtype=np.complex128)
        # calculate upper environments
        upper_env_Bs, upper_env_Lambdas =\
            self._calculate_upper_environments(target_Bs, target_Lambdas)
        # prepare initial state
        initial_state_gradient = self.gradient_initial_state(upper_env_Bs[0])
        Bs, Lambdas = self.initial_state()
        # sweep through layers (and upper environments) from bottom to top
        for layer, upper_env in enumerate(upper_env_Bs):
            # first deal with right half of the system
            right_envs = self._calculate_right_environments(Bs, upper_env)
            left_env = self._calculate_left_environment(Bs, upper_env)
            # sweep through the gates from left to right
            params = np.moveaxis(self.params[:, self.Lc:, layer], 1, 0)
            for i, (param, right_env) in enumerate(zip(params, right_envs)):
                gate, gate_grads = self.calculate_gate_with_gradient(param)
                idx = self.Lc + i
                # calculate environment and remove old two-qubit gate
                # we can reuse part of this calculation for updating left_env
                left_env = np.einsum('ijakb,ialc->ijlckb',
                                     left_env,
                                     upper_env[idx+1])
                left_env = np.einsum('ijklm,ilmanb->ijkanb',
                                     gate.conj()\
                                         .reshape(self.batch, *(self.d,)*4),
                                     left_env)
                left_env = np.einsum('ijkalb,ibmc->ijkalmc',
                                     left_env,
                                     Bs[idx+1])
                # calculate gradient
                environment = np.einsum('ijkalmb,iab->ijklm',
                                        left_env,
                                        right_env)\
                    .reshape(self.batch, self.d**2, self.d**2)
                current_gradients = np.einsum('ijk,iajk->ia', # -> (batch, num_params)
                                              environment, # (batch, 4, 4)
                                              gate_grads) # (batch, num_params, 4, 4)
                gradient[:, idx, layer, :] = current_gradients # (batch, L-1, layers, num_params)
                # update left_env and shift to next site
                left_env = np.einsum('ijkalmb,ijnlm->ikanb',
                                     left_env,
                                     gate.reshape(self.batch, *(self.d,)*4))
            # remove old gates from right half of upper_env
            upper_env, _ = self.apply_one_layer_right_conj(
                                                    layer,
                                                    upper_env,
                                                    upper_env_Lambdas[layer])
            # apply gates on the right half of MPS
            Bs, Lambdas = self.apply_one_layer_right(layer, Bs, Lambdas)
            # now deal with left half of the system
            left_envs = self._calculate_left_environments(Bs, upper_env)
            right_env = self._calculate_right_environment(Bs, upper_env)
            # sweep through the gates from right to left
            params = np.moveaxis(self.params[:, self.Lc-1::-1, layer], 1, 0)
            for i, (param, left_env) in enumerate(zip(params, left_envs)):
                gate, gate_grads = self.calculate_gate_with_gradient(param)
                idx = self.Lc-1 - i
                # calculate environment and remove old two-qubit gate
                # we can reuse part of this calculation for updating right_env
                right_env = np.einsum('iajb,ibkcl->iajkcl',
                                      upper_env[idx],
                                      right_env)
                right_env = np.einsum('ijklm,ialmbn->iajkbn',
                                      gate.conj()\
                                          .reshape(self.batch, *(self.d,)*4),
                                      right_env)
                right_env = np.einsum('iclb,iajkbm->iajkclm',
                                      Bs[idx],
                                      right_env)
                # calculate gradient
                environment = np.einsum('iab,iajkblm->ijklm',
                                        left_env,
                                        right_env)\
                    .reshape(self.batch, self.d**2, self.d**2)
                current_gradients = np.einsum('ijk,iajk->ia', # -> (batch, num_params)
                                              environment, # (batch, 4, 4)
                                              gate_grads) # (batch, num_params, 4, 4)
                gradient[:, idx, layer, :] = current_gradients # (batch, L-1, layers, num_params)
                # update right_env with new gate
                right_env = np.einsum('iajkblm,inklm->iajbn',
                                      right_env,
                                      gate.reshape(self.batch, *(self.d,)*4))
            # update MPS tensors with new gates in left half
            Bs, Lambdas = self.apply_one_layer_left(layer, Bs, Lambdas)
        # Full gradient is 2 Re(<target|state(x)>^* d/dx<target|state(x)>)
        overlap = np.einsum('iajaj->i', right_env)
        initial_state_gradient = 2 * np.einsum('i,i...->i...',
                                               overlap.conj(),
                                               initial_state_gradient).real
        gradient = 2 * np.einsum('i,i...->i...',
                                 overlap.conj(),
                                 gradient).real
        return initial_state_gradient, gradient, overlap
    
    def simple_gradient_ascent(self, target_Bs, target_Lambdas=None, lr=1e-3, iters=20):
        target_Bs = [B.conj() for B in target_Bs]
        if target_Lambdas is None:
            target_Lambdas = self.compute_Lambdas(target_Bs)
        overlaps = []
        # iterate
        for i in range(iters):
            init_grad, grad, overlap = self.gradient(target_Bs, target_Lambdas)
            overlaps.append(overlap)
            self.initial_state_params += lr * init_grad
            self.params += lr * grad
        Bs, Lambdas = self.fully_contract()
        overlaps.append(self._overlap(Bs, target_Bs))
        return overlaps, Bs, Lambdas
    
    def _initial_state_finite_diff_grad(self):
        grad = np.zeros((self.batch, self.L, 2, 1, self.d, 1),
                        dtype=np.complex128)
        eps = 1e-3
        
        def fourth_order_finite_diff(fp2, fp1, fm1, fm2, eps):
            return (-fp2 + 8 * fp1 - 8 * fm1 + fm2)/(12*eps)
        
        for param_idx in range(2):
            self.initial_state_params[..., param_idx] += eps
            Bs_plus, _ = self.initial_state()
            self.initial_state_params[..., param_idx] += eps
            Bs_plus2, _ = self.initial_state()
            self.initial_state_params[..., param_idx] -= 4*eps
            Bs_minus2, _ = self.initial_state()
            self.initial_state_params[..., param_idx] += eps
            Bs_minus, _ = self.initial_state()
            self.initial_state_params[..., param_idx] += eps
            for i in range(self.L):
                grad[:, i, param_idx] = fourth_order_finite_diff(
                    Bs_plus2[i], Bs_plus[i], Bs_minus[i], Bs_minus2[i], eps)
        return grad
    
    def _gate_finite_diff_grad(self):
        grad = np.zeros((self.batch, self.L-1, self.layers,
                         self.num_params, self.d**2, self.d**2),
                        dtype=np.complex128)
        eps = 1e-3
        
        def fourth_order_finite_diff(fp2, fp1, fm1, fm2, eps):
            return (-fp2 + 8 * fp1 - 8 * fm1 + fm2)/(12*eps)
        
        for param_idx in range(self.num_params):
            self.params[..., param_idx] += eps
            gate_plus = self.calculate_gate(self.params)
            self.params[..., param_idx] += eps
            gate_plus2 = self.calculate_gate(self.params)
            self.params[..., param_idx] -= 4*eps
            gate_minus2 = self.calculate_gate(self.params)
            self.params[..., param_idx] += eps
            gate_minus = self.calculate_gate(self.params)
            self.params[..., param_idx] += eps
            grad[..., param_idx, :, :] = fourth_order_finite_diff(
                gate_plus2, gate_plus, gate_minus, gate_minus2, eps)
        return grad
    
    def _overlap_finite_diff_grad(self, target_Bs):
        initial_gates_grad = np.zeros((self.batch, self.L, 2))
        grad = np.zeros((self.batch, self.L-1, self.layers, self.num_params))
        eps = 1e-3
        
        def fourth_order_finite_diff(fp2, fp1, fm1, fm2, eps):
            return (-fp2 + 8 * fp1 - 8 * fm1 + fm2)/(12*eps)
        
        for i in range(self.L):
            for param_idx in range(2):
                self.initial_state_params[:, i, param_idx] += eps
                Bs, _ = self.fully_contract()
                ovrlp_plus = np.abs(self._overlap(Bs, target_Bs))**2
                self.initial_state_params[:, i, param_idx] += eps
                Bs, _ = self.fully_contract()
                ovrlp_plus2 = np.abs(self._overlap(Bs, target_Bs))**2
                self.initial_state_params[:, i, param_idx] -= 4*eps
                Bs, _ = self.fully_contract()
                ovrlp_minus2 = np.abs(self._overlap(Bs, target_Bs))**2
                self.initial_state_params[:, i, param_idx] += eps
                Bs, _ = self.fully_contract()
                ovrlp_minus = np.abs(self._overlap(Bs, target_Bs))**2
                self.initial_state_params[:, i, param_idx] += eps
                initial_gates_grad[:, i, param_idx] = fourth_order_finite_diff(
                    ovrlp_plus2, ovrlp_plus, ovrlp_minus, ovrlp_minus2, eps)
        
        for i in range(self.L-1):
            for j in range(self.layers):
                for param_idx in range(self.num_params):
                    self.params[:, i, j, param_idx] += eps
                    Bs, _ = self.fully_contract()
                    ovrlp_plus = np.abs(self._overlap(Bs, target_Bs))**2
                    self.params[:, i, j, param_idx] += eps
                    Bs, _ = self.fully_contract()
                    ovrlp_plus2 = np.abs(self._overlap(Bs, target_Bs))**2
                    self.params[:, i, j, param_idx] -= 4*eps
                    Bs, _ = self.fully_contract()
                    ovrlp_minus2 = np.abs(self._overlap(Bs, target_Bs))**2
                    self.params[:, i, j, param_idx] += eps
                    Bs, _ = self.fully_contract()
                    ovrlp_minus = np.abs(self._overlap(Bs, target_Bs))**2
                    self.params[:, i, j, param_idx] += eps
                    grad[:, i, j, param_idx] = fourth_order_finite_diff(
                        ovrlp_plus2, ovrlp_plus, ovrlp_minus, ovrlp_minus2, eps)
        return initial_gates_grad, grad
    
    def _check_grads(self):
        # initial state
        print('\nInitial state:')
        fd_state, _ = self.initial_state()
        fd_grad = self._initial_state_finite_diff_grad()
        ex_state, _, ex_grad = self.initial_state_with_gradient()
        print('State diff. :',
              np.linalg.norm(np.array(fd_state) - np.array(ex_state)))
        print('Grad diff.:',
              np.linalg.norm(fd_grad - np.transpose(ex_grad, (1,0,2,3,4,5))))
        
        # gates and gate gradients
        print('\nGates:')
        fd_gates = self.calculate_gate(self.params)
        fd_grad = self._gate_finite_diff_grad()
        ex_gates, ex_grad = self.calculate_gate_with_gradient(self.params)
        print('Gates diff.:', np.linalg.norm(fd_gates - ex_gates))
        print('Grad diff. :', np.linalg.norm(fd_grad - ex_grad))
        
        # overlap and overlap gradient
        # create random MPS
        chi_max = 8 # maximal bond dimension of the MPS
        chis = [min(2**i, 2**(self.L-i), chi_max) for i in range(self.L+1)]
        Bs = [1. + np.random.normal(size=(self.batch, chis[i], 2, chis[i+1]))
              for i in range(self.L)]
        Bs, Lambdas = self.compute_Lambdas(Bs)
        target_Bs = [B.conj() for B in Bs]
        target_Lambdas = Lambdas
        print('\nOverlap:')
        Bs, _ = self.fully_contract()
        fd_overlap = self._overlap(Bs, target_Bs)
        fd_initial_grad, fd_grad = self._overlap_finite_diff_grad(target_Bs)
        ex_initial_grad, ex_grad, ex_overlap = self.gradient(target_Bs,
                                                             target_Lambdas)
        print('Overlap diff.         :',
              np.linalg.norm(fd_overlap - ex_overlap))
        print('Grad init. state diff.:',
              np.linalg.norm(fd_initial_grad - ex_initial_grad))
        print('Grad gates diff.      :',
              np.linalg.norm(fd_grad - ex_grad))

class gradient_three_cnot_sequential_circuit(gradient_sequential_circuit):
    
    def __init__(self, L, layers, batchsize=1, Lc=None, initial_gates=None):
        # number of parameters
        self.num_params = 9
        
        # initialize via base class
        super().__init__(L, layers, batchsize, Lc, initial_gates)
    
    def calculate_gate(self, params):
        # gate of the form:
        # |q0> --●--[Ry]-[X]-[Ry]--●--[SU(2)]-
        #        |        |        |          
        # |q1> -[X]-[Rz]--●--[Id]-[X]-[SU(2)]-
        
        shape = params.shape # ..., num_params
        params = params.reshape(-1, self.num_params)
        
        def rotation_gate(angles, pauli):
            return np.cos(angles/2)[:, None, None] * self.Id[None, :, :]\
            - 1j * np.sin(angles/2)[:, None, None] * pauli[None, :, :]
        
        def kron(A, B):
            return np.einsum('ijk,ilm->ijlkm', A, B)\
                .reshape(-1, self.d**2, self.d**2)
        
        # first layer of single-qubit rotations
        R1 = kron(rotation_gate(params[:, 0], self.Y),
                  rotation_gate(params[:, 1], self.Z))
        # second layer of single-qubit rotations
        R2 = kron(rotation_gate(params[:, 2], self.Y),
                  self.Id[None])
        # arbitrary single-qubit gates
        SU = kron(*[rotation_gate(params[:, 3*i+5], self.Z)\
                  @ rotation_gate(params[:, 3*i+4], self.Y)\
                  @ rotation_gate(params[:, 3*i+3], self.Z)\
                      for i in range(2)])
        # full gate with three CNOTs
        gate = SU @ self.CNOT @ R2 @ self.rCNOT @ R1 @ self.CNOT
        return gate.reshape(*shape[:-1], self.d**2, self.d**2)
    
    def calculate_gate_with_gradient(self, params):
        shape = params.shape # ..., num_params
        params = params.reshape(-1, self.num_params)
        
        def rotation_gate(angles, pauli):
            return np.cos(angles/2)[:, None, None] * self.Id[None, :, :]\
            - 1j * np.sin(angles/2)[:, None, None] * pauli[None, :, :]
        
        def kron(A, B):
            return np.einsum('ijk,ilm->ijlkm', A, B)\
                .reshape(-1, self.d**2, self.d**2)
        
        # gate with gradients
        gate_with_grad = np.zeros(
                    (params.shape[0], self.num_params+1, self.d**2, self.d**2),
                    dtype=np.complex128)
        # first CNOT
        gate_with_grad[..., :, :] = self.CNOT
        # first layer of single-qubit rotations
        gate_with_grad = kron(rotation_gate(params[:, 0], self.Y),
                              rotation_gate(params[:, 1], self.Z))[:, None]\
                       @ gate_with_grad
        gate_with_grad[..., 0+1, :, :] = np.kron(-1j*self.Y/2, self.Id)\
                                       @ gate_with_grad[..., 0+1, :, :]
        gate_with_grad[..., 1+1, :, :] = np.kron(self.Id, -1j*self.Z/2)\
                                       @ gate_with_grad[..., 1+1, :, :]
        # second CNOT gate
        gate_with_grad = self.rCNOT @ gate_with_grad
        # third layer of single-qubit rotations
        gate_with_grad = kron(rotation_gate(params[:, 2], self.Y),
                              self.Id[None])[:, None]\
                       @ gate_with_grad
        gate_with_grad[..., 2+1, :, :] = np.kron(-1j*self.Y/2, self.Id)\
                                       @ gate_with_grad[..., 2+1, :, :]
        # third CNOT gate
        gate_with_grad = self.CNOT @ gate_with_grad
        # arbitrary single-qubit gates
        for i, axis in enumerate([self.Z, self.Y, self.Z]):
            gate_with_grad = kron(rotation_gate(params[:, i+3], axis),
                                  rotation_gate(params[:, i+6], axis))[:, None]\
                           @ gate_with_grad
            gate_with_grad[..., i+3+1, :, :] = np.kron(-1j*axis/2, self.Id)\
                                           @ gate_with_grad[..., i+3+1, :, :]
            gate_with_grad[..., i+6+1, :, :] = np.kron(self.Id, -1j*axis/2)\
                                           @ gate_with_grad[..., i+6+1, :, :]
        gate = gate_with_grad[..., 0, :, :]\
                .reshape(*shape[:-1], self.d**2, self.d**2)
        grad = gate_with_grad[..., 1:, :, :]\
                .reshape(*shape, self.d**2, self.d**2)
        return gate, grad

class gradient_two_cnot_sequential_circuit(gradient_sequential_circuit):
    
    def __init__(self, L, layers, batchsize=1, Lc=None, initial_gates=None):
        # number of parameters
        self.num_params = 8
        
        # initialize via base class
        super().__init__(L, layers, batchsize, Lc, initial_gates)
    
    def calculate_gate(self, params):
        # gate of the form:
        # |q0> --●--[Ry]--●--[SU(2)]-
        #        |        |          
        # |q1> -[X]-[Ry]-[X]-[SU(2)]-
        
        shape = params.shape # ..., num_params
        params = params.reshape(-1, self.num_params)
        
        def rotation_gate(angles, pauli):
            return np.cos(angles/2)[:, None, None] * self.Id[None, :, :]\
            - 1j * np.sin(angles/2)[:, None, None] * pauli[None, :, :]
        
        def kron(A, B):
            return np.einsum('ijk,ilm->ijlkm', A, B)\
                .reshape(-1, self.d**2, self.d**2)
        
        # single-qubit y-rotations
        Ry = kron(rotation_gate(params[:, 0], self.Y),
                  rotation_gate(params[:, 1], self.Y))
        # arbitrary single-qubit gates
        SU = kron(*[rotation_gate(params[:, 3*i+4], self.Z)\
                  @ rotation_gate(params[:, 3*i+3], self.Y)\
                  @ rotation_gate(params[:, 3*i+2], self.Z)\
                      for i in range(2)])
        # full gate with two CNOTs
        gate = SU @ self.CNOT @ Ry @ self.CNOT
        return gate.reshape(*shape[:-1], self.d**2, self.d**2)
    
    def calculate_gate_with_gradient(self, params):
        shape = params.shape # ..., num_params
        params = params.reshape(-1, self.num_params)
        
        def rotation_gate(angles, pauli):
            return np.cos(angles/2)[:, None, None] * self.Id[None, :, :]\
            - 1j * np.sin(angles/2)[:, None, None] * pauli[None, :, :]
        
        def kron(A, B):
            return np.einsum('ijk,ilm->ijlkm', A, B)\
                .reshape(-1, self.d**2, self.d**2)
        
        # gate with gradients
        gate_with_grad = np.zeros(
                    (params.shape[0], self.num_params+1, self.d**2, self.d**2),
                    dtype=np.complex128)
        # first CNOT
        gate_with_grad[..., :, :] = self.CNOT
        # single-qubit y-rotations
        gate_with_grad = kron(rotation_gate(params[:, 0], self.Y),
                              rotation_gate(params[:, 1], self.Y))[:, None]\
                       @ gate_with_grad
        gate_with_grad[..., 0+1, :, :] = np.kron(-1j*self.Y/2, self.Id)\
                                       @ gate_with_grad[..., 0+1, :, :]
        gate_with_grad[..., 1+1, :, :] = np.kron(self.Id, -1j*self.Y/2)\
                                       @ gate_with_grad[..., 1+1, :, :]
        # second CNOT gate
        gate_with_grad = self.CNOT @ gate_with_grad
        # arbitrary single-qubit gates
        for i, axis in enumerate([self.Z, self.Y, self.Z]):
            gate_with_grad = kron(rotation_gate(params[:, i+2], axis),
                                  rotation_gate(params[:, i+5], axis))[:, None]\
                           @ gate_with_grad
            gate_with_grad[..., i+2+1, :, :] = np.kron(-1j*axis/2, self.Id)\
                                           @ gate_with_grad[..., i+2+1, :, :]
            gate_with_grad[..., i+5+1, :, :] = np.kron(self.Id, -1j*axis/2)\
                                           @ gate_with_grad[..., i+5+1, :, :]
        gate = gate_with_grad[..., 0, :, :]\
                .reshape(*shape[:-1], self.d**2, self.d**2)
        grad = gate_with_grad[..., 1:, :, :]\
                .reshape(*shape, self.d**2, self.d**2)
        return gate, grad

class gradient_single_cnot_sequential_circuit(gradient_sequential_circuit):
    
    def __init__(self, L, layers, batchsize=1, Lc=None, initial_gates=None):
        # number of parameters
        self.num_params = 6
        
        # initialize via base class
        super().__init__(L, layers, batchsize, Lc, initial_gates)
    
    def calculate_gate(self, params):
        # gate of the form:
        # |q0> --●--[SU(2)]-
        #        |          
        # |q1> -[X]-[SU(2)]-
        
        shape = params.shape # ..., num_params
        params = params.reshape(-1, self.num_params)
        
        def rotation_gate(angles, pauli):
            return np.cos(angles/2)[:, None, None] * self.Id[None, :, :]\
            - 1j * np.sin(angles/2)[:, None, None] * pauli[None, :, :]
        
        def kron(A, B):
            return np.einsum('ijk,ilm->ijlkm', A, B)\
                .reshape(-1, self.d**2, self.d**2)
        
        # arbitrary single-qubit gates
        SU = kron(*[rotation_gate(params[:, 3*i+2], self.Z)\
                  @ rotation_gate(params[:, 3*i+1], self.Y)\
                  @ rotation_gate(params[:, 3*i+0], self.Z)\
                      for i in range(2)])
        # full gate with two CNOTs
        gate = SU @ self.CNOT
        return gate.reshape(*shape[:-1], self.d**2, self.d**2)
    
    def calculate_gate_with_gradient(self, params):
        shape = params.shape # ..., num_params
        params = params.reshape(-1, self.num_params)
        
        def rotation_gate(angles, pauli):
            return np.cos(angles/2)[:, None, None] * self.Id[None, :, :]\
            - 1j * np.sin(angles/2)[:, None, None] * pauli[None, :, :]
        
        def kron(A, B):
            return np.einsum('ijk,ilm->ijlkm', A, B)\
                .reshape(-1, self.d**2, self.d**2)
        
        # gate with gradients
        gate_with_grad = np.zeros(
                    (params.shape[0], self.num_params+1, self.d**2, self.d**2),
                    dtype=np.complex128)
        # first CNOT
        gate_with_grad[..., :, :] = self.CNOT
        # arbitrary single-qubit gates
        for i, axis in enumerate([self.Z, self.Y, self.Z]):
            gate_with_grad = kron(rotation_gate(params[:, i+0], axis),
                                  rotation_gate(params[:, i+3], axis))[:, None]\
                           @ gate_with_grad
            gate_with_grad[..., i+0+1, :, :] = np.kron(-1j*axis/2, self.Id)\
                                           @ gate_with_grad[..., i+0+1, :, :]
            gate_with_grad[..., i+3+1, :, :] = np.kron(self.Id, -1j*axis/2)\
                                           @ gate_with_grad[..., i+3+1, :, :]
        gate = gate_with_grad[..., 0, :, :]\
                .reshape(*shape[:-1], self.d**2, self.d**2)
        grad = gate_with_grad[..., 1:, :, :]\
                .reshape(*shape, self.d**2, self.d**2)
        return gate, grad