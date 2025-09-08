import pickle
import numpy as np

def calc_MPS(states, chi_max=256, normalize=False, d=2):
    """
    Calculate MPS tensors from d^L state amplitudes.
    
    """
    
    if len(states.shape) > 1:
        batchsize = states.shape[0]
    else:
        batchsize = 1
    states = states.reshape(batchsize, -1)
    L = states.shape[-1]
    L = int(np.log(L)/np.log(d))
    
    A_tensors = []
    Lambda_tensors = []
    
    psi_rest = states.reshape(batchsize, 1, -1)
    
    for bond in range(1, L):
        # bond dimensions of previous step
        _, chi, dim_rest = psi_rest.shape
        assert dim_rest == d**(L-bond+1)
        
        # move cut by one site and reshape wave function
        psi_rest = psi_rest.reshape(batchsize, chi*d, dim_rest//d)
        
        # perform SVD
        A, Lambda, psi_rest = np.linalg.svd(psi_rest, full_matrices=False)
        
        # and truncate
        chi_new = min(A.shape[-1], chi_max)
        
        A_tensors.append(A[:,:,:chi_new].reshape(batchsize, chi, d, chi_new))
        Lambda = Lambda[:,:chi_new]
        if normalize:
            Lambda /= np.sqrt(np.sum(Lambda**2, axis=1))[:,None]
        Lambda_tensors.append(Lambda)
        psi_rest = psi_rest[:,:chi_new,:]
        
        # multiply Schmidt values to wave function
        psi_rest = Lambda[:,:,None] * psi_rest
    
    # save last MPS tensor
    A_tensors.append(psi_rest.reshape(batchsize, chi_new, d, 1))
    
    return A_tensors, Lambda_tensors

def calc_entropy(Lambda_tensors):
    """
    Calculate von Neumann entanglement entropy per bond
    from list of Schmdit values per bond.
    
    """
    
    entropy = []
    for bond in Lambda_tensors:
        bond = bond**2/np.sum(bond**2, axis=1)[:,None]
        bond[bond < 1e-30] = 1e-30
        entropy.append(-np.sum(bond * np.log(bond), axis=1))
    return np.array(entropy) # [num_bonds, batchsize]

def calc_state(A_list, renormalize=True):
    """
    Calculate full state, i.e., all 2^L state amplitudes, from MPS.
    
    """
    
    states = A_list[0]
    batchsize = states.shape[0]
    for A in A_list[1:]:
        states = np.einsum(states, [*np.arange(len(states.shape))],#         0, 1, ..., l-1
                           A, [0, *(np.arange(3) + len(states.shape)) - 1],# 0, l-1, l, l+1
                           [*np.arange(len(states.shape)-1), *(np.arange(2)+len(states.shape))])
    states = states.reshape(batchsize, -1)
    if renormalize:
        states /= np.sqrt(np.einsum('ij,ij->i', states.conj(), states))[:,None]
    return states

def right_canonical(As):
    batchsize = As[0].shape[0]
    Bs = []
    Lt = np.ones((batchsize, 1, 1))
    for A in As[::-1]:
        _, chil, d, chir = A.shape
        A = np.einsum('iajb,icb->iajc', A, Lt)\
            .reshape(batchsize, chil, d * chir)
        Qt, Lt = np.linalg.qr(A.transpose(0,2,1))
        Bs.append(Qt.transpose(0,2,1).reshape(batchsize, chil, d, chir))
    return Bs[::-1]