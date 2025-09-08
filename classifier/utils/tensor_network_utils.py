import numpy as np
from jax import jit
import jax.numpy as jnp
from tqdm import tqdm

def select_first_n_samples_per_class(A_tensors, labels, n_samples_per_class=10):
    selected_indices = []
    for class_label in np.unique(labels):
        class_indices = np.where(labels == class_label)[0][:n_samples_per_class]
        selected_indices.extend(class_indices)
    A_tensors = [A[selected_indices] for A in A_tensors]
    return A_tensors, labels[selected_indices]


def create_balanced_batches(A_tensors, y, batch_size, shuffle=True):
    """
    Args:
        A_tensors (list of np.ndarray or torch.Tensor): List of feature arrays/tensors.
            Each element must have shape (N, ...) where N is the total number of samples.
        y (np.ndarray or torch.Tensor): Label array/tensor of shape (N,).
        batch_size (int): Total number of samples in each batch.
        shuffle (bool): Whether to shuffle indices within each class before batching.
    
    Returns:
        list of tuples: Each element is (A_batch, y_batch) with the desired balanced shape.
    """

    y = np.asarray(y)

    classes = np.unique(y)
    n_classes = len(classes)

    if batch_size % n_classes != 0:
        raise ValueError("batch_size must be divisible by the number of classes.")

    samples_per_class = batch_size // n_classes

    class_indices = {}
    for c in classes:
        class_indices[c] = np.where(y == c)[0]
        if shuffle:
            np.random.shuffle(class_indices[c])

    n_batches = min(len(idx_list) for idx_list in class_indices.values()) // samples_per_class

    # Create the list of batches
    batches = []
    for i in range(n_batches):
        batch_indices = []
        # For each class, take the i-th chunk of size `samples_per_class`
        for c in classes:
            start = i * samples_per_class
            end = (i + 1) * samples_per_class
            batch_indices.extend(class_indices[c][start:end])

        # Convert batch_indices to an array (often better for slicing)
        batch_indices = np.array(batch_indices)

        # Slice each A_tensors element
        A_batch = [A[batch_indices] for A in A_tensors]
        y_batch = y[batch_indices]

        # Append (A_batch, y_batch) to the final list
        batches.append((A_batch, y_batch))

    return batches


@jit
def svd_jit(A):
    # A must be a jnp.ndarray
    # shape: (m, n)
    U, S, V = jnp.linalg.svd(A, full_matrices=False)
    return U, S, V

def left_isometric(Ms, normalize=True):
    As = []
    R = jnp.ones((1, 1))
    for M in Ms:
        A = jnp.einsum('ab, bjc -> ajc', R, M)
        chil, d, chir = A.shape
        A = A.reshape(chil * d, chir)
        Q, R = jnp.linalg.qr(A)
        As.append(Q.reshape(chil, d, -1))
    if not normalize:
        As[-1] *= jnp.squeeze(R)
    else:
        As[-1] *= jnp.sign(jnp.squeeze(R))
    return As

def right_isometric(Ms, normalize=True):
    Bs = []
    Lt = jnp.ones((1, 1))
    for M in Ms[::-1]:
        B = jnp.einsum('ajb, cb -> ajc', M, Lt)
        chil, d, chir = B.shape
        B = B.reshape(chil, d * chir)
        Qt, Lt = jnp.linalg.qr(B.T)
        Bs.append(Qt.T.reshape(-1, d, chir))
    if not normalize:
        Bs[-1] *= jnp.squeeze(Lt)
    else:
        Bs[-1] *= jnp.sign(jnp.squeeze(Lt))
    return Bs[::-1]

def truncate(Ms, chi_max=10, svd_min=1e-8, normalize=True, which='right_to_left'):
    if which == 'right_to_left':
        # sweeps right to left
        # assumes left-canonical form, returns right-canonical form
        for i in range(len(Ms)-1, 0, -1):
            # two-site tensor
            M0 = Ms[i-1]
            M1 = Ms[i]
            theta = jnp.einsum('aib, bjc -> aijc', M0, M1)
            chil, d1, d2, chir = theta.shape
            theta = theta.reshape(chil * d1, d2 * chir)
            # singular value decomposition
            X, S, Y = svd_jit(theta)
            # new bond dimension
            cutoff = max(1, jnp.linalg.norm(S)) * svd_min
            chi_new = min(jnp.sum(S > cutoff), chi_max)
            # normalize truncated state
            S = S[:chi_new]
            if normalize:
                S /= jnp.linalg.norm(S[:chi_new])
            # truncate and save
            Ms[i-1] = (X[:, :chi_new] * S[None, :])\
                .reshape(chil, d1, chi_new)
            Ms[i] = Y[:chi_new]\
                .reshape(chi_new, d2, chir)
    elif which == 'left_to_right':
        # sweeps left to right
        # assumes right-canonical form, returns left-canonical form
        for i in range(len(As)-1):
            # two-site tensor
            M0 = Ms[i]
            M1 = Ms[i+1]
            theta = jnp.einsum('aib, bjc -> aijc', M0, M1)
            chil, d1, d2, chir = theta.shape
            theta = theta.reshape(chil * d1, d2 * chir)
            # singular value decomposition
            X, S, Y = svd_jit(theta)
            # new bond dimension
            cutoff = max(1, jnp.linalg.norm(S)) * svd_min
            chi_new = min(jnp.sum(S > cutoff), chi_max)
            # normalize truncated state
            S = S[:chi_new]
            if normalize:
                S /= jnp.linalg.norm(S[:chi_new])
            # truncate and save
            Ms[i] = X[:, :chi_new]\
                .reshape(chil, d1, chi_new)
            Ms[i+1] = (S[:, None] * Y[:chi_new])\
                .reshape(chi_new, d2, chir)
    else:
        raise ValueError("Wrong argument 'which' given for 'truncate'.")
    return Ms

def truncate_batched(As, chi_max=2, which='right_to_left', d=2, svd_min=1e-8):
    batch = As[0].shape[0]
    if which == 'right_to_left':
        # sweeps right to left
        # assumes left-canonical form, returns right-canonical form
        for i in range(len(As)-1, 0, -1):
            # two-site tensor
            A0 = As[i-1]
            A1 = As[i]
            theta = jnp.einsum('iajb,ibkc->iajkc', A0, A1)
            _, chil, _, _, chir = theta.shape
            theta = theta.reshape(batch, chil * d, d * chir)
            # singular value decomposition
            X, S, Y = svd_jit(theta)
            # new bond dimension
            chi_new = min(S.shape[1],
                            max(1, max(jnp.sum(S > svd_min, axis=1))),
                            chi_max)
            # normalize truncated state
            S = S[:, :chi_new]\
                / jnp.sqrt(jnp.sum(S[:, :chi_new]**2, axis=1))[:, None]
            # truncate and save
            As[i-1] = (X[:, :, :chi_new] * S[:, None, :])\
                .reshape(batch, -1, d, chi_new)
            As[i] = Y[:, :chi_new, :]\
                .reshape(batch, chi_new, d, -1)
    elif which == 'left_to_right':
        # sweeps left to right
        # assumes right-canonical form, returns left-canonical form
        for i in range(len(As)-1):
            # two-site tensor
            A0 = As[i]
            A1 = As[i+1]
            theta = jnp.einsum('iajb,ibkc->iajkc', A0, A1)
            _, chil, _, _, chir = theta.shape
            theta = theta.reshape(batch, chil * d, d * chir)
            # singular value decomposition
            X, S, Y = svd_jit(theta)
            # new bond dimension
            chi_new = min(S.shape[1],
                            max(1, max(jnp.sum(S > svd_min, axis=1))),
                            chi_max)
            # normalize truncated state
            S = S[:, :chi_new]\
                / jnp.sqrt(jnp.sum(S[:, :chi_new]**2, axis=1))[:, None]
            # truncate and save
            As[i] = X[:, :, :chi_new]\
                .reshape(batch, -1, d, chi_new)
            As[i+1] = (S[:, :, None] * Y[:, :chi_new, :])\
                .reshape(batch, chi_new, d, -1)
    else:
        raise ValueError("Wrong argument 'which' given for 'truncate'.")
    return As

@jit
def double_MPS_tensor(M):
    """
    Given M of shape (batch, chiL, d, chiR),
    return shape (batch, chiL^2, d^2, chiR^2).
    """
    batch, chiL, d, chiR = M.shape
    # Typical doubling: M -> M * conj(M), merging indices
    M_doubled = jnp.einsum('iajc, ibkd -> iabjkcd', M, M.conj())
    return M_doubled.reshape(batch, chiL**2, d**2, chiR**2)

def double_and_truncate_batched(As, chi_max=2, which='right_to_left', d=2, svd_min=1e-8):
    """
    Incrementally double an MPS from either the rightmost or leftmost end,
    and perform a two-site SVD-based truncation at each step.
    
    Args:
      As: list of MPS tensors, each shape (batch, chiL, d, chiR).
      chi_max: Max bond dimension after truncation.
      which: "right_to_left" or "left_to_right".
      d: The original single-site (physical) dimension.
      svd_min: Threshold for singular values in truncation.

    Returns:
      The MPS list `As`, where each site has been doubled exactly once
      (final shape ~ (chi^2, d^2, chi^2)) and bond dimension truncated
      according to `chi_max`.
    """
    batch = As[0].shape[0]
    L = len(As)

    if which == 'right_to_left':
        # -------------------------------------------------
        # 1) Double the rightmost site exactly once
        # -------------------------------------------------
        As[L-1] = double_MPS_tensor(As[L-1])  # shape => (batch, chi^2, d^2, chi^2)

        # -------------------------------------------------
        # 2) Sweep from i = L-2 down to i = 0
        #    For each i, double As[i], then contract with As[i+1].
        # -------------------------------------------------
        for i in range(L-2, -1, -1):
            # Double the newly visited site As[i], which is not yet doubled
            As[i] = double_MPS_tensor(As[i])   # now shape => (batch, chi^2, d^2, chi^2)

            # Form two-site block 
            # shape => (batch, chi^2, d^2, d^2, chi^2)
            theta = jnp.einsum('iajb, ibkc->iajkc', As[i], As[i+1])

            # Reshape for SVD
            #   let _ = batch, chil = chi^2, [2nd dim] = d^2, [3rd dim] = d^2, chir = chi^2
            _, chil, _, _, chir = theta.shape
            # interpret each d^2 as the new "physical dimension"
            # so we group (chil * d) and (d * chir),
            # but remember that your new 'd' is actually d^2.
            # For a simpler approach, we do:
            theta = theta.reshape(batch, chil * d, d * chir)
            # shape => (batch, (chi^2)*(d^2), (d^2)*(chi^2))

            # SVD
            X, S, Y = svd_jit(theta)

            # Determine new bond dimension
            chi_new = min(
                S.shape[1],
                max(1, jnp.max(jnp.sum(S > svd_min, axis=1))),
                chi_max
            )
            # Normalize truncated state
            S_trunc = S[:, :chi_new]
            norms = jnp.sqrt(jnp.sum(S_trunc**2, axis=1, keepdims=True))
            S_trunc = S_trunc / norms

            # Truncate and reshape
            # X => (batch, (chil*d), chi_new)
            X_trunc = X[:, :, :chi_new]  # (batch, chil*d, chi_new)
            As[i] = (X_trunc * S_trunc[:, None, :])\
                     .reshape(batch, -1, d, chi_new)

            # Y => (batch, chi_new, (d*chir))
            Y_trunc = Y[:, :chi_new, :]
            As[i+1] = Y_trunc.reshape(batch, chi_new, d, -1)

    elif which == 'left_to_right':
        # -------------------------------------------------
        # 1) Double the leftmost site exactly once
        # -------------------------------------------------
        As[0] = double_MPS_tensor(As[0])  # shape => (batch, chi^2, d^2, chi^2)

        # -------------------------------------------------
        # 2) Sweep from i = 0 to i = L-2
        #    For each i, double As[i+1], then contract with As[i].
        # -------------------------------------------------
        for i in range(L-1):
            # Double the newly visited site: As[i+1]
            if i+1 < L:
                As[i+1] = double_MPS_tensor(As[i+1])

            theta = jnp.einsum('iajb, ibkc->iajkc', As[i], As[i+1])

            _, chil, _, _, chir = theta.shape
            theta = theta.reshape(batch, chil * d, d * chir)

            X, S, Y = svd_jit(theta)

            chi_new = min(
                S.shape[1],
                max(1, jnp.max(jnp.sum(S > svd_min, axis=1))),
                chi_max
            )
            S_trunc = S[:, :chi_new]
            norms = jnp.sqrt(jnp.sum(S_trunc**2, axis=1, keepdims=True))
            S_trunc = S_trunc / norms

            X_trunc = X[:, :, :chi_new]
            As[i] = X_trunc.reshape(batch, -1, d, chi_new)

            Y_trunc = Y[:, :chi_new, :]
            As[i+1] = (S_trunc[:, :, None] * Y_trunc)\
                          .reshape(batch, chi_new, d, -1)

    else:
        raise ValueError("Wrong argument 'which' given for 'truncate'.")

    return As



def MPS_addition(M1s, M2s, chi_max=100, svd_min=1e-8, normalize=True):
    Ms = []
    # left edge
    _, d, chi1r = M1s[0].shape
    _, _, chi2r = M2s[0].shape
    M = jnp.zeros((1, d, chi1r+chi2r))
    M = M.at[:, :, :chi1r].set(M1s[0])
    M = M.at[:, :, chi1r:].set(M2s[0])
    Ms.append(M)
    # bulk tensors
    for M1, M2 in zip(M1s[1:-1], M2s[1:-1]):
        chi1l, d, chi1r = M1.shape
        chi2l, _, chi2r = M2.shape
        M = jnp.zeros((chi1l+chi2l, d, chi1r+chi2r))
        M = M.at[:chi1l, :, :chi1r].set(M1)
        M = M.at[chi1l:, :, chi1r:].set(M2)
        Ms.append(M)
    # right edge
    chi1l, d, _ = M1s[-1].shape
    chi2l, _, _ = M2s[-1].shape
    M = jnp.zeros((chi1l+chi2l, d, 1))
    M = M.at[:chi1l].set(M1s[-1])
    M = M.at[chi1l:].set(M2s[-1])
    Ms.append(M)
    # compress MPS
    Ms = left_isometric(Ms,
                        normalize=normalize)
    Ms = truncate(Ms,
                  chi_max=chi_max,
                  svd_min=svd_min,
                  normalize=normalize,
                  which='right_to_left')
    return Ms

def MPS_batch_addition(Ms, chi_max=100, svd_min=1e-8, normalize=True):
    # left edge
    batch, _, d, chir = Ms[0].shape
    Ms[0] = Ms[0].transpose(1, 2, 0, 3)\
                 .reshape(1, d, batch * chir)
    # bulk tensors
    for i, M in enumerate(Ms[1:-1], 1):
        batch, chil, d, chir = M.shape
        Mnew = jnp.zeros((batch * chil, d, batch * chir))
        for j in range(batch):
            Mnew = Mnew.at[j*chil:(j+1)*chil, :, j*chir:(j+1)*chir].set(M[j])
            # Mnew[j*chil:(j+1)*chil, :, j*chir:(j+1)*chir] = M[j]
        Ms[i] = Mnew
    # right edge
    batch, chil, d, _ = Ms[-1].shape
    Ms[-1] = Ms[-1].reshape(batch * chil, d, 1)
    # compress MPS
    Ms = left_isometric(Ms,
                        normalize=normalize)
    Ms = truncate(Ms,
                  chi_max=chi_max,
                  svd_min=svd_min,
                  normalize=normalize,
                  which='right_to_left')
    return Ms

def MPS_pretraining(dataset, Lc=None, batch=100, num_classes=10, chi_max=100, chi_final=32, svd_min=1e-8):
    # first stack batches of MPS for each class
    print('Combining batches of input states...', flush=True)
    summed_Ms = [[] for k in range(num_classes)]
    for states, labels in tqdm(dataset, desc="Processing batches"):
        for k in range(num_classes):
            idxs = (np.array(labels) == k)
            if np.sum(idxs) > 0:

                Ms = [M[idxs] for M in states]
                Ms = truncate_batched(Ms,
                            chi_max=100,
                            svd_min=svd_min,
                            which='right_to_left')

                summed_Ms[k].append(MPS_batch_addition(Ms,
                                                       chi_max=chi_max,
                                                       svd_min=svd_min,
                                                       normalize=False))
    # for each class, sum consecutive MPS
    print('Summing up remaining states pairwise...', flush=True)
    for k in range(num_classes):
        num_iters = jnp.ceil(jnp.log2(len(summed_Ms[k]))).astype(jnp.int32)
        for j in range(num_iters):
            Ms = summed_Ms[k]
            temp_Ms = []
            for i in range(len(Ms)//2):
                temp_Ms.append(MPS_addition(Ms[2*i],
                                            Ms[2*i+1],
                                            chi_max=chi_max,
                                            svd_min=svd_min,
                                            normalize=False))
            if len(Ms)%2 == 1:
                temp_Ms.append(Ms[-1])
            summed_Ms[k] = temp_Ms
    # insert class label into MPS on site Lc
    print('Adding leg for class labels to MPS...', flush=True)

    if Lc is None:
        Lc = len(summed_Ms[0][0])//2
    for k in range(num_classes):
        Ms = summed_Ms[k][0]
        chil, d, chir = Ms[Lc].shape
        M = jnp.zeros((chil, num_classes, chil))
        M = M.at[:, k, :].set(jnp.eye(chil))
        # M[:, k, :] = jnp.eye(chil)
        # uncomment the following line and comment the two after for extra tensor for classification
        # Ms.insert(Lc, M)
        Ms[Lc] = jnp.einsum('akb, bjc -> akjc', M, Ms[Lc])\
                     .reshape(chil, num_classes * d, chir)
        summed_Ms[k] = Ms
    # scale states by common factor to avoid precision loss
    scale = max([jnp.abs(summed_Ms[k][0]).max() for k in range(num_classes)])
    for k in range(num_classes):
        summed_Ms[k][0] /= scale
    # sum over classes
    print('Sum pairwise over different classes...', flush=True)
    num_iters = jnp.ceil(jnp.log2(num_classes)).astype(jnp.int32)
    for j in range(num_iters-1):
        temp_Ms = []
        for i in range(len(summed_Ms)//2):
            temp_Ms.append(MPS_addition(summed_Ms[2*i],
                                        summed_Ms[2*i+1],
                                        chi_max=chi_max,
                                        svd_min=svd_min,
                                        normalize=False))
        if len(summed_Ms)%2 == 1:
            temp_Ms.append(summed_Ms[-1])
        summed_Ms = temp_Ms
    # final summation combined with truncation to final bond dimension
    print('Final sum, normalize and rescale...', flush=True)
    summed_Ms = MPS_addition(summed_Ms[0],
                             summed_Ms[1],
                             chi_max=chi_final,
                             svd_min=svd_min,
                             normalize=True)
    # renormalize such that total norm is prod_i=1^L d_i
    summed_Ms = [jnp.sqrt(M.shape[1]) * M for M in summed_Ms]
    # fix physical and label dimension on site Lc
    chil, d, chir = summed_Ms[Lc].shape
    summed_Ms[Lc] = summed_Ms[Lc]\
                    .reshape(chil, num_classes, d//num_classes, chir)\
                    .transpose(1, 0, 2, 3)
    return summed_Ms


def MPO_pretraining(dataset, Lc=None, batch=100, num_classes=10, chi_max=100, chi_final=32, svd_min=1e-8):
    # first stack batches of MPS for each class
    print('Combining batches of input states...', flush=True)
    summed_Ms = [[] for k in range(num_classes)]
    for states, labels in tqdm(dataset, desc="Processing batches"):
        for k in range(num_classes):
            idxs = (np.array(labels) == k)
            if np.sum(idxs) > 0:

                Ms = [M[idxs] for M in states]
                Ms = truncate_batched(Ms,
                            chi_max=100,
                            svd_min=svd_min,
                            which='right_to_left')

                Ms = double_and_truncate_batched(Ms,
                            chi_max=256,
                            svd_min=svd_min,
                            which='left_to_right',
                            d=4)

                summed_Ms[k].append(MPS_batch_addition(Ms,
                                                       chi_max=chi_max,
                                                       svd_min=svd_min,
                                                       normalize=False))
    # for each class, sum consecutive MPS
    print('Summing up remaining states pairwise...', flush=True)
    for k in range(num_classes):
        num_iters = jnp.ceil(jnp.log2(len(summed_Ms[k]))).astype(jnp.int32)
        for j in range(num_iters):
            Ms = summed_Ms[k]
            temp_Ms = []
            for i in range(len(Ms)//2):
                temp_Ms.append(MPS_addition(Ms[2*i],
                                            Ms[2*i+1],
                                            chi_max=chi_max,
                                            svd_min=svd_min,
                                            normalize=False))
            if len(Ms)%2 == 1:
                temp_Ms.append(Ms[-1])
            summed_Ms[k] = temp_Ms
    # insert class label into MPS on site Lc
    print('Adding leg for class labels to MPS...', flush=True)

    if Lc is None:
        Lc = len(summed_Ms[0][0])//2
    for k in range(num_classes):
        Ms = summed_Ms[k][0]
        chil, d, chir = Ms[Lc].shape
        M = jnp.zeros((chil, num_classes, chil))
        M = M.at[:, k, :].set(jnp.eye(chil))
        # M[:, k, :] = jnp.eye(chil)
        # uncomment the following line and comment the two after for extra tensor for classification
        # Ms.insert(Lc, M)
        Ms[Lc] = jnp.einsum('akb, bjc -> akjc', M, Ms[Lc])\
                     .reshape(chil, num_classes * d, chir)
        summed_Ms[k] = Ms
    # scale states by common factor to avoid precision loss
    scale = max([jnp.abs(summed_Ms[k][0]).max() for k in range(num_classes)])
    for k in range(num_classes):
        summed_Ms[k][0] /= scale
    # sum over classes
    print('Sum pairwise over different classes...', flush=True)
    num_iters = jnp.ceil(jnp.log2(num_classes)).astype(jnp.int32)
    for j in range(num_iters-1):
        temp_Ms = []
        for i in range(len(summed_Ms)//2):
            temp_Ms.append(MPS_addition(summed_Ms[2*i],
                                        summed_Ms[2*i+1],
                                        chi_max=chi_max,
                                        svd_min=svd_min,
                                        normalize=False))
        if len(summed_Ms)%2 == 1:
            temp_Ms.append(summed_Ms[-1])
        summed_Ms = temp_Ms
    # final summation combined with truncation to final bond dimension
    print('Final sum, normalize and rescale...', flush=True)
    summed_Ms = MPS_addition(summed_Ms[0],
                             summed_Ms[1],
                             chi_max=chi_final,
                             svd_min=svd_min,
                             normalize=True)
    # renormalize such that total norm is prod_i=1^L d_i
    summed_Ms = [jnp.sqrt(M.shape[1]) * M for M in summed_Ms]
    # fix physical and label dimension on site Lc
    chil, d, chir = summed_Ms[Lc].shape
    summed_Ms[Lc] = summed_Ms[Lc]\
                    .reshape(chil, num_classes, d//num_classes, chir)\
                    .transpose(1, 0, 2, 3)
    # unpack physical dimensions to two physical MPO legs
    def reshape_to_MPO_tensor(M):
        shape = M.shape
        d = int(jnp.sqrt(shape[-2]))
        return M.reshape(*shape[:-2], d, d, shape[-1])
    summed_Ms = [reshape_to_MPO_tensor(M) for M in summed_Ms]
    return summed_Ms
