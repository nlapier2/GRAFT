import numpy as np
from sklearn.isotonic import IsotonicRegression
from utils import *

def fit_isotonic_on_pairs(S_ext_OO: np.ndarray, Y_O: np.ndarray) -> "IsotonicRegression|None":
    """
    Fit isotonic regression mapping external similarity -> target similarity,
    using only training perts (O). Returns a fitted IsotonicRegression or None.
    """
    # target similarity among O (using correlation-style similarity of DELTAS)
    Zt = row_standardize(Y_O)  # (|O|, G)
    S_tgt_OO = Zt @ Zt.T
    # take off-diagonal upper triangle pairs
    iu, ju = np.triu_indices(S_ext_OO.shape[0], k=1)
    x = S_ext_OO[iu, ju].astype(np.float64)
    y = S_tgt_OO[iu, ju].astype(np.float64)
    # Guard against degenerate cases
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        print("[iso] Degenerate pairwise similarities; skipping isotonic calibration.")
        return None
    iso = IsotonicRegression(y_min=-1.0, y_max=1.0, increasing=True, out_of_bounds="clip")
    iso.fit(x, y)
    return iso

def apply_isotonic_matrix(iso: "IsotonicRegression|None", S: np.ndarray) -> np.ndarray:
    """Apply fitted isotonic regressor elementwise to a similarity matrix; symmetrize and fix diag."""
    if iso is None:
        return S
    S_flat = S.ravel()
    S_cal = iso.predict(S_flat).reshape(S.shape)
    S_cal = 0.5 * (S_cal + S_cal.T)
    np.fill_diagonal(S_cal, 1.0)
    return S_cal

def sharpen_neighbors(K_UO: np.ndarray, tau: float = 1.0, topk: int = 0) -> np.ndarray:
    """
    A4: Neighbor sharpening. Elementwise power on similarities then optional top-k per row.
    Operates ONLY on K_UO (cross block) to avoid changing the fit on O.
    """
    if tau <= 1.0 and (topk is None or topk <= 0):
        return K_UO
    Kp = np.maximum(K_UO, 0.0).astype(np.float32)
    if tau > 1.0:
        Kp = np.power(Kp, tau, dtype=np.float32)
    if topk and topk > 0:
        topk = min(topk, Kp.shape[1])
        # threshold each row to its k-th largest value
        part = np.partition(Kp, Kp.shape[1] - topk, axis=1)
        thresh = part[:, Kp.shape[1] - topk : Kp.shape[1] - topk + 1]
        Kp[Kp < thresh] = 0.0
    Kp_sum = Kp.sum(axis=1, keepdims=True) + 1e-8
    Kp /= Kp_sum
    return Kp

def subspace_boost(Y_U_hat_delta: np.ndarray, Y_O_delta: np.ndarray, k: int, gamma: float) -> np.ndarray:
    """
    A3: Subspace boosting in gene space. Boost components along the top-k PCs
    computed from training deltas Y_O (|O| x G). Uses SVD to avoid extra deps.
    """
    if k <= 0 or gamma <= 0:
        return Y_U_hat_delta
    k = min(k, min(Y_O_delta.shape[0], Y_O_delta.shape[1]))
    # Center across perts before SVD to focus on between-pert variation
    Yc = Y_O_delta - Y_O_delta.mean(axis=0, keepdims=True)
    # thin SVD: Yc = U S Vt ; Vt is (G x G) truncated to k
    try:
        U, S, Vt = np.linalg.svd(Yc, full_matrices=False)
    except np.linalg.LinAlgError:
        return Y_U_hat_delta  # fall back safely if SVD fails
    P = Vt[:k, :].T  # G x k
    proj = (Y_U_hat_delta @ P) @ P.T  # project into top-k subspace
    boosted = Y_U_hat_delta + gamma * proj
    return boosted

def sharpen_effects(
    pred_mat: np.ndarray,
    ctrl_mean: np.ndarray,
    mode: str,
    gamma: float = 1.5,
    topk_frac: float = 0.1,
    alpha: float = 0.3,
    beta: float = 0.2,
    sigmoid_B: float = 0.7,
    preserve_q: float = 0.95,
):
    """
    Post-process predicted effects Δ = pred - ctrl, apply a monotone sharpening, then
    reconstruct predictions pred' = ctrl + Δ'. Operates in-place on a copy and returns it.
    """
    if mode == "none":
        return pred_mat
    pred = pred_mat.copy()
    # Effects (same shape as pred)
    delta = pred - ctrl_mean[None, :]
    A = np.abs(delta)
    sign = np.sign(delta)

    if mode == "power":
        # Δ' = sign(Δ) * |Δ|^γ ; then rescale to preserve the chosen quantile magnitude
        Dp = (A ** max(gamma, 1.0000001))
        # scale to preserve q-th abs magnitude (per-row)
        q_old = np.quantile(A, preserve_q, axis=1, keepdims=True)
        q_new = np.quantile(Dp, preserve_q, axis=1, keepdims=True) + 1e-12
        scale = np.where(q_new > 0, q_old / q_new, 1.0)
        delta_sharp = sign * (Dp * scale)

    elif mode == "topk":
        # Inflate top-k% |Δ| by (1+alpha), shrink others by (1-beta)
        P, G = delta.shape
        k = np.maximum(1, (topk_frac * G).astype(int) if isinstance(topk_frac, np.ndarray) else int(round(topk_frac * G)))
        delta_sharp = delta.copy()
        for i in range(P):
            idx = np.argpartition(A[i], G - k)[-k:]
            not_idx = np.setdiff1d(np.arange(G), idx, assume_unique=False)
            delta_sharp[i, idx] *= (1.0 + alpha)
            delta_sharp[i, not_idx] *= (1.0 - beta)

    elif mode == "sigmoid":
        # Δ' = A * tanh(B * Δ), with A chosen so that q-th |Δ| is preserved
        B = sigmoid_B
        T = np.tanh(B * delta)
        q_old = np.quantile(A, preserve_q, axis=1, keepdims=True)
        q_new = np.quantile(np.abs(T), preserve_q, axis=1, keepdims=True) + 1e-12
        Arow = np.where(q_new > 0, q_old / q_new, 1.0)
        delta_sharp = Arow * T
    else:
        return pred_mat

    pred_sharp = ctrl_mean[None, :] + delta_sharp
    return pred_sharp
