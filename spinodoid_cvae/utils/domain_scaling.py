# utils/domain_scaling.py

# note: this module provides domain-aware scaling for spinodoid parameters:
# s = [theta1, theta2, theta3, rho]
# - thetas are either exactly 0 (inactive) or in [15°, 90°] (active)
# - rho is in [0.3, 1.0]
# - active thetas and rho are scaled to [-1, 1]; inactive thetas map to 0 in normalized space
# - includes numpy and torch implementations with identical behavior

import numpy as np

try:
    import torch
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False

# === physical bounds ===
THETA_MIN_ACTIVE = 15.0
THETA_MAX_ACTIVE = 90.0
THETA_RANGE = THETA_MAX_ACTIVE - THETA_MIN_ACTIVE
RHO_MIN, RHO_MAX = 0.3, 1.0
RHO_RANGE = RHO_MAX - RHO_MIN

# threshold for deciding if theta is inactive (mask = 0)
MASK_THRESHOLD_DEG = 0.75  # degrees; if below this, treat as inactive

# -------------------------------------
# numpy helpers (used in evaluation)
# -------------------------------------

def _scale_theta_np(theta_deg: float) -> float:
    # scale theta to [-1, 1] if active, else 0 for inactive
    # 15° maps to -1, 90° maps to +1
    if theta_deg < MASK_THRESHOLD_DEG:
        return 0.0
    return ((theta_deg - THETA_MIN_ACTIVE) / THETA_RANGE) * 2.0 - 1.0

def _unscale_theta_np(theta_scaled: float) -> float:
    # unscale theta from [-1, 1] to degrees
    # exactly 0 stays 0 (inactive), else [-1, 1] -> [15°, 90°]
    if theta_scaled < 1e-6:
        return 0.0
    return ((theta_scaled + 1.0) / 2.0) * THETA_RANGE + THETA_MIN_ACTIVE

def _scale_rho_np(rho_val: float) -> float:
    # scale rho in [0.3, 1.0] to [-1, 1]
    return ((rho_val - RHO_MIN) / RHO_RANGE) * 2.0 - 1.0

def _unscale_rho_np(rho_scaled: float) -> float:
    # unscale rho from [-1, 1] to [0.3, 1.0]
    return ((rho_scaled + 1.0) / 2.0) * RHO_RANGE + RHO_MIN

def normalize_S(S):
    """
    normalize full s = [theta1, theta2, theta3, rho] (numpy).
    thetas: inactive -> 0, active [15, 90] -> [-1, 1]
    rho: [0.3, 1.0] -> [-1, 1]
    """
    t = np.asarray(S[:3], dtype=float)
    rho = float(S[3])
    t_scaled = np.array([_scale_theta_np(x) for x in t], dtype=float)
    rho_scaled = _scale_rho_np(rho)
    return np.concatenate([t_scaled, [rho_scaled]]).astype(float)

def denormalize_S(Sn):
    """
    unnormalize full s from normalized space to physical units (numpy).
    """
    t_scaled = np.asarray(Sn[:3], dtype=float)
    rho_scaled = float(Sn[3])
    t = np.array([_unscale_theta_np(y) for y in t_scaled], dtype=float)
    rho = _unscale_rho_np(rho_scaled)
    return np.concatenate([t, [rho]]).astype(float)

def pack_active(Sn, active_idx):
    """
    keep only active thetas + rho from normalized full s (numpy).
    active_idx: indices (0-2) of thetas that are active in this pattern.
    returns shape (k+1,), where k = len(active_idx)
    """
    theta = np.asarray(Sn[:3], dtype=float)
    rho = float(Sn[3])
    return np.concatenate([theta[active_idx], [rho]]).astype(float)

def unpack_active(Sn_active, active_idx):
    """
    reinsert zeros for inactive thetas (normalized space) (numpy).
    sn_active = [theta_active(s)..., rho_scaled]
    returns full normalized s with 3 thetas + rho
    """
    k = len(active_idx)
    theta_active = np.asarray(Sn_active[:k], dtype=float)
    rho_scaled = float(Sn_active[k])
    theta_full = np.zeros(3, dtype=float)
    theta_full[active_idx] = theta_active
    return np.concatenate([theta_full, [rho_scaled]]).astype(float)

# -------------------------------------
# torch helpers (used in training)
# -------------------------------------

if _HAS_TORCH:

    def _scale_theta_torch(theta_deg: "torch.Tensor") -> "torch.Tensor":
        # elementwise: inactive -> 0; active [15, 90] -> [-1, 1]
        inactive = theta_deg < MASK_THRESHOLD_DEG
        scaled_active = ((theta_deg - THETA_MIN_ACTIVE) / THETA_RANGE) * 2.0 - 1.0
        return torch.where(inactive, torch.zeros_like(theta_deg), scaled_active)

    def _unscale_theta_torch(theta_scaled: "torch.Tensor") -> "torch.Tensor":
        # elementwise: exactly 0 stays 0; else [-1, 1] -> [15, 90]
        # we check equality to 0.0 because inactive thetas are set exactly to 0
        is_zero = theta_scaled == 0.0
        unscaled_active = ((theta_scaled + 1.0) / 2.0) * THETA_RANGE + THETA_MIN_ACTIVE
        return torch.where(is_zero, torch.zeros_like(theta_scaled), unscaled_active)

    def _scale_rho_torch(rho: "torch.Tensor") -> "torch.Tensor":
        return ((rho - RHO_MIN) / RHO_RANGE) * 2.0 - 1.0

    def _unscale_rho_torch(rho_scaled: "torch.Tensor") -> "torch.Tensor":
        return ((rho_scaled + 1.0) / 2.0) * RHO_RANGE + RHO_MIN

    def normalize_S_torch(S_batch: "torch.Tensor") -> "torch.Tensor":
        """
        normalize full s (torch).
        s_batch: (b,4) with degrees + rho
        returns (b,4) normalized with inactive thetas set exactly to 0
        """
        t = S_batch[:, :3]
        rho = S_batch[:, 3:4]
        t_scaled = _scale_theta_torch(t)
        rho_scaled = _scale_rho_torch(rho)
        return torch.cat([t_scaled, rho_scaled], dim=1)

    def denormalize_S_torch(Sn_batch: "torch.Tensor") -> "torch.Tensor":
        """
        unnormalize full s from normalized space to physical units (torch).
        """
        t_scaled = Sn_batch[:, :3]
        rho_scaled = Sn_batch[:, 3:4]
        t = _unscale_theta_torch(t_scaled)
        rho = _unscale_rho_torch(rho_scaled)
        return torch.cat([t, rho], dim=1)

    def pack_active_torch(Sn_batch: "torch.Tensor", active_idx: list[int]) -> "torch.Tensor":
        """
        keep only active thetas + rho from normalized full s (torch).
        returns (b, k+1)
        """
        theta = Sn_batch[:, :3]
        rho = Sn_batch[:, 3:4]
        theta_active = theta[:, active_idx]
        return torch.cat([theta_active, rho], dim=1)

    def unpack_active_torch(Sn_active_batch: "torch.Tensor", active_idx: list[int]) -> "torch.Tensor":
        """
        reinsert zeros for inactive thetas (normalized space) (torch).
        sn_active_batch: (b, k+1)
        returns (b, 4)
        """
        b = Sn_active_batch.shape[0]
        k = len(active_idx)
        theta_active = Sn_active_batch[:, :k]
        rho_scaled = Sn_active_batch[:, k:k+1]
        theta_full = torch.zeros((b, 3), dtype=Sn_active_batch.dtype, device=Sn_active_batch.device)
        theta_full[:, active_idx] = theta_active
        return torch.cat([theta_full, rho_scaled], dim=1)

# -----------------------------------------------------------------------------
# tiny utils for pattern handling (shared by train/eval)
# -----------------------------------------------------------------------------

def active_indices_from_pattern(pattern: str) -> list[int]:
    """
    parse a theta sparsity pattern string like "001" -> [2]
    returns 0-based indices of active thetas
    """
    pattern = pattern.strip()
    assert len(pattern) == 3 and set(pattern) <= {"0", "1"}, "pattern must be a 3-char string of 0/1"
    return [i for i, ch in enumerate(pattern) if ch == "1"]
