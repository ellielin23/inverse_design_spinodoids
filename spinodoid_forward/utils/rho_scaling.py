# utils/rho_scaling.py

RHO_MIN = 0.30      # domain lower bound (kept for other funcs)
EPS_SIG = 1e-8      # floor to keep NLL well-defined

def g_none(rho):
    return rho

def g_linear(rho, rho_min=RHO_MIN):
    # maps rho_min -> 1,    1 -> 0
    rho = float(rho)
    if rho >= 1.0:
        return 0.0
    s = (1.0 - rho) / (1.0 - rho_min)
    return max(0.0, min(1.0, s))

def g_power(rho, k=0.5, rho_min=RHO_MIN):
    # steeper near rho=1 if k>1
    rho = float(rho)
    if rho >= 1.0:
        return 0.0
    base = max(0.0, (1.0 - rho) / (1.0 - rho_min))
    return base**k

def g_hinge(rho, knee=0.96, rho_min=RHO_MIN):
    # mostly unscaled until a knee, then drops to 0 at 1
    rho = float(rho)
    if rho >= 1.0:
        return 0.0
    if rho <= rho_min:
        return 1.0
    if rho <= knee:
        # gentle slope between rho_min and knee
        return (knee - rho) / (knee - rho_min)
    # from knee to 1, sharper drop
    return max(0.0, (1.0 - rho) / (1.0 - knee))

def g_targeted(
    rho: float,
    mode: str = "power",   # "power" or "linear"
    rho_thresh: float = 0.90,
    floor: float = 0.12,   # minimum scaling factor (e.g., 15% of original σ)
    k: float = 1.0         # exponent if mode=="power"
):
    """
    Thresholded + floored scaling:
      - For rho <= rho_thresh: no scaling (scale = 1.0).
      - For rho >  rho_thresh: apply chosen base scaling on [rho_thresh, 1]:
            t = (1 - rho) / (1 - rho_thresh) ∈ [0,1]
            mode=="linear": scale = t
            mode=="power" : scale = t**k
        Then enforce a floor: scale = max(floor, scale).
      - At rho >= 1: return 'floor' (never zero to keep NLL stable).

    Returns a factor in [floor, 1].
    """
    rho = float(rho)
    if rho <= rho_thresh:
        return 1.0
    if rho >= 1.0:
        return floor

    # map rho ∈ [rho_thresh, 1] -> t ∈ [1, 0]
    t = (1.0 - rho) / (1.0 - rho_thresh)
    t = max(0.0, min(1.0, t))

    if mode == "linear":
        scale = t
    elif mode == "power":
        scale = t**k
    else:
        raise ValueError(f"g_targeted: unknown mode '{mode}', use 'linear' or 'power'.")

    return max(floor, min(1.0, scale))

def g_targeted_tail(rho, rho_thresh=0.92, knee=0.98, k1=1.2, k2=2.0, floor=0.05):
    """
    Piecewise targeted shrink:
      - ρ <= rho_thresh: scale = 1
      - rho_thresh < ρ <= knee:   scale = t^k1
      - knee < ρ < 1:            scale = u^k2  (steeper near 1)
      - ρ >= 1: scale = floor
    where t = (1-ρ)/(1-rho_thresh), u = (1-ρ)/(1-knee).
    """
    rho = float(rho)
    if rho <= rho_thresh:
        return 1.0
    if rho >= 1.0:
        return floor
    if rho <= knee:
        t = (1.0 - rho) / (1.0 - rho_thresh)
        return max(floor, min(1.0, t**k1))
    # ultra-tail
    u = (1.0 - rho) / (1.0 - knee + 1e-12)
    return max(floor, min(1.0, u**k2))


def g_tail_strict(rho, rho_thresh=0.92, knee=0.985, k1=1.0, k2=2.0, floor=0.12):
    """
    Monotone three-region shrink:
      - rho <= rho_thresh:           scale = 1
      - rho_thresh < rho <= knee:    scale = [(1-rho)/(1-rho_thresh)]^k1
      - knee < rho < 1:              scale = [(1-rho)/(1-knee)]^k2
      - rho >= 1:                    scale = floor
    'floor' prevents overconfidence/NLL blow-ups in the limit.
    """
    rho = float(rho)
    if rho <= rho_thresh:
        return 1.0
    if rho >= 1.0:
        return floor

    if rho <= knee:
        t = (1.0 - rho) / (1.0 - rho_thresh)
        return max(floor, min(1.0, t**k1))

    # sharp but safer tail
    u = (1.0 - rho) / (1.0 - knee + 1e-12)
    return max(floor, min(1.0, u**k2))