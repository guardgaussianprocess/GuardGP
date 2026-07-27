# -*- coding: utf-8 -*-
import numpy as np

try:
    import torch
    _TORCH_AVAILABLE = True
except Exception:
    _TORCH_AVAILABLE = False


def neal_func(x):
    """
    Standard benchmark function for robust regression experiments.
    f(x) = 0.3 + 0.4x + 0.5 sin(2.7x) + 1.1/(1 + x^2)

    Parameters:
      x: np.ndarray or torch.Tensor (1D)
    Returns:
      Values of the same shape
    """
    return 0.3 + 0.4 * x + 0.5 * np.sin(2.7 * x) + 1.1 / (1 + x**2)


def _build_x_by_periods(
    n_total=None,
    n_periods=None,           # Number of periods, e.g. 10
    points_per_period=None,   # Points per period, e.g. 100
    omega=2.7,                # Sine frequency: neal_func uses sin(2.7 x)
    start_at_zero=True        # True: interval [0, L]; False: interval [-L/2, L/2]
):
    """
    Accurately generate x according to "n_periods x points_per_period" (strictly equidistant, increasing).
    Constraints:
      - If both n_periods and points_per_period are given => n_total = n_periods * points_per_period
      - If n_periods and n_total are given => automatically align points_per_period to integer, and update n_total
      - If points_per_period and n_total are given => automatically align n_periods to integer, and update n_total
    Returns:
      x (np.ndarray, shape=(n_total,)), T, n_periods(int), n_total(int)
    """
    T = 2 * np.pi / float(omega)  # Period

    if n_periods is None and points_per_period is None:
        raise ValueError("At least specify one of n_periods or points_per_period.")

    if n_periods is not None and points_per_period is not None:
        n_total_calc = int(n_periods) * int(points_per_period)
        if n_total is not None and n_total != n_total_calc:
            # Align by periods
            n_total = n_total_calc
        else:
            n_total = n_total_calc
    elif n_periods is not None:
        if n_total is None:
            raise ValueError("n_periods provided but missing points_per_period or n_total.")
        ppp = int(round(n_total / int(n_periods)))
        n_total = ppp * int(n_periods)
    else:  # Only points_per_period provided
        if n_total is None:
            raise ValueError("points_per_period provided but missing n_periods or n_total.")
        n_periods = int(round(n_total / int(points_per_period)))
        n_total = int(points_per_period) * int(n_periods)

    L = n_periods * T
    if start_at_zero:
        x0, x1 = 0.0, L
    else:
        x0, x1 = -L / 2.0, L / 2.0

    x = np.linspace(x0, x1, n_total)
    return x, T, int(n_periods), int(n_total)


def make_stream_dataset_multioutlier_time(
    n_total=200,
    outlier_ratio=0.1,
    outlier_type='uniform',   # 'uniform' | 'focused' | 'asymmetric'
    normal_noise=0.2,
    x_range=(-3, 3),
    shuffle=True,
    random_seed=5,
    to_tensor=True,           # Whether to convert to PyTorch tensor
    dtype=None,               # torch.dtype, e.g. torch.float64; None defaults to torch standard
    device='cpu',

    # ====== Added: Periodic sampling parameters (specify one or both) ======
    use_periodic_x=False,     # True enables precise periodic sampling; False uses random x logic
    n_periods=None,           # Number of periods (e.g. 10)
    points_per_period=None,   # Points per period (e.g. 100)
    period_omega=2.7,         # Sine frequency, matches neal_func by default
    start_at_zero=True        # Period interval start [0, L], or centered [-L/2, L/2]
):
    """
    Generate regression data (with multiple outlier types), compatible with legacy interface,
    supporting fixed time series & precise periodic sampling.
    Returns: x_ob, y_ob, is_outlier
      - If to_tensor=True: returns torch.Tensor (x_ob shape=(N,1); y_ob shape=(N,); is_outlier bool tensor)
      - Otherwise returns np.ndarray

    Usage:
    1) Legacy (random points):
        make_stream_dataset_multioutlier(n_total=200, shuffle=True, use_periodic_x=False, ...)
    2) New (fixed time series & precise periodic), e.g. "1000 points total, 10 periods, 100 points/period":
        make_stream_dataset_multioutlier(n_total=1000, use_periodic_x=True,
                                         n_periods=10, points_per_period=100, period_omega=2.7,
                                         shuffle=False)
    """
    rng = np.random.default_rng(random_seed)
    n_outlier = int(round(n_total * float(outlier_ratio)))

    # ========== 1) Generate x (two modes) ==========
    if use_periodic_x:
        # Precise alignment by period, covering n_periods, equidistant, strict temporal order
        x_ob, _T, _nP, n_total = _build_x_by_periods(
            n_total=n_total,
            n_periods=n_periods,
            points_per_period=points_per_period,
            omega=period_omega,
            start_at_zero=start_at_zero
        )
        shuffle = False  # Fixed temporal order, no shuffle by default
    else:
        # Legacy logic: random uniform sampling
        x_lo, x_hi = float(x_range[0]), float(x_range[1])
        x_ob = rng.random(n_total) * (x_hi - x_lo) + x_lo

    # ========== 2) Generate clean curve & initialize ==========
    y_clean = neal_func(x_ob)
    y_ob = np.copy(y_clean)
    is_outlier = np.zeros(n_total, dtype=bool)
    std_y = np.std(y_clean)

    # ========== 3) Inject outliers ==========
    if n_outlier > 0:
        if outlier_type == 'uniform':
            idx = rng.choice(n_total, n_outlier, replace=False)
            shift = rng.uniform(3 * std_y, 5 * std_y, size=n_outlier)
            sign = rng.choice(np.array([-1.0, 1.0]), size=n_outlier)
            y_ob[idx] = y_clean[idx] + shift * sign
            is_outlier[idx] = True

        elif outlier_type == 'focused':
            # Focused near median x (Note: if use_periodic_x=True, this only focuses values without shuffling order)
            x_median = np.median(x_ob)
            jitter_x = rng.normal(0.0, 0.1, size=n_outlier)
            x_focused = x_median + jitter_x
            y_focused = neal_func(x_focused) - 3 * std_y + rng.normal(0.0, 0.5 * std_y, size=n_outlier)
            idx = rng.choice(n_total, n_outlier, replace=False)
            x_ob[idx] = x_focused
            y_ob[idx] = y_focused
            is_outlier[idx] = True

        elif outlier_type == 'asymmetric':
            idx = rng.choice(n_total, n_outlier, replace=False)
            shift = rng.uniform(3 * std_y, 5 * std_y, size=n_outlier)
            y_ob[idx] = y_clean[idx] + shift  # One-sided shift
            is_outlier[idx] = True

        else:
            raise ValueError(f"Unsupported outlier_type: {outlier_type}")

    # ========== 4) Normal noise (added only to non-outliers) ==========
    mask_normal = ~is_outlier
    if np.any(mask_normal):
        y_ob[mask_normal] += rng.normal(0.0, float(normal_noise), size=int(mask_normal.sum()))

    # ========== 5) Shuffle (if periodic time series enabled, default is no shuffle) ==========
    if shuffle:
        idx2 = np.arange(n_total)
        rng.shuffle(idx2)
        x_ob = x_ob[idx2]
        y_ob = y_ob[idx2]
        is_outlier = is_outlier[idx2]

    # ========== 6) Convert to Tensor or Numpy ==========
    if to_tensor:
        if not _TORCH_AVAILABLE:
            raise ImportError("PyTorch not detected. Please set to_tensor=False or install torch first.")
        torch_dtype = dtype if dtype is not None else torch.get_default_dtype()
        x_ob_t = torch.from_numpy(x_ob).to(dtype=torch_dtype, device=device).unsqueeze(1)  # (N,1)
        y_ob_t = torch.from_numpy(y_ob).to(dtype=torch_dtype, device=device)
        is_outlier_t = torch.from_numpy(is_outlier)
        return x_ob_t, y_ob_t, is_outlier_t
    else:
        return x_ob.reshape(-1, 1), y_ob, is_outlier
    

def make_stream_dataset_multioutlier_clean(
    n_total=200,
    outlier_ratio=0.1,
    outlier_type='uniform',      # 'uniform' | 'focused' | 'asymmetric'
    normal_noise=0.2,
    x_range=(-3, 3),
    shuffle=True,
    random_seed=5,
    to_tensor=True,
    dtype=torch.float64,
    device='cpu',
    # === Added: Guarantee clean prefix ===
    clean_prefix=0,              # Ensure first clean_prefix points have is_outlier=False
    shuffle_tail_only=True,      # If shuffle required and clean_prefix>0, only shuffle tail after prefix
):
    """
    Generate regression data with multi-type outliers, **guaranteeing the first clean_prefix points are clean**.
    When clean_prefix > 0 and shuffle=True:
      - If shuffle_tail_only=True: only shuffle samples in interval [clean_prefix, n_total), keeping prefix untouched;
      - If shuffle_tail_only=False: shuffle all samples, which cannot guarantee clean prefix (not recommended).
    Returns: x_ob, y_ob, is_outlier
    """
    rng = np.random.default_rng(random_seed)
    n_outlier = int(round(n_total * float(outlier_ratio)))
    n_outlier = max(0, min(n_outlier, n_total - clean_prefix))  # Cannot exceed non-prefix region

    # 1) Generate equidistant/random x sequentially
    x_lo, x_hi = float(x_range[0]), float(x_range[1])
    x_ob = rng.random(n_total) * (x_hi - x_lo) + x_lo

    # 2) Baseline y
    y_clean = neal_func(x_ob)
    y_ob = np.copy(y_clean)

    # 3) Initialize labels (prefix forced clean)
    is_outlier = np.zeros(n_total, dtype=bool)

    # 4) Sample outliers from index set allowing outliers (excluding prefix)
    candidate = np.arange(clean_prefix, n_total)
    if n_outlier > 0 and candidate.size > 0:
        idx = rng.choice(candidate, n_outlier, replace=False)

        std_y = float(np.std(y_clean))
        if outlier_type == 'uniform':
            shift = rng.uniform(3 * std_y, 5 * std_y, size=n_outlier)
            sign = rng.choice(np.array([-1.0, 1.0]), size=n_outlier)
            y_ob[idx] = y_clean[idx] + shift * sign
            is_outlier[idx] = True

        elif outlier_type == 'focused':
            # Generate focused outliers, written only to candidate region
            x_median = np.median(x_ob)
            jitter_x = rng.normal(0.0, 0.1, size=n_outlier)
            x_focused = x_median + jitter_x
            y_focused = neal_func(x_focused) - 3 * std_y + rng.normal(0.0, 0.5 * std_y, size=n_outlier)
            x_ob[idx] = x_focused
            y_ob[idx] = y_focused
            is_outlier[idx] = True

        elif outlier_type == 'asymmetric':
            shift = rng.uniform(3 * std_y, 5 * std_y, size=n_outlier)
            y_ob[idx] = y_clean[idx] + shift  # One-sided shift
            is_outlier[idx] = True

        else:
            raise ValueError(f"Unsupported outlier_type: {outlier_type}")

    # 5) Add noise to normal points (exclude outliers)
    mask_normal = ~is_outlier
    if np.any(mask_normal):
        y_ob[mask_normal] += rng.normal(0.0, float(normal_noise), size=int(mask_normal.sum()))

    # 6) Shuffle logic
    if shuffle:
        if clean_prefix > 0 and shuffle_tail_only:
            # Only shuffle tail, maintaining prefix order and clean property
            tail_idx = np.arange(clean_prefix, n_total)
            rng.shuffle(tail_idx)
            # Re-concatenate index sequence
            new_order = np.concatenate([np.arange(clean_prefix), tail_idx])
        else:
            # Global shuffle (not recommended if clean_prefix>0 as it breaks clean prefix semantics)
            new_order = np.arange(n_total)
            rng.shuffle(new_order)

        x_ob = x_ob[new_order]
        y_ob = y_ob[new_order]
        is_outlier = is_outlier[new_order]

    # 7) Assertion (ensure prefix is indeed outlier-free)
    if clean_prefix > 0 and (not shuffle or shuffle_tail_only):
        assert not np.any(is_outlier[:clean_prefix]), "Outlier found in prefix, violating clean_prefix constraint."

    # 8) Convert to Tensor
    if to_tensor:
        x_t = torch.from_numpy(x_ob).to(dtype=dtype, device=device).unsqueeze(1)
        y_t = torch.from_numpy(y_ob).to(dtype=dtype, device=device)
        m_t = torch.from_numpy(is_outlier)
        return x_t, y_t, m_t
    else:
        return x_ob.reshape(-1, 1), y_ob, is_outlier
