import numpy as np

try:
    import torch
    _TORCH_AVAILABLE = True
except Exception:
    _TORCH_AVAILABLE = False


def neal_func(x):
   
    return 0.3 + 0.4 * x + 0.5 * np.sin(2.7 * x) + 1.1 / (1 + x**2)


def _build_x_by_periods(
    n_total=None,
    n_periods=None,           
    points_per_period=None,   
    omega=2.7,                
    start_at_zero=True       
):
   
    T = 2 * np.pi / float(omega) 

   

    if n_periods is not None and points_per_period is not None:
        n_total_calc = int(n_periods) * int(points_per_period)
        if n_total is not None and n_total != n_total_calc:
        
            n_total = n_total_calc
        else:
            n_total = n_total_calc
    elif n_periods is not None:
        
        ppp = int(round(n_total / int(n_periods)))
        n_total = ppp * int(n_periods)
    else:  
        
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
    to_tensor=True,           
    dtype=None,               
    device='cpu',

    
    use_periodic_x=False,     
    n_periods=None,           
    points_per_period=None,   
    period_omega=2.7,         
    start_at_zero=True       
):
   
    rng = np.random.default_rng(random_seed)
    n_outlier = int(round(n_total * float(outlier_ratio)))

   
    if use_periodic_x:
      
        x_ob, _T, _nP, n_total = _build_x_by_periods(
            n_total=n_total,
            n_periods=n_periods,
            points_per_period=points_per_period,
            omega=period_omega,
            start_at_zero=start_at_zero
        )
        shuffle = False  
    else:
      
        x_lo, x_hi = float(x_range[0]), float(x_range[1])
        x_ob = rng.random(n_total) * (x_hi - x_lo) + x_lo

   
    y_clean = neal_func(x_ob)
    y_ob = np.copy(y_clean)
    is_outlier = np.zeros(n_total, dtype=bool)
    std_y = np.std(y_clean)

    if n_outlier > 0:
        if outlier_type == 'uniform':
            idx = rng.choice(n_total, n_outlier, replace=False)
            shift = rng.uniform(3 * std_y, 5 * std_y, size=n_outlier)
            sign = rng.choice(np.array([-1.0, 1.0]), size=n_outlier)
            y_ob[idx] = y_clean[idx] + shift * sign
            is_outlier[idx] = True

        elif outlier_type == 'focused':
          
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
            y_ob[idx] = y_clean[idx] + shift  
            is_outlier[idx] = True

        else:
            raise ValueError(f"Unsupported outlier_type: {outlier_type}")

    
    mask_normal = ~is_outlier
    if np.any(mask_normal):
        y_ob[mask_normal] += rng.normal(0.0, float(normal_noise), size=int(mask_normal.sum()))

    if shuffle:
        idx2 = np.arange(n_total)
        rng.shuffle(idx2)
        x_ob = x_ob[idx2]
        y_ob = y_ob[idx2]
        is_outlier = is_outlier[idx2]

  
    if to_tensor:
        
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
    
    clean_prefix=0,              
    shuffle_tail_only=True,      
):
   
    rng = np.random.default_rng(random_seed)
    n_outlier = int(round(n_total * float(outlier_ratio)))
    n_outlier = max(0, min(n_outlier, n_total - clean_prefix))  


    x_lo, x_hi = float(x_range[0]), float(x_range[1])
    x_ob = rng.random(n_total) * (x_hi - x_lo) + x_lo

    
    y_clean = neal_func(x_ob)
    y_ob = np.copy(y_clean)

  
    is_outlier = np.zeros(n_total, dtype=bool)

    
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
            
            x_median = np.median(x_ob)
            jitter_x = rng.normal(0.0, 0.1, size=n_outlier)
            x_focused = x_median + jitter_x
            y_focused = neal_func(x_focused) - 3 * std_y + rng.normal(0.0, 0.5 * std_y, size=n_outlier)
            x_ob[idx] = x_focused
            y_ob[idx] = y_focused
            is_outlier[idx] = True

        elif outlier_type == 'asymmetric':
            shift = rng.uniform(3 * std_y, 5 * std_y, size=n_outlier)
            y_ob[idx] = y_clean[idx] + shift  
            is_outlier[idx] = True

        else:
            raise ValueError(f"Unsupported outlier_type: {outlier_type}")


    mask_normal = ~is_outlier
    if np.any(mask_normal):
        y_ob[mask_normal] += rng.normal(0.0, float(normal_noise), size=int(mask_normal.sum()))

    if shuffle:
        if clean_prefix > 0 and shuffle_tail_only:
           
            tail_idx = np.arange(clean_prefix, n_total)
            rng.shuffle(tail_idx)
            
            new_order = np.concatenate([np.arange(clean_prefix), tail_idx])
        else:
            
            new_order = np.arange(n_total)
            rng.shuffle(new_order)

        x_ob = x_ob[new_order]
        y_ob = y_ob[new_order]
        is_outlier = is_outlier[new_order]


    if to_tensor:
        x_t = torch.from_numpy(x_ob).to(dtype=dtype, device=device).unsqueeze(1)
        y_t = torch.from_numpy(y_ob).to(dtype=dtype, device=device)
        m_t = torch.from_numpy(is_outlier)
        return x_t, y_t, m_t
    else:
        return x_ob.reshape(-1, 1), y_ob, is_outlier
