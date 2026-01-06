
def DATA_add_noise_func_info(data_obj_params):
    return data_obj_params["add_noise_params"]

def DATA_add_noise_old(u_clean, data_obj_params):
    
    gamma       = data_obj_params["add_noise_params"]["gamma"]
    noise_percent = data_obj_params["add_noise_params"]["noise_percent"]
    seed   =  data_obj_params["add_noise_params"]["seed"]

    """
    Generate additive noise scaled as a percentage of the original signal magnitude.

    Parameters:
    - u_clean (np.ndarray): Noise-free cell density data.
    - gamma (float): Exponent from the assumed noise model.
    - noise_percent (float): Noise level as a percentage (e.g., 5 = 5% of signal).
    - seed (int): Seed for reproducible Gaussian noise.

    Returns:
    - tuple: (func_info dict, noisy np.ndarray)
    """


    t_start_idx = 0
    u = u_clean.copy()
    signal = u_clean[..., t_start_idx:]
    shape = signal.shape

    np.random.seed(seed)

    # Convert percent to fraction (e.g., 5% → 0.05)
    noise_frac = noise_percent / 100.0


    avg = np.mean(np.random.normal(size=shape))
    np.random.seed(seed)
    noise = noise_frac * np.random.normal(size=shape) * signal**gamma 
    noise_clipped = noise.clip(1e-5)
    signal_clipped = signal.clip(1e-5)
    noise *= noise_frac/np.mean(np.abs(noise_clipped)/signal_clipped) 

    u[..., t_start_idx:] += noise 

    func_info = DATA_add_noise_func_info(data_obj_params)
    return func_info, u.clip(0, np.inf)



def DATA_add_noise(u_clean, data_obj_params):
    """
    Add zero-mean Gaussian noise N(0,1)*|signal|**γ so that the mean absolute
    relative error after *clipping non-negative densities* equals
    `noise_percent` to within 1 × 10⁻⁴ %  (four decimal places).

    Parameters
    ----------
    u_clean : np.ndarray
        Noise-free cell-density data (non-negative).
    data_obj_params : dict
        data_obj_params["add_noise_params"] must contain:
            ├─ "gamma"         : float
            ├─ "noise_percent" : float       # e.g. 5 → 5 %
            └─ "seed"          : int

    Returns
    -------
    tuple
        (func_info : dict,  u_noisy : np.ndarray)
    """

    # ───── 1. unpack parameters ──────────────────────────────────────────────
    add_noise_params    = data_obj_params["add_noise_params"]
    gamma        = float(add_noise_params["dataGamma"])
    noise_pc     = float(  add_noise_params["dataNoisePercent"])
    seed         = int( add_noise_params["dataNoiseSeed"])

    t_start_idx  = 0
    eps          = 1e-8                      # floor to avoid divide-by-0
    target_frac  = noise_pc / 100.0          # desired mean |noise|/|signal|
    tol          = 1e-6                      # 0.0001 % as a fraction
    max_iter     = 25                        # usually converges in <10

    # ───── 2. prepare arrays & RNG ───────────────────────────────────────────
    u_noisy      = u_clean.copy() + tol
    signal       = u_noisy[..., t_start_idx:]        # view (no copy)
    shape        = signal.shape
    denom        = np.clip(np.abs(signal), eps, None)

    rng          = np.random.default_rng(seed)
    base_noise   = rng.standard_normal(size=shape) * np.abs(signal) ** gamma

    # ───── 3. initial analytic scale (ignores future clipping) ───────────────
    current_frac = np.mean(np.abs(base_noise) / denom)
    if current_frac == 0:
        raise ValueError("Zero baseline noise – check γ or input signal.")

    scale        = target_frac / current_frac

    # ───── 4. iterative scale–clip loop to hit target within tolerance ──────
    for _ in range(max_iter):
        noise      = base_noise * scale
        proposal   = signal + noise
        proposal   = np.clip(proposal, 0.0, np.inf)      # physical clipping
        err_frac   = np.mean(np.abs(proposal - signal) / denom)

        if abs(err_frac - target_frac) <= tol:
            break                                         # tolerance reached
        # multiplicative correction (guarantees convergence for monotone case)
        scale     *= target_frac / err_frac
    else:
        raise RuntimeError(
            f"Noise scaling did not converge after {max_iter} iterations "
            f"(final error={err_frac*100:.5f} %, target={noise_pc:.5f} %)."
        )

    # ───── 5. commit noise ---------------------------------------------------
    u_noisy[..., t_start_idx:] = proposal

    # ───── 6. diagnostics & bookkeeping --------------------------------------
    #ERROR = err_frac * 100.0                  # same metric you use later

    func_info = DATA_add_noise_func_info(data_obj_params)
    #func_info["achieved_percent"] = round(ERROR, 5)       # save for logs

    sigma_effective = np.std(base_noise * scale)  # pre-clipping noise std

    additional_info = {}
    additional_info["sigma_effective"] = sigma_effective
    #additional_info["max_u_clean"] = np.max(u_clean)
    #additional_info["min_u_clean"] = np.min(u_clean)  # corrected to use np.min


    func_info = DATA_add_noise_func_info(data_obj_params)

    return func_info, u_noisy,  additional_info



def DATA_add_noise(u_clean, data_obj_params, epsilon=1e-12):
    """
    Add zero-mean Gaussian noise N(0,1)*|signal|**γ so that the mean absolute
    relative error after *clipping non-negative densities* equals
    `noise_percent` to within 1 × 10⁻⁴ %  (four decimal places).

    Parameters
    ----------
    u_clean : np.ndarray
        Noise-free cell-density data (non-negative).
    data_obj_params : dict
        data_obj_params["add_noise_params"] must contain:
            ├─ "gamma"         : float
            ├─ "noise_percent" : float       # e.g. 5 → 5 %
            └─ "seed"          : int

    Returns
    -------
    np.ndarray
        u_noisy : np.ndarray  (same shape as u_clean)
    """

    # ───── 1. unpack parameters ──────────────────────────────────────────────
    add_noise_params = data_obj_params["add_noise_params"]
    gamma            = float(add_noise_params["dataGamma"])
    noise_pc         = float(add_noise_params["dataNoisePercent"])
    seed             = int(add_noise_params["dataNoiseSeed"])

    t_start_idx      = 0
    target_frac      = noise_pc / 100.0          # desired mean |noise|/|signal|
    tol              = 1e-6                      # 0.0001 % as a fraction
    max_iter         = 25                        # usually converges in <10

    # ───── 2. prepare arrays & RNG ───────────────────────────────────────────
    u_noisy = u_clean.copy()
    signal  = u_noisy[..., t_start_idx:]         # view (no copy)
    shape   = signal.shape

    rng        = np.random.default_rng(seed)
    base_noise = rng.standard_normal(size=shape) * np.abs(signal) ** gamma

    # ───── 2a. mask out near-zero signal for relative-error metric ──────────
    # We only enforce the target relative error on entries with "significant"
    # signal; near-zero entries otherwise dominate the metric.
    #
    # u_min is chosen relative to the global scale of the data.
    sig_max = float(np.max(np.abs(signal))) if signal.size else 0.0
    u_min   = max(epsilon, 1e-3)   # relative + absolute floor

    mask = np.abs(signal) >= u_min                     # bool array
    if not np.any(mask):
        raise ValueError(
            "All signal entries are below the minimum threshold for "
            "relative error; cannot define a meaningful noise level."
        )

    # denom is only defined on the masked (non-tiny) entries
    denom = np.abs(signal[mask])

    # ───── 3. initial analytic scale (ignores future clipping) ───────────────
    current_frac = np.mean(np.abs(base_noise[mask]) / denom)
    if current_frac == 0:
        raise ValueError("Zero baseline noise – check γ or input signal.")

    scale = target_frac / current_frac

    # ───── 4. iterative scale–clip loop to hit target within tolerance ──────
    for _ in range(max_iter):
        noise    = base_noise * scale
        proposal = signal + noise
        proposal = np.clip(proposal, 0.0, np.inf)      # physical clipping

        # relative error only on sufficiently large entries (mask)
        err_frac = np.mean(np.abs(proposal[mask] - signal[mask]) / denom)

        if abs(err_frac - target_frac) <= tol:
            break                                      # tolerance reached

        # multiplicative correction (guarantees convergence for monotone case)
        scale *= target_frac / err_frac
    else:
        raise RuntimeError(
            f"Noise scaling did not converge after {max_iter} iterations "
            f"(final error={err_frac*100:.5f} %, target={noise_pc:.5f} %)."
        )

    # ───── 5. commit noise ---------------------------------------------------
    u_noisy[..., t_start_idx:] = proposal

        # ───── 6. diagnostics & bookkeeping --------------------------------------
    #ERROR = err_frac * 100.0                  # same metric you use later

    func_info = DATA_add_noise_func_info(data_obj_params)
    #func_info["achieved_percent"] = round(ERROR, 5)       # save for logs

    sigma_effective = np.std(base_noise * scale)  # pre-clipping noise std

    additional_info = {}
    additional_info["sigma_effective"] = sigma_effective


    return func_info, u_noisy,  additional_info

