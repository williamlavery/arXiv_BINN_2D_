


import numpy as np
import torch

from ..density.prepare_model_loss import prepare_model_run_data

def prepare_grow_run_data(models_dics, plot_params=None, best_epoch_termination=True, val_loss_best=True):
    import numpy as np
    try:
        import torch
    except ImportError:
        torch = None

    if plot_params is None:
        plot_params = {}

    run_data = []
    model_keys = list(models_dics.keys())
    dic_values = list(models_dics.values())

    # Determine scaling factor for epoch time based on print frequency
    sample_model = list(dic_values[0].values())[0]
    sf = sample_model.print_freq if len(sample_model.train_loss_list) > len(sample_model.epoch_times) else 1

    if val_loss_best:
        total_stats,_ = prepare_model_run_data(models_dics, plot_params=None, best_epoch_termination=True)

    for i, (key, model_dic) in enumerate(zip(model_keys, dic_values)):
        model_wrappers = list(model_dic.values())

        # Use user-defined or default styling
        color, linestyle, markerstyle, label = plot_params.get(
            key, [f"C{i}", "-", ".", f"Run {i+1}"]
        )

        best_epochs_lst = []
        grow_errs = []
        frozen_switch_epochs = []

        # ---------- First pass: compute best epochs and max length ----------
        max_len = 0
        for m in model_wrappers:
            # Epoch index where validation loss is best
            val_losses = np.array(m.val_loss_list)
            best_val_epoch = np.argmin(np.abs(val_losses - m.best_val_loss))
            best_epochs_lst.append(best_val_epoch)

            # Use full length of recorded growth errors for this model
            series_len = len(m.growth_errors)
            max_len = max(max_len, series_len)
            frozen_switch_epoch = getattr(m, 'frozen_switch_epoch', None)
            frozen_switch_epochs.append(frozen_switch_epoch)

        # ---------- Second pass: build padded growth error arrays ----------
        for m in model_wrappers:
            series_vals = []

            for grow_err in m.growth_errors:
                # Try to reproduce original behaviour (mean over column 0 of a tensor),
                # but handle non-tensor / different shapes gracefully.
                try:
                    if torch is not None and isinstance(grow_err, torch.Tensor):
                        val = torch.mean(grow_err[:, 0]).detach().cpu().numpy()
                    else:
                        arr = np.asarray(grow_err)
                        if arr.ndim > 1:
                            val = np.mean(arr[:, 0])
                        else:
                            val = np.mean(arr)
                except Exception:
                    arr = np.asarray(grow_err)
                    if arr.ndim > 1:
                        val = np.mean(arr[:, 0])
                    else:
                        val = np.mean(arr)

                series_vals.append(val)

            padded = np.full(max_len, np.nan, dtype=float)
            series_len = len(series_vals)
            padded[:series_len] = series_vals
            grow_errs.append(padded)

        grow_errs_arr = np.vstack(grow_errs)

        # ---------- Running mins across time ----------
        running_mins = np.minimum.accumulate(
            np.nan_to_num(grow_errs_arr, nan=np.inf),
            axis=1,
        )
        running_min_min = np.nanmin(running_mins, axis=0)
        running_min_max = np.nanmax(running_mins, axis=0)

        if val_loss_best:
            x_orig = total_stats[i]['epochs']
            best_epoch_idx_orig = total_stats[i]['best_epoch_seed_idx'][0]
            best_epoch_idx = best_epoch_idx_orig//sf + 1
            best_seed_idx = total_stats[i]['best_epoch_seed_idx'][1]
        else:
            best_epoch_idx = np.nanargmin(running_min_min)
            best_seed_idx = np.nanargmin(grow_errs_arr[:, best_epoch_idx]) # CHECK this is correct

        epochs = np.arange(max_len)  # full recorded training length

        mean_vals = np.nanmean(grow_errs_arr, axis=0)
        std_vals = np.nanstd(grow_errs_arr, axis=0)

        # ---------- Final growth MSE stats (matching prepare_growth_mse_data) ----------
        # Each best_growth_error can be an array/tensor; take its mean per run
        final_mses = np.array([m.best_growth_error for m in model_wrappers], dtype=np.float64)

        avg_mse = float(np.mean(final_mses))
        std_mse = float(np.std(final_mses))
        min_mse = float(np.min(final_mses))
        max_mse = float(np.max(final_mses))
        
        # For backward-compatibility, keep mean_final/std_final aliases
        mean_final = avg_mse
        std_final = std_mse

        label_with_stats = f"{label}"  # keep label as-is; you can append stats if desired

        min_vals = np.nanmin(grow_errs_arr, axis=0)
        max_vals = np.nanmax(grow_errs_arr, axis=0)

        if best_epoch_termination:
            mean_vals = mean_vals[: best_epoch_idx + 1]
            min_vals = min_vals[: best_epoch_idx + 1]
            max_vals = max_vals[: best_epoch_idx + 1]
            epochs = epochs[: best_epoch_idx + 1]
        
        try:
            max_frozen_switch_epochs  = max(frozen_switch_epochs) 
        except:
            max_frozen_switch_epochs = None

        run_data.append(
            {
                # --- scalar summary (from prepare_growth_mse_data) ---
                "label": label_with_stats,
                "avg_mse": avg_mse,
                "std_mse": std_mse,
                "min_mse": min_mse,
                "max_mse": max_mse,
                "final_mses": final_mses,
                "color": color,
                "frozen_switch_epoch": max_frozen_switch_epochs,


                # --- aliases/backwards-compatible fields ---
                "mean_final": mean_final,
                "std_final": std_final,

                # --- time-series and running-min info ---
                "epochs": epochs,
                "mean_vals": mean_vals,
                "std_vals": std_vals,
                "min_vals": min_vals,
                "max_vals": max_vals,
                "running_min_min": running_min_min,
                "running_min_max": running_min_max,
                "best_epoch_seed_idx": [best_epoch_idx, best_seed_idx],
                "best_epochs_lst": np.array(best_epochs_lst),
                "best_epoch_idx_orig": best_epoch_idx_orig,
                "x_orig": x_orig,

                # --- styling ---
                "linestyle": linestyle,
                "markerstyle": markerstyle,
            }
        )

    return run_data

