"""
run_data_preparation.py

Run-level data aggregation for diffusion-network experiments.

This module is part of a small utilities package with four files:


- run_data_preparation.py
    * Prepares aggregated run statistics (loss curves, timing, diffusion MSE)
      for plotting across multiple model runs and seeds.

This file contains no plotting itself: it only prepares arrays and
statistics for downstream visualizations.
"""

import numpy as np
import torch


def prepare_diff_run_data(dn_models_dics, plot_params=None):
    """
    Aggregate diffusion-error histories across runs for each configuration.

    Parameters
    ----------
    dn_models_dics : dict
        Nested dictionary of model wrappers, typically
        {config_key: {seed: modelWrapper, ...}, ...}.
        Each modelWrapper is expected to expose:
        - diffusion_errors (list of tensors or arrays)
        - best_diffusion_error
        - val_loss_list, best_val_loss
        - train_loss_list, epoch_times, print_freq
    plot_params : dict, optional
        Plot styling overrides for each config key.

    Returns
    -------
    list of dict
        Each dict contains:
        - 'epochs', 'mean_vals', 'std_vals'
        - 'min_vals', 'max_vals'
        - 'running_min_min', 'running_min_max'
        - 'best_epoch_idx', 'best_epochs_lst'
        - 'mean_final', 'hticks'
        - 'color', 'linestyle', 'markerstyle', 'label'
    """
    if plot_params is None:
        plot_params = {}

    run_data = []
    model_keys = list(dn_models_dics.keys())
    dic_values = list(dn_models_dics.values())

    # Determine scaling factor for epoch time based on print frequency
    sample_model = list(dic_values[0].values())[0]
    sf = (
        sample_model.print_freq
        if len(sample_model.train_loss_list) > len(sample_model.epoch_times)
        else 1
    )

    for i, (key, model_dic) in enumerate(zip(model_keys, dic_values)):
        model_wrappers = list(model_dic.values())

        color, linestyle, markerstyle, label = plot_params.get(
            key, [f"C{i}", "-", ".", f"Run {i+1}"]
        )

        best_epochs_lst = []
        diff_errs_raw = [[] for _ in range(len(model_wrappers))]
        diff_errs = []

        # Find maximum length up to best-val epochs across all runs
        max_len = 0
        for m in model_wrappers:
            best_val_epoch = np.argmin(
                np.abs(np.array(m.val_loss_list) - m.best_val_loss)
            )
            best_epochs_lst.append(best_val_epoch)
            max_len = max(max_len, best_val_epoch)
        max_len = max_len // sf + 1

        # Collect diffusion-error histories, padded to common length
        for j, m in enumerate(model_wrappers):
            try:
                best_epochs_scaled = best_epochs_lst[j] // sf + 1
                padded = np.full(max_len, np.nan)
                for diff_err in m.diffusion_errors:
                    diff_errs_raw[j].append(
                        torch.mean(diff_err[:, 0]).detach().cpu().numpy()
                    )
                padded[:best_epochs_scaled] = diff_errs_raw[j][:best_epochs_scaled]
                diff_errs.append(padded)
            except Exception:
                diff_err = m.diffusion_errors[: best_epochs_lst[j] // sf]
                padded_diff_err = np.pad(
                    np.array(diff_err),
                    (0, max_len - len(diff_err)),
                    constant_values=np.nan,
                )
                diff_errs.append(padded_diff_err)

        diff_errs_arr = np.vstack(diff_errs)
        running_mins = np.minimum.accumulate(
            np.nan_to_num(diff_errs_arr, nan=np.inf), axis=1
        )
        running_min_min = np.nanmin(running_mins, axis=0)
        running_min_max = np.nanmax(running_mins, axis=0)
        best_epoch_idx = np.nanargmin(running_min_min)
        epochs = np.arange(max_len)

        mean_vals = np.nanmean(diff_errs_arr, axis=0)
        std_vals = np.nanstd(diff_errs_arr, axis=0)

        # Final value / horizontal tick
        try:
            hticks = np.mean([m.best_diffusion_error for m in model_wrappers])
        except Exception:
            hticks_per_run = []
            for i_m, m in enumerate(model_wrappers):
                hticks_per_run.append(diff_errs_arr[i_m][best_epochs_lst[i_m] // sf])
            hticks = np.mean(hticks_per_run)

        try:
            last_vals = np.array(
                [mw.best_diffusion_error for mw in model_wrappers]
            )
        except Exception:
            last_vals = np.array(
                [
                    row[~np.isnan(row)][-1] if np.any(~np.isnan(row)) else np.nan
                    for row in running_mins
                ]
            )

        mean_final = np.nanmean(last_vals)
        std_final = np.nanstd(last_vals)
        label_with_stats = f"{label}"

        run_data.append(
            {
                "mean_final": mean_final,
                "epochs": epochs,
                "mean_vals": mean_vals,
                "std_vals": std_vals,
                "hticks": hticks,
                "min_vals": np.nanmin(diff_errs_arr, axis=0),
                "max_vals": np.nanmax(diff_errs_arr, axis=0),
                "running_min_min": running_min_min,
                "running_min_max": running_min_max,
                "best_epoch_idx": best_epoch_idx,
                "best_epochs_lst": np.array(best_epochs_lst),
                "label": label_with_stats,
                "color": color,
                "linestyle": linestyle,
                "markerstyle": markerstyle,
            }
        )

    return run_data





def prepare_timing_data_control(dn_models_dics_c, dn_models_dics, plot_params=None):
    """
    Aggregate timing statistics when training-time comes from a control model.

    Parameters
    ----------
    dn_models_dics_c : dict
        Nested dict of control model wrappers (for timing).
    dn_models_dics : dict
        Nested dict of main model wrappers (for epoch counts).
    plot_params : dict, optional
        Styling overrides for each configuration key.

    Returns
    -------
    list of dict
        One dict per configuration:
        - 'avg_epoch_time', 'std_epoch_time'
        - 'run_time_mean', 'run_time_std'
        - 'color', 'label'
    """
    if plot_params is None:
        plot_params = {}

    raw_timing_data = []
    model_keys = list(dn_models_dics.keys())
    dic_values_c = list(dn_models_dics_c.values())
    dic_values = list(dn_models_dics.values())

    for i, (key, model_dic_c, model_dic) in enumerate(
        zip(model_keys, dic_values_c, dic_values)
    ):
        model_wrappers = list(model_dic.values())
        model_wrappers_c = list(model_dic_c.values())

        sample_model = model_wrappers[0]
        sf = (
            sample_model.print_freq
            if len(sample_model.train_loss_list) - 3 > len(sample_model.epoch_times)
            else 1
        )
        color, linestyle, markerstyle, label = plot_params.get(
            key, [f"C{i}", "-", ".", f"Run {i+1}"]
        )

        epoch_times_arr = []
        for m_c in model_wrappers_c:
            arr = np.array(m_c.epoch_times, dtype=np.float64)
            p95 = np.percentile(arr, 95)
            clipped = arr[arr <= p95]
            epoch_times_arr.append(clipped)

        total_num_epochs = [len(m.epoch_times) for m in model_wrappers]

        mean_epoch_times = [np.mean(times) for times in epoch_times_arr]
        avg_epoch_time = np.mean(mean_epoch_times)
        std_epoch_time = np.std(mean_epoch_times)

        total_time_arr = [
            np.mean(times) * epoch_num * sf
            for times, epoch_num in zip(epoch_times_arr, total_num_epochs)
        ]
        run_time_mean = np.mean(total_time_arr)
        run_time_std = np.std(total_time_arr)

        raw_timing_data.append(
            {
                "label": label,
                "avg_epoch_time": avg_epoch_time,
                "std_epoch_time": std_epoch_time,
                "run_time_mean": run_time_mean,
                "run_time_std": run_time_std,
                "color": color,
            }
        )

    return raw_timing_data


def prepare_diffusion_mse_data(dn_models_dics, plot_params=None):
    """
    Compute final diffusion MSE statistics across runs for each configuration.

    Parameters
    ----------
    dn_models_dics : dict
        Nested dictionary of model wrappers, typically
        {config_key: {seed: modelWrapper, ...}, ...}.
        Each modelWrapper is expected to expose `best_diffusion_error`.
    plot_params : dict, optional
        Styling overrides for each configuration key.

    Returns
    -------
    list of dict
        Each dict has:
        - 'label': label for the configuration
        - 'avg_mse': mean of final diffusion MSE across repeats
        - 'std_mse': standard deviation across repeats
        - 'color': color associated with this configuration
    """
    if plot_params is None:
        plot_params = {}

    mse_data = []
    model_keys = list(dn_models_dics.keys())
    dic_values = list(dn_models_dics.values())

    for i, (key, model_dic) in enumerate(zip(model_keys, dic_values)):
        model_wrappers = list(model_dic.values())
        color, linestyle, markerstyle, label = plot_params.get(
            key, [f"C{i}", "-", ".", f"Run {i+1}"]
        )

        final_mses = [m.best_diffusion_error for m in model_wrappers]
        mean_mses = [np.mean(final_mse) for final_mse in final_mses]
        avg_mse = np.mean(mean_mses)
        std_mse = np.std(mean_mses)

        mse_data.append(
            {
                "label": label,
                "avg_mse": avg_mse,
                "std_mse": std_mse,
                "color": color,
            }
        )

    return mse_data





def prepare_diffusion_mse_data(dn_models_dics, plot_params=None):
    """
    Compute final diffusion MSE statistics across runs for each configuration.

    Parameters
    ----------
    dn_models_dics : dict
        Nested dictionary of model wrappers, typically
        {config_key: {seed: modelWrapper, ...}, ...}.
        Each modelWrapper is expected to expose `best_diffusion_error`.
    plot_params : dict, optional
        Styling overrides for each configuration key.

    Returns
    -------
    list of dict
        Each dict has:
        - 'label': label for the configuration
        - 'avg_mse': mean of final diffusion MSE across repeats
        - 'std_mse': standard deviation across repeats
        - 'min_mse': min of final diffusion MSE across repeats
        - 'max_mse': max of final diffusion MSE across repeats
        - 'color': color associated with this configuration
    """
    if plot_params is None:
        plot_params = {}

    mse_data = []
    model_keys = list(dn_models_dics.keys())
    dic_values = list(dn_models_dics.values())

    for i, (key, model_dic) in enumerate(zip(model_keys, dic_values)):
        model_wrappers = list(model_dic.values())
        color, linestyle, markerstyle, label = plot_params.get(
            key, [f"C{i}", "-", ".", f"Run {i+1}"]
        )

        # Each best_diffusion_error can be an array/tensor; we take its mean as a scalar per run
        final_mses = [m.best_diffusion_error for m in model_wrappers]
        mean_mses = [np.mean(final_mse) for final_mse in final_mses]

        mean_mses = np.array(mean_mses, dtype=np.float64)
        avg_mse = float(np.mean(mean_mses))
        std_mse = float(np.std(mean_mses))
        min_mse = float(np.min(mean_mses))
        max_mse = float(np.max(mean_mses))

        mse_data.append(
            {
                "label": label,
                "avg_mse": avg_mse,
                "std_mse": std_mse,
                "min_mse": min_mse,
                "max_mse": max_mse,
                "color": color,
            }
        )

    return mse_data




def prepare_model_run_data(dn_models_dics, plot_params=None):
    """
    Aggregate validation-loss data across runs for each model configuration.

    Parameters
    ----------
    dn_models_dics : dict
        Nested dictionary of model wrappers, typically
        {config_key: {seed: modelWrapper, ...}, ...}.
        Each modelWrapper is expected to have attributes:
        - val_loss_list
        - train_loss_list
        - epoch_times
        - best_val_loss
        - last_improved  (epoch index of best_val_loss)
    plot_params : dict, optional
        Plot styling overrides, mapping `config_key` to
        [color, linestyle, markerstyle, label].

    Returns
    -------
    tuple
        (run_data, raw_timing_data)

        run_data : list of dict
            One entry per configuration with keys like:
            - 'label', 'mean_final_loss', 'std_final_loss'
            - 'epochs', 'running_min_min', 'running_min_max'
            - 'best_epoch_idx', 'best_epochs_lst'
            - 'hticks', 'color', 'linestyle', 'markerstyle'
        raw_timing_data : list of dict
            Per-configuration timing statistics:
            - 'avg_epoch_time', 'std_epoch_time'
            - 'run_time_mean', 'run_time_std', 'color', 'label'
    """
    import numpy as np

    if plot_params is None:
        plot_params = {}

    run_data = []
    raw_timing_data = []
    model_keys = list(dn_models_dics.keys())
    dic_values = list(dn_models_dics.values())

    for i, (key, model_dic) in enumerate(zip(model_keys, dic_values)):
        model_wrappers = list(model_dic.values())
        sample_model = model_wrappers[0]
        sf = 1  # sampling factor; left at 1 (can be adapted if needed)

        # Extract individual plot parameters or use defaults
        color, linestyle, markerstyle, label = plot_params.get(
            key, [f"C{i}", "-", ".", f"Run {i+1}"]
        )
        val_losses_raw = []

        # Collect validation (or train) losses
        for m in model_wrappers:
            if len(m.val_loss_list):
                losses = np.array(m.val_loss_list, dtype=np.float64)
            else:
                losses = np.array(m.train_loss_list, dtype=np.float64)
            val_losses_raw.append(losses)

        # Collect epoch times (clipped at 80th percentile)
        epoch_times_arr = []
        for m in model_wrappers:
            arr = np.array(m.epoch_times, dtype=np.float64)
            p80 = np.percentile(arr, 80)
            clipped = arr[arr <= p80]
            epoch_times_arr.append(clipped)

        # Pad losses to same length and stack
        max_len_val = max(len(v) for v in val_losses_raw)
        val_losses = [
            np.pad(v, (0, max_len_val - len(v)), constant_values=np.nan)
            for v in val_losses_raw
        ]
        val_losses = np.vstack(val_losses)

        # Running minima across epochs per run
        running_mins = np.full_like(val_losses, np.nan)
        for j in range(val_losses.shape[0]):
            v = val_losses[j]
            running_mins[j] = np.minimum.accumulate(np.nan_to_num(v, nan=np.inf))

        running_min_min = np.nanmin(running_mins, axis=0)
        running_min_max = np.nanmax(running_mins, axis=0)
        epochs = np.arange(max_len_val)

        # Final-loss statistics (based on last valid value in each run)
        last_valid_vals = np.array(
            [
                row[~np.isnan(row)][-1] if np.any(~np.isnan(row)) else np.nan
                for row in val_losses
            ]
        )
        mean_final = np.nanmean(last_valid_vals)
        std_final = np.nanstd(last_valid_vals)

        # Per-run best-epoch and best-loss collections for summary ticks
        hticks_vals = []
        best_epochs_lst = []
        for idx, m in enumerate(model_wrappers):
            # epoch index of best_val_loss in the raw losses for this run
            best_val_epoch = np.argmin(
                np.abs(val_losses_raw[idx] - m.best_val_loss)
            )
            best_epochs_lst.append(best_val_epoch)
            hticks_vals.append(m.best_val_loss)
        hticks = np.mean(hticks_vals)

        # Best model per group (lowest best_val_loss across seeds)
        final_loss = [m.best_val_loss for m in model_wrappers]
        best_seed_idx = int(np.nanargmin(final_loss))
        best_model = model_wrappers[best_seed_idx]

        # Use .last_improved from the best model as the group best epoch
        # Assume 0-based; if it looks 1-based (>= max_len_val) and -1 fits, shift.
        best_epoch_idx = getattr(best_model, "last_improved", None)
        if best_epoch_idx is None:
            # Fallback: use epoch where this seed achieves best_val_loss
            best_epoch_idx = best_epochs_lst[best_seed_idx]

        best_epoch_idx = int(best_epoch_idx)

        # Heuristic for 1-based indexing: shift if needed
        if best_epoch_idx >= max_len_val and (best_epoch_idx - 1) < max_len_val:
            best_epoch_idx = best_epoch_idx - 1

        # Clip to valid range just in case
        best_epoch_idx = max(0, min(best_epoch_idx, max_len_val - 1))

        mean_epoch_times = [np.mean(times) for times in epoch_times_arr]
        avg_epoch_time = np.mean(mean_epoch_times)
        std_epoch_time = np.std(mean_epoch_times)

        total_num_epochs = [len(m.epoch_times) for m in model_wrappers]
        total_time_arr = [
            np.mean(times) * epoch_num * sf
            for times, epoch_num in zip(epoch_times_arr, total_num_epochs)
        ]
        run_time_mean = np.mean(total_time_arr)
        run_time_std = np.std(total_time_arr)

        mean_str = f"{mean_final:.2e}"
        std_str = f"{std_final:.1e}"
        label_with_stats = f"{label} ({mean_str} ± {std_str})"

        run_data.append(
            {
                "label": label,
                "mean_final_loss": np.mean(final_loss),
                "std_final_loss": np.std(final_loss),
                "mean_final": mean_final,
                "hticks": hticks,
                "best_epochs_lst": best_epochs_lst,
                "epochs": epochs,
                "running_min_min": running_min_min,
                "running_min_max": running_min_max,
                "best_epoch_idx": best_epoch_idx,      # <-- now from best seed's .last_improved
                "label_with_stats": label_with_stats,
                "color": color,
                "linestyle": linestyle,
                "markerstyle": markerstyle,
            }
        )

        raw_timing_data.append(
            {
                "label": label,
                "avg_epoch_time": avg_epoch_time,
                "std_epoch_time": std_epoch_time,
                "run_time_mean": run_time_mean,
                "run_time_std": run_time_std,
                "color": color,
            }
        )

    return run_data, raw_timing_data


def prepare_diff_run_data(dn_models_dics, plot_params=None):
    import numpy as np

    if plot_params is None:
        plot_params = {}

    run_data = []
    model_keys = list(dn_models_dics.keys())
    dic_values = list(dn_models_dics.values())

    # Determine scaling factor for epoch time based on print frequency
    sample_model = list(dic_values[0].values())[0]
    sf = sample_model.print_freq if len(sample_model.train_loss_list) > len(sample_model.epoch_times) else 1

    for i, (key, model_dic) in enumerate(zip(model_keys, dic_values)):
        model_wrappers = list(model_dic.values())

        # Use user-defined or default styling
        color, linestyle, markerstyle, label = plot_params.get(
            key, [f'C{i}', '-', '.', f'Run {i+1}']
        )

        best_epochs_lst = []
        diff_errs_raw = [[] for _ in range(len(model_wrappers))]
        diff_errs = []

        max_len = 0
        for m in model_wrappers:
            best_val_epoch = np.argmin(np.abs(np.array(m.val_loss_list) - m.best_val_loss))
            best_epochs_lst.append(best_val_epoch)
            max_len = max(max_len, best_val_epoch)
        max_len = max_len // sf + 1

        for j, m in enumerate(model_wrappers):
            try:
                total_num_epochs = len(m.train_loss_list)
                best_epochs_scaled = best_epochs_lst[j] // sf + 1
                padded = np.full(max_len, np.nan)
                for diff_err in m.diffusion_errors:
                    diff_errs_raw[j].append(torch.mean(diff_err[:, 0]).detach().cpu().numpy())
                padded[:best_epochs_scaled] = diff_errs_raw[j][:best_epochs_scaled]
                diff_errs.append(padded)
            except:
                diff_err = m.diffusion_errors[:best_epochs_lst[j] // sf]
                padded_diff_err = np.pad(np.array(diff_err), (0, max_len - len(diff_err)), constant_values=np.nan)
                diff_errs.append(padded_diff_err)

        diff_errs_arr = np.vstack(diff_errs)
        running_mins = np.minimum.accumulate(np.nan_to_num(diff_errs_arr, nan=np.inf), axis=1)
        running_min_min = np.nanmin(running_mins, axis=0)
        running_min_max = np.nanmax(running_mins, axis=0)
        best_epoch_idx = np.nanargmin(running_min_min)
        epochs = np.arange(max_len)

        mean_vals = np.nanmean(diff_errs_arr, axis=0)
        std_vals = np.nanstd(diff_errs_arr, axis=0)

    
        last_vals = np.array([mw.best_diffusion_error for mw in model_wrappers])


        mean_final = np.nanmean(last_vals)
        std_final = np.nanstd(last_vals)
        label_with_stats = f"{label}"# ({mean_final:.2e} ± {std_final:.1e})"

        run_data.append({
            'mean_final': mean_final,
            'epochs': epochs,
            'mean_vals': mean_vals,
            'std_vals': std_vals,
            'min_vals': np.nanmin(diff_errs_arr, axis=0),
            'max_vals': np.nanmax(diff_errs_arr, axis=0),
            'running_min_min': running_min_min,
            'running_min_max': running_min_max,
            'best_epoch_idx': best_epoch_idx,
            'best_epochs_lst': np.array(best_epochs_lst),
            'label': label_with_stats,
            'color': color,
            'linestyle': linestyle,
            'markerstyle': markerstyle
        })

    return run_data