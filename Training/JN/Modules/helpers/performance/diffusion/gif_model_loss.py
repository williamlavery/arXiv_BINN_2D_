import io
import itertools
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from ...utils import hist_properties
from .prepare_diff_mse import prepare_diff_run_data
from .plotting_mse_running_min import (
    _infer_print_freq,
    _gather_all_run_data,
    _merge_diff_plot_settings,
    _create_broken_x_axes,
    _format_broken_x_axes,
    _add_broken_axis_diagonals,
    _add_line_legend,
    _add_marker_legend,
    _marker_for_run,
)


def _plot_eval_D_multi_on_ax(
    ax,
    modelWrapper_dics,
    dataobjs,
    D_sym_true_lst,
    colors,
    labels,
    frame_idx,
    Dnum=1,
    num_bins=50,
    K=1700,
    fill=True,
    legend_pos=(0.5, 0.5),
    legend_ncols=2,
    legend_fontsize=12,
    linestyles=None,
    errs=None,
    xlim=None,
    ylim=None,
):
    """
    Same logic as plot_eval_D_multi, but:
      - draws into an existing axis `ax`
      - uses wrapper.diffusion_preds[frame_idx] to animate D(u) over epochs.
    """
    if errs:
        err_up, err_low = errs
    else:
        err_up = err_low = None

    # Allow passing a single dataobj
    if not isinstance(dataobjs, (list, tuple)):
        dataobjs = [dataobjs]

    colors = colors[: len(labels)]

    # Compute 5–95% u-percentiles for each data object
    hist_props_list = [hist_properties(d, num_bins) for d in dataobjs]
    low_us = [hp["low_count"] for hp in hist_props_list]
    high_us = [hp["high_count"] for hp in hist_props_list]

    results = []
    D_true_lst = []
    u_grids = []
    u_grids_K = []

    # If linestyles is None, reuse your default cycle
    if linestyles is None:
        linestyles = itertools.cycle(["-", ":", "-.", (0, (3, 1, 1, 1)), (0, (1, 1))])

    # ------------------------------------------------
    # Loop over configurations
    # ------------------------------------------------
    for wrapper_dic, color, label in zip(modelWrapper_dics.values(), colors, labels):
        # Sample model for THIS configuration
        sample_model = list(wrapper_dic.values())[0].model
        u_vals_np = sample_model.u_vals.flatten()

        u_grids.append(u_vals_np)
        u_grids_K.append(u_vals_np * K)

        D_ensemble = []

        for wrapper in wrapper_dic.values():
            # true D (does not depend on epoch)
            D_true_check = list(wrapper.model.D_true)
            if D_true_check not in D_true_lst:
                D_true_lst.append(D_true_check)

            # --- epoch-dependent prediction from diffusion_preds ---
            preds = wrapper.diffusion_preds
            if len(preds) == 0:
                continue

            # clip frame_idx just in case some runs are shorter
            idx = min(frame_idx, len(preds) - 1)
            diff_pred = np.asarray(preds[idx].detach().numpy()).flatten()  # shape [N_u]

            if diff_pred.shape[0] != u_vals_np.shape[0]:
                raise ValueError(
                    "wrapper.diffusion_preds entry length does not match u_vals grid."
                )

            D_ensemble.append(diff_pred[None, :])  # [1, N_u]

        if not D_ensemble:
            # no predictions for this config at this frame
            continue

        D_ensemble = np.concatenate(D_ensemble, axis=0)  # [runs, N_u]
        D_mean = D_ensemble.mean(axis=0)
        D_min = D_ensemble.min(axis=0)
        D_max = D_ensemble.max(axis=0)

        # we don't really need MSE here for plotting
        results.append((label, color, D_mean, D_min, D_max))

    # ------------------------------------------------
    # Plot on given axis
    # ------------------------------------------------
    # Ensemble predictions for each configuration
    for j, (label, color, D_mean, D_min, D_max) in enumerate(results):
        # handle linestyles whether it's a list or cycle
        try:
            ls = linestyles[j]
        except TypeError:
            ls = next(linestyles)

        u_grid_K = u_grids_K[j]

        ax.plot(u_grid_K, D_mean, lw=3, color=color, linestyle=ls, label=label)
        if fill:
            ax.fill_between(u_grid_K, D_min, D_max, alpha=0.4, color=color)

    # True curves + error bands + color-coded percentile lines
    for i, (D_true, D_sym, low_u, high_u, u_grid, u_grid_K) in enumerate(
        zip(D_true_lst, D_sym_true_lst, low_us, high_us, u_grids, u_grids_K)
    ):
        D_true = np.array(D_true)
        col = colors[i % len(colors)]

        # True curve
        if i == 0:
            ax.plot(
                u_grid_K,
                D_true,
                "--",
                lw=1,
                color="k",
                label=f"$D_{Dnum}(u)$",
                zorder=4,
            )
        else:
            ax.plot(
                u_grid_K,
                D_true,
                "--",
                lw=1,
                color="k",
                zorder=4,
            )

        # Error bands around true curve (all i, but only first with legend labels)
        if errs is not None:
            if i == 0:
                ax.plot(
                    u_grid_K,
                    D_true * (1 + err_up / 100),
                    "--",
                    lw=1,
                    color="r",
                    label=f"{err_up}%",
                    zorder=3,
                )
                ax.plot(
                    u_grid_K,
                    D_true * (1 - err_low / 100),
                    "-.",
                    lw=1,
                    color="r",
                    label=f"{err_low}%",
                    zorder=3,
                )
            else:
                ax.plot(
                    u_grid_K,
                    D_true * (1 + err_up / 100),
                    "--",
                    lw=1,
                    color="r",
                    zorder=3,
                )
                ax.plot(
                    u_grid_K,
                    D_true * (1 - err_low / 100),
                    "-.",
                    lw=1,
                    color="r",
                    zorder=3,
                )

        # Color-coded percentile lines for this dataset
        if i == 0:
            ax.axvline(
                low_u * K,
                color=col,
                ls="-.",
                lw=1,
                alpha=1,
                zorder=2,
                label="5% $u$-perc.",
            )
            ax.axvline(
                high_u * K,
                color=col,
                ls="--",
                lw=1,
                alpha=1,
                zorder=2,
                label="95% $u$-perc.",
            )
        else:
            ax.axvline(
                low_u * K,
                color=col,
                ls="-.",
                lw=1,
                alpha=1,
                zorder=2,
            )
            ax.axvline(
                high_u * K,
                color=col,
                ls="--",
                lw=1,
                alpha=1,
                zorder=2,
            )

    ax.set_xlabel(r"cell density [cells mm$^{-2}]$", fontsize=11)
    ax.set_ylabel(r"cell diffusion [mm$^2$ days$^{-1}$]", fontsize=11)
    ax.set_facecolor("white")
    ax.legend(
        loc="center",
        bbox_to_anchor=legend_pos,
        ncols=legend_ncols,
        fontsize=legend_fontsize,
    )

    # Fixed limits (to avoid jiggle between frames)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    # Prevent further autoscaling changes between frames
    ax.set_autoscale_on(False)
    
def save_gif_running_min_MSE_diff_loss_broken_x_log(
    models_dics_list,
    plot_params=None,
    plot_settings=None,
    out_path="running_min_diff_loss_broken_xaxis_loglog.gif",
    stride=1,
    dpi=120,
    fps=6,
    # --- real-space D(u) subplot args (all optional) ---
    modelWrapper_dics=None,
    dataobjs=None,
    D_sym_true_lst=None,
    colors=None,
    labels=None,
    Dnum=1,
    K=1700,
    num_bins=50,
    realspace_fill=True,
    legend_pos=(0.5, 0.5),
    legend_ncols=2,
    legend_fontsize=12,
    linestyles_real=None,
    errs=None,
    realspace_xlim=None,
    realspace_ylim=None,
):
    """
    Save an animated GIF of the running-min diffusion MSE loss for multiple
    experiment groups, using the same broken log–log x-axis layout as
    `plot_running_min_MSE_diff_loss_broken_x_log_lst`, and (optionally)
    a real-space D(u) subplot on the right.

    Left: broken x-axis loss vs epoch (two panels), drawn fully as a fixed
          backdrop; a vertical line moves to indicate the current epoch.
    Right: diffusion D(u) in real space (cell density vs diffusion).
    """
    import imageio.v2 as imageio  # lazy import

    # ---------- Settings ----------
    merged = _merge_diff_plot_settings(plot_settings)
    xaxis = merged["xaxis"]
    legend = merged["legend"]
    fill = merged["fill"]
    line_width = merged["line_width"]
    figsize = merged["figsize"]
    fontsizes = merged["fontsizes"]
    es_entries = merged["es_entries"]
    loss_ylim = merged["ylim"]  # y-limits for loss panels (can be None)

    # Marker / linestyle cycles (same as static plot)
    marker_styles = ["o", "s", "D", "^", "v", "P", "*", "X"]
    line_styles = ["-", "--", "-.", ":"]

    # ---------- Infer print frequency ----------
    sf = _infer_print_freq(models_dics_list)

    # ---------- Collect all runs ----------
    all_run_data, label_to_color = _gather_all_run_data(
        models_dics_list, plot_params, marker_styles, line_styles
    )

    if not all_run_data:
        raise ValueError("No run data found; cannot build GIF.")

    # ---------- Precompute x/y arrays for each run (full trajectories) ----------  ### NEW
    for run in all_run_data:
        epochs = np.asarray(run["epochs"])
        if len(epochs) == 0:
            run["_x_plot"] = np.array([])
            run["_y_mean"] = np.array([])
            run["_y_min"] = np.array([])
            run["_y_max"] = np.array([])
            continue

        x_full = np.where(epochs * sf == 0, 1e-1, epochs * sf)
        run["_x_plot"] = x_full
        run["_y_mean"] = np.asarray(run["mean_vals"])
        run["_y_min"] = np.asarray(run["min_vals"])
        run["_y_max"] = np.asarray(run["max_vals"])

    # ---------- Global, fixed y-limits for the loss panels (if not supplied) ----------
    if loss_ylim is None:
        all_y = []
        for run in all_run_data:
            all_y.extend(run["_y_min"])
            all_y.extend(run["_y_max"])
            all_y.extend(run["_y_mean"])
        if len(all_y) > 0:
            y_min = np.min(all_y)
            y_max = np.max(all_y)
            if y_max == y_min:
                # avoid zero-extent axis
                if y_min == 0:
                    y_min, y_max = -0.5, 0.5
                else:
                    y_min -= 0.5 * abs(y_min)
                    y_max += 0.5 * abs(y_max)
            pad = 0.05 * (y_max - y_min)
            loss_ylim = (y_min - pad, y_max + pad)

    # ---------- Global, fixed x-limits for epoch axes ----------
    x_all = []
    for run in all_run_data:
        x_plot = run.get("_x_plot", np.array([]))
        if len(x_plot) > 0:
            x_all.append(x_plot)
    if not x_all:
        raise ValueError("Runs contain zero epochs; cannot build GIF.")
    x_all = np.concatenate(x_all)
    global_x_min = float(np.min(x_all))
    global_x_max = float(np.max(x_all))

    # Maximum number of evaluation points across runs
    max_len = max(len(run["epochs"]) for run in all_run_data)
    if max_len == 0:
        raise ValueError("Runs contain zero epochs; cannot build GIF.")

    # Frame stepping
    step = max(1, int(stride))

    frames = []
    break_x = xaxis["break"]

    # ---------- Determine fixed real-space axis limits if not provided ----------
    do_realspace = (
        modelWrapper_dics is not None
        and dataobjs is not None
        and D_sym_true_lst is not None
        and colors is not None
        and labels is not None
    )

    if do_realspace and (realspace_xlim is None or realspace_ylim is None):
        all_x_real = []
        all_y_real = []

        for wrapper_dic in modelWrapper_dics.values():
            for wrapper in wrapper_dic.values():
                # x-coordinates (u grid in real units)
                u_vals_np = wrapper.model.u_vals.flatten() * K
                all_x_real.append(u_vals_np)

                # all predicted diffusion curves across epochs
                for preds in wrapper.diffusion_preds:
                    arr = np.asarray(preds.detach().numpy()).flatten()
                    all_y_real.append(arr)

        if all_x_real and realspace_xlim is None:
            x_all_r = np.concatenate(all_x_real)
            realspace_xlim = (x_all_r.min(), x_all_r.max())

        if all_y_real and realspace_ylim is None:
            y_all_r = np.concatenate(all_y_real)
            y_min = y_all_r.min()
            y_max = y_all_r.max()
            if y_max == y_min:
                if y_min == 0:
                    y_min, y_max = -0.5, 0.5
                else:
                    y_min -= 0.5 * abs(y_min)
                    y_max += 0.5 * abs(y_max)
            pad = 0.05 * (y_max - y_min)
            realspace_ylim = (y_min - pad, y_max + pad)

    # ---------- Build frames ----------
    for k in range(1, max_len + 1, step):
        # 1x3 layout: [ax1 | ax2 | axD]
        fig = plt.figure(figsize=(figsize[0] * 1.5, figsize[1]))  # a bit wider
        gs = fig.add_gridspec(
            1,
            3,
            width_ratios=[1, 5, 4],
            wspace=0.25,
        )
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1], sharey=ax1)
        axD = fig.add_subplot(gs[0, 2])

        # --- FIXED BACKDROP: plot full runs on loss axes (no truncation) ---   ### CHANGED
        for run in all_run_data:
            x = run["_x_plot"]
            if len(x) == 0:
                continue

            y_mean = run["_y_mean"]
            y_min = run["_y_min"]
            y_max = run["_y_max"]

            color = run["color"]
            linestyle = run.get("linestyle", "-")

            mask1 = x <= break_x
            mask2 = x > break_x

            # central line = mean (full trajectory)
            ax1.plot(
                x[mask1],
                y_mean[mask1],
                color=color,
                linestyle=linestyle,
                lw=line_width,
            )
            ax2.plot(
                x[mask2],
                y_mean[mask2],
                color=color,
                linestyle=linestyle,
                lw=line_width,
            )

            # min–max band (full)
            if fill:
                ax1.fill_between(
                    x[mask1],
                    y_min[mask1],
                    y_max[mask1],
                    alpha=0.2,
                    color=color,
                )
                ax2.fill_between(
                    x[mask2],
                    y_min[mask2],
                    y_max[mask2],
                    alpha=0.2,
                    color=color,
                )

            # --- Best-model marker (static, same on every frame) ---            ### CHANGED
            best_idx, best_seed = run.get("best_epoch_seed_idx", [None, None])
            if (
                best_idx is not None
                and 0 <= best_idx < len(run["epochs"])
                and best_seed is not None
                and 0 <= best_seed < len(run.get("final_mses", []))
            ):
                marker = _marker_for_run(run, es_entries)

                x_best = run["x_orig"][run["best_epoch_idx_orig"]]
                y_best = run["final_mses"][best_seed]

                target_ax = ax1 if x_best <= break_x else ax2
                target_ax.scatter(
                    x_best,
                    y_best,
                    facecolors=color,
                    edgecolors="black",
                    linewidths=0.8,
                    s=40,
                    zorder=5,
                    marker=marker,
                )

        # --- Formatting & decorations for loss panels ---
        _format_broken_x_axes(ax1, ax2, xaxis, fontsizes)
        _add_broken_axis_diagonals(ax1, ax2)

        # Fixed y-axis limits for loss
        if loss_ylim is not None:
            ax1.set_ylim(loss_ylim)
            ax2.set_ylim(loss_ylim)

        # Fixed x-axis limits for loss
        ax1.set_xscale("log")
        ax2.set_xscale("log")
        ax1.set_xlim(global_x_min, break_x)
        ax2.set_xlim(break_x, global_x_max)

        # Freeze autoscaling so backdrop never moves
        ax1.set_autoscale_on(False)
        ax2.set_autoscale_on(False)

        _add_line_legend(ax1, ax2, label_to_color, legend, line_width)
        _add_marker_legend(ax2, es_entries, legend)

        # --- Sliding vertical line: current epoch cursor ---------------------- ### NEW
        # Collect all epochs that exist at index k-1
        current_epochs = [
            run["epochs"][k - 1] for run in all_run_data if len(run["epochs"]) >= k
        ]
        if current_epochs:
            current_max_epoch = max(current_epochs)
            x_cursor = current_max_epoch * sf
            if x_cursor == 0:
                x_cursor = 1e-1

            # Draw cursor on the appropriate panel
            cursor_ax = ax1 if x_cursor <= break_x else ax2
            cursor_ax.axvline(
                x_cursor,
                color="k",
                linestyle="--",
                linewidth=1.5,
                zorder=6,
            )

        # --- Real-space D(u) subplot (if info provided) ---
        if do_realspace:
            frame_idx = k - 1  # 0-based index into wrapper.diffusion_preds

            _plot_eval_D_multi_on_ax(
                ax=axD,
                modelWrapper_dics=modelWrapper_dics,
                dataobjs=dataobjs,
                D_sym_true_lst=D_sym_true_lst,
                colors=colors,
                labels=labels,
                frame_idx=frame_idx,
                Dnum=Dnum,
                num_bins=num_bins,
                K=K,
                fill=realspace_fill,
                legend_pos=legend_pos,
                legend_ncols=legend_ncols,
                legend_fontsize=legend_fontsize,
                linestyles=linestyles_real,
                errs=errs,
                xlim=realspace_xlim,
                ylim=realspace_ylim,
            )
            axD.set_autoscale_on(False)
        else:
            axD.axis("off")

        # Global, fixed layout instead of tight_layout() to avoid jiggle
        fig.subplots_adjust(left=0.08, right=0.98, top=0.9, bottom=0.12, wspace=0.25)

        # Render frame → PNG buffer
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, facecolor="None")
        plt.close(fig)
        buf.seek(0)
        frames.append(imageio.imread(buf))

    print(f"Generated {len(frames)} frames for GIF.")
    print({f.shape for f in frames})

    # ---------- Write GIF ----------
    imageio.mimsave(out_path, frames, duration=1.0 / fps)
    print(f"GIF saved to: {out_path}")

    return {
        "out_path": out_path,
        "num_frames": len(frames),
        "num_runs": len(all_run_data),
        "xaxis": xaxis,
    }
