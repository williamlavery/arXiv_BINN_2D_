import io
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio

# --- import DIFF helpers with aliases ----------------------------------------
from .diffusion.plotting_mse_running_min import (
    _infer_print_freq as _infer_print_freq_diff,
    _gather_all_run_data as _gather_all_run_data_diff,
    _merge_diff_plot_settings,
    _format_broken_x_axes as _format_broken_x_axes_diff,
    _add_broken_axis_diagonals as _add_broken_axis_diagonals_diff,
    _add_line_legend as _add_line_legend_diff,
    _add_marker_legend as _add_marker_legend_diff,
    _marker_for_run as _marker_for_run_diff,
)

from .diffusion.gif_model_loss import  _plot_eval_D_multi_on_ax

# --- import GROW helpers with aliases ----------------------------------------
from .growth.plotting_mse_running_min_growth import (
    _infer_print_freq as _infer_print_freq_grow,
    _gather_all_run_data as _gather_all_run_data_grow,
    _merge_grow_plot_settings,
    _format_broken_x_axes as _format_broken_x_axes_grow,
    _add_broken_axis_diagonals as _add_broken_axis_diagonals_grow,
    _add_line_legend as _add_line_legend_grow,
    _add_marker_legend as _add_marker_legend_grow,
    _marker_for_run as _marker_for_run_grow,
)

from .growth.gif_model_loss_growth import   _plot_eval_G_multi_on_ax

# assumes _plot_eval_D_multi_on_ax and _plot_eval_G_multi_on_ax are imported / defined
# from your existing code snippets.


def save_gif_running_min_MSE_diff_and_grow_broken_x_log(
    diff_models_dics_list,
    grow_models_dics_list,
    diff_plot_params=None,
    grow_plot_params=None,
    diff_plot_settings=None,
    grow_plot_settings=None,
    out_path="running_min_diff_and_grow_loss_broken_xaxis_loglog.gif",
    stride=1,
    dpi=120,
    fps=6,
    # --- real-space D(u) subplot args (all optional) ---
    diff_modelWrapper_dics=None,
    diff_dataobjs=None,
    D_sym_true_lst=None,
    # --- real-space G(u) subplot args (all optional) ---
    grow_modelWrapper_dics=None,
    grow_dataobjs=None,
    G_sym_true_lst=None,
    # --- common real-space styling ---
    D_colors=None,
    G_colors=None,
    labels=None,
    Dnum=1,
    Gnum=1,
    K=1700,
    num_bins=50,
    realspace_fill=True,
    legend_pos=(0.5, 0.5),
    legend_ncols=2,
    legend_fontsize=12,
    linestyles_real=None,
    errs_diff=None,
    errs_grow=None,
    realspace_xlim_diff=None,
    realspace_ylim_diff=None,
    realspace_xlim_grow=None,
    realspace_ylim_grow=None,
):
    """
    Create a single GIF with TWO synchronized rows:

        Row 1 (top):   diffusion running-min MSE + D(u) real-space
        Row 2 (bottom): growth   running-min MSE + G(u) real-space

    Each frame uses the same frame index (epoch index) in both rows.
    """

    # ---------------------------------------------------------------
    # 0. Basic settings + helpers
    # ---------------------------------------------------------------
    # DIFF settings
    merged_diff = _merge_diff_plot_settings(diff_plot_settings)
    xaxis_diff = merged_diff["xaxis"]
    legend_diff = merged_diff["legend"]
    fill_diff = merged_diff["fill"]
    line_width_diff = merged_diff["line_width"]
    figsize_diff = merged_diff["figsize"]
    fontsizes_diff = merged_diff["fontsizes"]
    es_entries_diff = merged_diff["es_entries"]
    loss_ylim_diff = merged_diff["ylim"]

    # GROW settings
    merged_grow = _merge_grow_plot_settings(grow_plot_settings)
    xaxis_grow = merged_grow["xaxis"]
    legend_grow = merged_grow["legend"]
    fill_grow = merged_grow["fill"]
    line_width_grow = merged_grow["line_width"]
    figsize_grow = merged_grow["figsize"]
    fontsizes_grow = merged_grow["fontsizes"]
    es_entries_grow = merged_grow["es_entries"]
    loss_ylim_grow = merged_grow["ylim"]

    # Markers / linestyle cycles (can be different for D/G if you want)
    marker_styles_diff = ["o", "s", "D", "^", "v", "P", "*", "X"]
    marker_styles_grow = ["o", "s", "G", "^", "v", "P", "*", "X"]
    line_styles = ["-", "--", "-.", ":"]

    # print frequency
    sf_diff = _infer_print_freq_diff(diff_models_dics_list)
    sf_grow = _infer_print_freq_grow(grow_models_dics_list)

    # ---------------------------------------------------------------
    # 1. Collect and precompute run data (DIFF)
    # ---------------------------------------------------------------
    diff_all_run_data, diff_label_to_color = _gather_all_run_data_diff(
        diff_models_dics_list, diff_plot_params, marker_styles_diff, line_styles
    )
    if not diff_all_run_data:
        raise ValueError("No diffusion run data found; cannot build GIF.")

    for run in diff_all_run_data:
        epochs = np.asarray(run["epochs"])
        if len(epochs) == 0:
            run["_x_plot"] = np.array([])
            run["_y_mean"] = np.array([])
            run["_y_min"] = np.array([])
            run["_y_max"] = np.array([])
            continue
        x_full = np.where(epochs * sf_diff == 0, 1e-1, epochs * sf_diff)
        run["_x_plot"] = x_full
        run["_y_mean"] = np.asarray(run["mean_vals"])
        run["_y_min"] = np.asarray(run["min_vals"])
        run["_y_max"] = np.asarray(run["max_vals"])

    # DIFF global y-limits
    if loss_ylim_diff is None:
        all_y = []
        for run in diff_all_run_data:
            all_y.extend(run["_y_min"])
            all_y.extend(run["_y_max"])
            all_y.extend(run["_y_mean"])
        if all_y:
            y_min = float(np.min(all_y))
            y_max = float(np.max(all_y))
            if y_max == y_min:
                if y_min == 0:
                    y_min, y_max = -0.5, 0.5
                else:
                    y_min -= 0.5 * abs(y_min)
                    y_max += 0.5 * abs(y_max)
            pad = 0.05 * (y_max - y_min)
            loss_ylim_diff = (y_min - pad, y_max + pad)

    # DIFF global x-limits
    x_all_diff = []
    for run in diff_all_run_data:
        x_plot = run.get("_x_plot", np.array([]))
        if len(x_plot) > 0:
            x_all_diff.append(x_plot)
    if not x_all_diff:
        raise ValueError("Diffusion runs contain zero epochs; cannot build GIF.")
    x_all_diff = np.concatenate(x_all_diff)
    global_x_min_diff = float(np.min(x_all_diff))
    global_x_max_diff = float(np.max(x_all_diff))
    max_len_diff = max(len(run["epochs"]) for run in diff_all_run_data)
    if max_len_diff == 0:
        raise ValueError("Diffusion runs contain zero epochs; cannot build GIF.")

    break_x_diff = xaxis_diff["break"]

    # ---------------------------------------------------------------
    # 2. Collect and precompute run data (GROW)
    # ---------------------------------------------------------------
    grow_all_run_data, grow_label_to_color = _gather_all_run_data_grow(
        grow_models_dics_list, grow_plot_params, marker_styles_grow, line_styles
    )
    if not grow_all_run_data:
        raise ValueError("No growth run data found; cannot build GIF.")

    for run in grow_all_run_data:
        epochs = np.asarray(run["epochs"])
        if len(epochs) == 0:
            run["_x_plot"] = np.array([])
            run["_y_mean"] = np.array([])
            run["_y_min"] = np.array([])
            run["_y_max"] = np.array([])
            continue
        x_full = np.where(epochs * sf_grow == 0, 1e-1, epochs * sf_grow)
        run["_x_plot"] = x_full
        run["_y_mean"] = np.asarray(run["mean_vals"])
        run["_y_min"] = np.asarray(run["min_vals"])
        run["_y_max"] = np.asarray(run["max_vals"])

    # GROW global y-limits
    if loss_ylim_grow is None:
        all_y = []
        for run in grow_all_run_data:
            all_y.extend(run["_y_min"])
            all_y.extend(run["_y_max"])
            all_y.extend(run["_y_mean"])
        if all_y:
            y_min = float(np.min(all_y))
            y_max = float(np.max(all_y))
            if y_max == y_min:
                if y_min == 0:
                    y_min, y_max = -0.5, 0.5
                else:
                    y_min -= 0.5 * abs(y_min)
                    y_max += 0.5 * abs(y_max)
            pad = 0.05 * (y_max - y_min)
            loss_ylim_grow = (y_min - pad, y_max + pad)

    # GROW global x-limits
    x_all_grow = []
    for run in grow_all_run_data:
        x_plot = run.get("_x_plot", np.array([]))
        if len(x_plot) > 0:
            x_all_grow.append(x_plot)
    if not x_all_grow:
        raise ValueError("Growth runs contain zero epochs; cannot build GIF.")
    x_all_grow = np.concatenate(x_all_grow)
    global_x_min_grow = float(np.min(x_all_grow))
    global_x_max_grow = float(np.max(x_all_grow))
    max_len_grow = max(len(run["epochs"]) for run in grow_all_run_data)
    if max_len_grow == 0:
        raise ValueError("Growth runs contain zero epochs; cannot build GIF.")

    break_x_grow = xaxis_grow["break"]

    # ---------------------------------------------------------------
    # 3. Optionally synchronize x-limits across DIFF and GROW
    # ---------------------------------------------------------------
    global_x_min = min(global_x_min_diff, global_x_min_grow)
    global_x_max = max(global_x_max_diff, global_x_max_grow)

    # use common number of frames so rows are synchronized
    max_len = max(max_len_diff, max_len_grow)
    step = max(1, int(stride))

    # ---------------------------------------------------------------
    # 4. Real-space axis limits for D(u) and G(u) if not provided
    # ---------------------------------------------------------------
    do_realspace_diff = (
        diff_modelWrapper_dics is not None
        and diff_dataobjs is not None
        and D_sym_true_lst is not None
        and D_colors is not None
        and labels is not None
    )

    do_realspace_grow = (
        grow_modelWrapper_dics is not None
        and grow_dataobjs is not None
        and G_sym_true_lst is not None
        and G_colors is not None
        and labels is not None
    )

    # D(u)
    if do_realspace_diff and (realspace_xlim_diff is None or realspace_ylim_diff is None):
        all_x_real = []
        all_y_real = []
        for wrapper_dic in diff_modelWrapper_dics.values():
            for wrapper in wrapper_dic.values():
                u_vals_np = wrapper.model.u_vals.flatten() * K
                all_x_real.append(u_vals_np)
                for preds in wrapper.diffusion_preds:
                    arr = np.asarray(preds.detach().numpy()).flatten()
                    all_y_real.append(arr)
        if all_x_real and realspace_xlim_diff is None:
            x_all_r = np.concatenate(all_x_real)
            realspace_xlim_diff = (x_all_r.min(), x_all_r.max())
        if all_y_real and realspace_ylim_diff is None:
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
            realspace_ylim_diff = (y_min - pad, y_max + pad)

    # G(u)
    if do_realspace_grow and (realspace_xlim_grow is None or realspace_ylim_grow is None):
        all_x_real = []
        all_y_real = []
        for wrapper_dic in grow_modelWrapper_dics.values():
            for wrapper in wrapper_dic.values():
                u_vals_np = wrapper.model.u_vals.flatten() * K
                all_x_real.append(u_vals_np)
                for preds in wrapper.growth_preds:
                    arr = np.asarray(preds.detach().numpy()).flatten()
                    all_y_real.append(arr)
        if all_x_real and realspace_xlim_grow is None:
            x_all_r = np.concatenate(all_x_real)
            realspace_xlim_grow = (x_all_r.min(), x_all_r.max())
        if all_y_real and realspace_ylim_grow is None:
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
            realspace_ylim_grow = (y_min - pad, y_max + pad)

    # ---------------------------------------------------------------
    # 5. Build frames (2 rows x 3 columns)
    # ---------------------------------------------------------------
    frames = []

    for k in range(1, max_len + 1, step):
        # Make figure a bit taller (two rows)
        fig = plt.figure(
            figsize=(figsize_diff[0] * 1.5, figsize_diff[1] * 2.0)
        )
        gs = fig.add_gridspec(
            2,
            3,
            height_ratios=[1, 1],
            width_ratios=[1, 5, 4],
            hspace=0.35,
            wspace=0.25,
        )

        # Row 1: DIFF
        ax1D = fig.add_subplot(gs[0, 0])
        ax2D = fig.add_subplot(gs[0, 1], sharey=ax1D)
        axD = fig.add_subplot(gs[0, 2])

        # Row 2: GROW
        ax1G = fig.add_subplot(gs[1, 0])
        ax2G = fig.add_subplot(gs[1, 1], sharey=ax1G)
        axG = fig.add_subplot(gs[1, 2])

        # ------------------ DIFF backdrop ------------------------
        for run in diff_all_run_data:
            x = run["_x_plot"]
            if len(x) == 0:
                continue
            y_mean = run["_y_mean"]
            y_min = run["_y_min"]
            y_max = run["_y_max"]
            color = run["color"]
            linestyle = run.get("linestyle", "-")

            mask1 = x <= break_x_diff
            mask2 = x > break_x_diff

            ax1D.plot(x[mask1], y_mean[mask1], color=color, linestyle=linestyle, lw=line_width_diff)
            ax2D.plot(x[mask2], y_mean[mask2], color=color, linestyle=linestyle, lw=line_width_diff)

            if fill_diff:
                ax1D.fill_between(x[mask1], y_min[mask1], y_max[mask1], alpha=0.2, color=color)
                ax2D.fill_between(x[mask2], y_min[mask2], y_max[mask2], alpha=0.2, color=color)

            best_idx, best_seed = run.get("best_epoch_seed_idx", [None, None])
            if (
                best_idx is not None
                and 0 <= best_idx < len(run["epochs"])
                and best_seed is not None
                and 0 <= best_seed < len(run.get("final_mses", []))
            ):
                marker = _marker_for_run_diff(run, es_entries_diff)
                x_best = run["x_orig"][run["best_epoch_idx_orig"]]
                y_best = run["final_mses"][best_seed]
                target_ax = ax1D if x_best <= break_x_diff else ax2D
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

        _format_broken_x_axes_diff(ax1D, ax2D, xaxis_diff, fontsizes_diff)
        _add_broken_axis_diagonals_diff(ax1D, ax2D)

        if loss_ylim_diff is not None:
            ax1D.set_ylim(loss_ylim_diff)
            ax2D.set_ylim(loss_ylim_diff)

        ax1D.set_xscale("log")
        ax2D.set_xscale("log")
        ax1D.set_xlim(global_x_min, break_x_diff)
        ax2D.set_xlim(break_x_diff, global_x_max)

        ax1D.set_autoscale_on(False)
        ax2D.set_autoscale_on(False)

        _add_line_legend_diff(ax1D, ax2D, diff_label_to_color, legend_diff, line_width_diff)
        _add_marker_legend_diff(ax2D, es_entries_diff, legend_diff)

        # sliding cursor for DIFF
        current_epochs_diff = [
            run["epochs"][k - 1]
            for run in diff_all_run_data
            if len(run["epochs"]) >= k
        ]
        if current_epochs_diff:
            current_max_epoch = max(current_epochs_diff)
            x_cursor = current_max_epoch * sf_diff
            if x_cursor == 0:
                x_cursor = 1e-1
            cursor_ax = ax1D if x_cursor <= break_x_diff else ax2D
            cursor_ax.axvline(
                x_cursor,
                color="k",
                linestyle="--",
                linewidth=1.5,
                zorder=6,
            )

        # ------------------ GROW backdrop ------------------------
        for run in grow_all_run_data:
            x = run["_x_plot"]
            if len(x) == 0:
                continue
            y_mean = run["_y_mean"]
            y_min = run["_y_min"]
            y_max = run["_y_max"]
            color = run["color"]
            linestyle = run.get("linestyle", "-")

            mask1 = x <= break_x_grow
            mask2 = x > break_x_grow

            ax1G.plot(x[mask1], y_mean[mask1], color=color, linestyle=linestyle, lw=line_width_grow)
            ax2G.plot(x[mask2], y_mean[mask2], color=color, linestyle=linestyle, lw=line_width_grow)

            if fill_grow:
                ax1G.fill_between(x[mask1], y_min[mask1], y_max[mask1], alpha=0.2, color=color)
                ax2G.fill_between(x[mask2], y_min[mask2], y_max[mask2], alpha=0.2, color=color)

            best_idx, best_seed = run.get("best_epoch_seed_idx", [None, None])
            if (
                best_idx is not None
                and 0 <= best_idx < len(run["epochs"])
                and best_seed is not None
                and 0 <= best_seed < len(run.get("final_mses", []))
            ):
                marker = _marker_for_run_grow(run, es_entries_grow)
                x_best = run["x_orig"][run["best_epoch_idx_orig"]]
                y_best = run["final_mses"][best_seed]
                target_ax = ax1G if x_best <= break_x_grow else ax2G
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

        _format_broken_x_axes_grow(ax1G, ax2G, xaxis_grow, fontsizes_grow)
        _add_broken_axis_diagonals_grow(ax1G, ax2G)

        if loss_ylim_grow is not None:
            ax1G.set_ylim(loss_ylim_grow)
            ax2G.set_ylim(loss_ylim_grow)

        ax1G.set_xscale("log")
        ax2G.set_xscale("log")
        ax1G.set_xlim(global_x_min, break_x_grow)
        ax2G.set_xlim(break_x_grow, global_x_max)

        ax1G.set_autoscale_on(False)
        ax2G.set_autoscale_on(False)

        _add_line_legend_grow(ax1G, ax2G, grow_label_to_color, legend_grow, line_width_grow)
        _add_marker_legend_grow(ax2G, es_entries_grow, legend_grow)

        # sliding cursor for GROW
        current_epochs_grow = [
            run["epochs"][k - 1]
            for run in grow_all_run_data
            if len(run["epochs"]) >= k
        ]
        if current_epochs_grow:
            current_max_epoch = max(current_epochs_grow)
            x_cursor = current_max_epoch * sf_grow
            if x_cursor == 0:
                x_cursor = 1e-1
            cursor_ax = ax1G if x_cursor <= break_x_grow else ax2G
            cursor_ax.axvline(
                x_cursor,
                color="k",
                linestyle="--",
                linewidth=1.5,
                zorder=6,
            )

        # ------------------ Real-space D(u) ----------------------
        frame_idx = k - 1
        if do_realspace_diff:
            _plot_eval_D_multi_on_ax(
                ax=axD,
                modelWrapper_dics=diff_modelWrapper_dics,
                dataobjs=diff_dataobjs,
                D_sym_true_lst=D_sym_true_lst,
                colors=D_colors,
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
                errs=errs_diff,
                xlim=realspace_xlim_diff,
                ylim=realspace_ylim_diff,
            )
            axD.set_autoscale_on(False)
        else:
            axD.axis("off")

        # ------------------ Real-space G(u) ----------------------
        if do_realspace_grow:
            _plot_eval_G_multi_on_ax(
                ax=axG,
                modelWrapper_dics=grow_modelWrapper_dics,
                dataobjs=grow_dataobjs,
                G_sym_true_lst=G_sym_true_lst,
                colors=G_colors,
                labels=labels,
                frame_idx=frame_idx,
                Gnum=Gnum,
                num_bins=num_bins,
                K=K,
                fill=realspace_fill,
                legend_pos=legend_pos,
                legend_ncols=legend_ncols,
                legend_fontsize=legend_fontsize,
                linestyles=linestyles_real,
                errs=errs_grow,
                xlim=realspace_xlim_grow,
                ylim=realspace_ylim_grow,
            )
            axG.set_autoscale_on(False)
        else:
            axG.axis("off")

        # Layout
        fig.subplots_adjust(left=0.07, right=0.98, top=0.93, bottom=0.08, wspace=0.25, hspace=0.35)

        # Save frame to buffer
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, facecolor="None")
        plt.close(fig)
        buf.seek(0)
        frames.append(imageio.imread(buf))

    print(f"Generated {len(frames)} frames for combined DIFF+GROW GIF.")

    # Write GIF
    imageio.mimsave(out_path, frames, duration=1.0 / fps)
    print(f"GIF saved to: {out_path}")

    return {
        "out_path": out_path,
        "num_frames": len(frames),
        "num_runs_diff": len(diff_all_run_data),
        "num_runs_grow": len(grow_all_run_data),
        "xaxis_diff": xaxis_diff,
        "xaxis_grow": xaxis_grow,
    }
