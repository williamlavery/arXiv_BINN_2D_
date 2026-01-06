import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Patch
import sys

from .run_data_preparation import (
    prepare_model_run_data,
    prepare_diff_run_data,
    prepare_timing_data_control,

)


def plot_running_min_MSE_diff_loss_broken_x_log_lst(
    models_dics_list,
    plot_params=None,
    plot_settings=None,
):
    """
    Plot running-min diffusion MSE loss for multiple experiment groups,
    in a style analogous to plot_running_min_val_loss_broken_x_log_lst.

    Each entry in `models_dics_list` is treated as a separate group and
    assigned a distinct marker and linestyle, while colors come from
    `plot_params` / configuration keys. All runs are combined and sorted by
    final performance (hticks).
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.lines import Line2D

    # ---------- Default settings (analogous to val-loss function) ----------
    default_settings = {
        "xaxis": {"min": 1, "max": 1e5, "break": 1e3},
        "legend": {
            "panel": 1,          # 1 → ax1, 2 → ax2
            "loc": (0.05, 0.95),
            "loc_upd": (0.35, 0.95),
            "fontsize": 10,
            "ncols": 1,
            "title": "$N_D$",   # legend title for lines
        },
        "name": "running_min_diff_loss_broken_xaxis_loglog.png",
        "fill": True,
        "line_lengths": {"hlength": 10000, "vlength_factor": 2.0},
        "line_widths_on_axis": {"hwidth": 1, "vwidth": 1},
        "line_width": 1.5,
        "fontsizes": {
            "xaxis": 12,
            "xtick_labels": 10,
            "yaxis": 12,
            "ytick_labels": 10,
        },
        "figsize": (7, 5),
        "es_entries": [
        ("1000", "o"),
        ("2000", "s"),
        ("3000", "D"),
        ]
    }

    plot_settings = plot_settings or {}
    settings = {**default_settings, **plot_settings}
    xaxis = {**default_settings["xaxis"], **settings.get("xaxis", {})}
    legend = {**default_settings["legend"], **settings.get("legend", {})}
    name = settings.get("name", default_settings["name"])
    fill = settings.get("fill", default_settings["fill"])
    line_width = settings.get("line_width", default_settings["line_width"])
    figsize = settings.get("figsize", default_settings["figsize"])
    line_width_on_axis = {
        **default_settings["line_widths_on_axis"],
        **settings.get("line_widths_on_axis", {}),
    }
    line_lengths = {
        **default_settings["line_lengths"],
        **settings.get("line_lengths", {}),
    }
    fontsizes = {**default_settings["fontsizes"], **settings.get("fontsizes", {})}
    es_entries = settings.get("es_entries", default_settings.get("es_entries", []))

    xfont = fontsizes["xaxis"]
    xtick_font = fontsizes["xtick_labels"]
    yfont = fontsizes["yaxis"]
    ytick_font = fontsizes["ytick_labels"]

    # Unused but kept for backward compatibility with settings
    hlength = line_lengths["hlength"]
    vlength_factor = line_lengths["vlength_factor"]
    hwidth = line_width_on_axis["hwidth"]
    vwidth = line_width_on_axis["vwidth"]

    break_x = xaxis["break"]
    x_min = xaxis["min"]
    x_max = xaxis["max"]

    # ---------- Marker / linestyle cycles (per group) ----------
    marker_styles = ["o", "s", "D", "^", "v", "P", "*", "X"]
    line_styles = ["-", "--", "-.", ":"]

    all_run_data = []
    label_to_color = {}  # for clean legend construction

    # ---------- Sample model to get print_freq (sf) ----------
    # Assumes nested dict structure as in your original code
    sample_model = list(list(models_dics_list[0].values())[0].values())[0]
    if len(sample_model.train_loss_list) > len(sample_model.epoch_times):
        sf = sample_model.print_freq
    else:
        sf = 1

    # ---------- Gather and annotate runs ----------
    for i, dn_models_dics in enumerate(models_dics_list):
        group_runs = prepare_diff_run_data(dn_models_dics, plot_params=plot_params)

        marker = marker_styles[i % len(marker_styles)]
        linestyle = line_styles[i % len(line_styles)]

        for run in group_runs:
            run["markerstyle"] = marker
            run["linestyle"] = linestyle

            label = run["label"]
            if label not in label_to_color:
                label_to_color[label] = run["color"]

        all_run_data.extend(group_runs)

    # Sort runs by final performance (hticks)
    all_run_data#.sort(key=lambda x: x["hticks"])

    # ---------- Figure with broken x-axis ----------
    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        sharey=True,
        figsize=figsize,
        gridspec_kw={"width_ratios": [1, 5], "wspace": 0.05},
    )

    # ---------- Plot lines and best-epoch markers ----------
    for run in all_run_data:
        # Epochs are multiplied by sf (print frequency), with 0 → 1e-1
        x = np.where(run["epochs"] * sf == 0, 1e-1, run["epochs"] * sf)

        # Running-min stats for diffusion MSE
        y_mean = run["mean_vals"]
        y_min = run["min_vals"]
        y_max = run["max_vals"]

        color = run["color"]
        linestyle = run.get("linestyle", "-")
        markerstyle = run.get("markerstyle", "o")

        mask1 = x <= break_x
        mask2 = x > break_x

        # central line = mean; shaded min–max band
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

        if fill:
            ax1.fill_between(
                x[mask1], y_min[mask1], y_max[mask1], alpha=0.2, color=color
            )
            ax2.fill_between(
                x[mask2], y_min[mask2], y_max[mask2], alpha=0.2, color=color
            )

        # Best-model marker: use best_epoch_idx if available, otherwise argmin hticks proxy
        best_idx = run.get("best_epoch_idx", None)
        if best_idx is None and len(y_mean) > 0:
            best_idx = int(np.argmin(y_mean))

        if best_idx is not None and 0 <= best_idx < len(x):
            ax2.scatter(
                x[best_idx],
                y_mean[best_idx],
                facecolors=color,
                edgecolors="black",
                linewidths=0.8,
                s=40,
                zorder=5,
                marker=markerstyle,
            )

    # ---------- Axes formatting ----------
    for ax in (ax1, ax2):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_facecolor("white")

    ax2.set_xlabel("epoch (log)", fontsize=xfont)
    ax1.set_xlim(left=x_min, right=break_x)
    ax2.set_xlim(left=break_x, right=x_max)
    ax1.set_ylabel(r"Diffusion MSE [mm$^4$ day$^{-2}$]", fontsize=yfont)

    ax1.spines["right"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax1.tick_params(labelright=False)
    ax2.tick_params(labelleft=False)

    ax1.yaxis.set_tick_params(labelsize=ytick_font)
    ax1.xaxis.set_tick_params(labelsize=xtick_font)
    ax2.xaxis.set_tick_params(labelsize=xtick_font)

    # ---------- Broken-axis diagonals ----------
    d = 0.015
    kwargs = dict(transform=ax1.transAxes, color="k", clip_on=False)
    ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)
    ax1.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
    kwargs.update(transform=ax2.transAxes)
    ax2.plot((-d, +d), (-d, +d), **kwargs)
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)

    # ---------- Legend for lines (unique labels, as in val-loss) ----------
    legend_title = legend.get("title", "$N_D$")

    sorted_labels = sorted(label_to_color.keys())
    line_handles = [
        Line2D(
            [0],
            [0],
            color=label_to_color[lbl],
            lw=line_width,
        )
        for lbl in sorted_labels
    ]

    line_legend_kwargs = dict(
        handles=line_handles,
        labels=sorted_labels,
        title=legend_title,
        fontsize=legend["fontsize"],
        title_fontsize=legend["fontsize"],
        ncols=legend["ncols"],
    )

    if legend["panel"] == 1:
        line_legend = ax1.legend(
            bbox_to_anchor=legend["loc"],
            **line_legend_kwargs,
        )
        ax1.add_artist(line_legend)
    else:
        line_legend = ax2.legend(
            bbox_to_anchor=legend["loc"],
            **line_legend_kwargs,
        )
        ax2.add_artist(line_legend)

    # ---------- Separate legend for ES markers (analogous to val-loss) ----------
    marker_handles = [
        Line2D(
            [0],
            [0],
            marker=mk,
            linestyle="",
            markerfacecolor="white",
            markeredgecolor="black",
            markeredgewidth=1.5,
            markersize=8,
        )
        for (label_es, mk) in es_entries
    ]
    marker_labels = [lab for (lab, _) in es_entries]

    marker_legend = ax2.legend(
        handles=marker_handles,
        labels=marker_labels,
        bbox_to_anchor=legend["loc_upd"],
        fontsize=legend["fontsize"],
        title="ES",
        title_fontsize=legend["fontsize"],
    )
    ax2.add_artist(marker_legend)

    fig.tight_layout()

    if name:
        plt.savefig(name, dpi=100, bbox_inches="tight", facecolor="None")
        print("saved plot:", name)
    plt.show()

    return {
        "name": name,
        "legend": legend,
        "fill": fill,
        "xaxis": xaxis,
        "num_runs": len(all_run_data),
    }

def plot_running_min_MSE_diff_loss_broken_x_log_lst(
    models_dics_list,
    plot_params=None,
    plot_settings=None,
):
    """
    Plot running-min diffusion MSE loss for multiple experiment groups.

    This version:
    - Does NOT use helper functions.
    - Aggregates diffusion_errors per label by padding with NaNs.
    - Computes pointwise min/mean/max across runs.
    - Chooses the "best" model per label using best_val_loss (np.argmin),
      and uses that model's last_improved to determine the best epoch index.
    - Uses colors from `plot_params` if provided, in a way compatible with the
      older implementation (per-label configs / color maps).
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.lines import Line2D

    # ---------- Default settings ----------
    default_settings = {
        "xaxis": {"min": 1, "max": 1e5, "break": 1e3},
        "legend": {
            "panel": 1,          # 1 → ax1, 2 → ax2
            "loc": (0.05, 0.95),
            "loc_upd": (0.35, 0.95),
            "fontsize": 10,
            "ncols": 1,
            "title": "$N_D$",   # legend title for lines
        },
        "name": "running_min_diff_loss_broken_xaxis_loglog.png",
        "fill": True,
        "line_lengths": {"hlength": 10000, "vlength_factor": 2.0},
        "line_widths_on_axis": {"hwidth": 1, "vwidth": 1},
        "line_width": 1.5,
        "fontsizes": {
            "xaxis": 12,
            "xtick_labels": 10,
            "yaxis": 12,
            "ytick_labels": 10,
        },
        "figsize": (7, 5),
        "es_entries": [
            ("1000", "o"),
            ("2000", "s"),
            ("3000", "D"),
        ],
    }

    plot_settings = plot_settings or {}
    settings = {**default_settings, **plot_settings}
    xaxis = {**default_settings["xaxis"], **settings.get("xaxis", {})}
    legend = {**default_settings["legend"], **settings.get("legend", {})}
    name = settings.get("name", default_settings["name"])
    fill = settings.get("fill", default_settings["fill"])
    line_width = settings.get("line_width", default_settings["line_width"])
    figsize = settings.get("figsize", default_settings["figsize"])
    line_width_on_axis = {
        **default_settings["line_widths_on_axis"],
        **settings.get("line_widths_on_axis", {}),
    }
    line_lengths = {
        **default_settings["line_lengths"],
        **settings.get("line_lengths", {}),
    }
    fontsizes = {**default_settings["fontsizes"], **settings.get("fontsizes", {})}
    es_entries = settings.get("es_entries", default_settings.get("es_entries", []))

    xfont = fontsizes["xaxis"]
    xtick_font = fontsizes["xtick_labels"]
    yfont = fontsizes["yaxis"]
    ytick_font = fontsizes["ytick_labels"]

    # Unused but kept for backward compatibility with settings
    hlength = line_lengths["hlength"]
    vlength_factor = line_lengths["vlength_factor"]
    hwidth = line_width_on_axis["hwidth"]
    vwidth = line_width_on_axis["vwidth"]

    break_x = xaxis["break"]
    x_min = xaxis["min"]
    x_max = xaxis["max"]

    # ---------- Marker / linestyle cycles (per group) ----------
    marker_styles = ["o", "s", "D", "^", "v", "P", "*", "X"]
    line_styles = ["-", "--"]

    # Basic color cycle (used as final fallback)
    default_colors = plt.rcParams.get("axes.prop_cycle", None)
    if default_colors is not None:
        default_colors = default_colors.by_key().get("color", ["C0", "C1", "C2", "C3"])
    else:
        default_colors = ["C0", "C1", "C2", "C3"]

    all_run_data = []
    label_to_color = {}  # for clean legend construction

    # ---------- Sample model to get print_freq (sf) ----------
    # Assumes nested dict structure as before: models_dics_list[g][...][...] = model
    sample_model = list(list(models_dics_list[0].values())[0].values())[0]
    if len(sample_model.train_loss_list) > len(sample_model.epoch_times):
        sf = sample_model.print_freq
    else:
        sf = 1

    # Helper to get a color for a given label, using plot_params like the
    # older helper-based implementation, with sensible fallbacks.
    def get_color_for_label(label, idx_for_fallback):
        """
        Priority:
        1. plot_params[label]["color"]  (per-label config dict)
        2. plot_params["colors"][label] (mapping label -> color)
        3. plot_params["colors"][idx]   (if "colors" is a list/tuple)
        4. default matplotlib color cycle (C0, C1, ...)
        """
        # 1 & 2 & 3: use plot_params if provided
        if isinstance(plot_params, dict):
            # Per-label dict: plot_params[label]["color"]
            if label in plot_params and isinstance(plot_params[label], dict):
                col = plot_params[label].get("color", None)
                if col is not None:
                    return col

            # Global "colors" entry
            colors_cfg = plot_params.get("colors", None)

            # Mapping label -> color
            if isinstance(colors_cfg, dict):
                if label in colors_cfg:
                    return colors_cfg[label]

            # Sequence of colors: index by idx_for_fallback
            if isinstance(colors_cfg, (list, tuple)) and len(colors_cfg) > 0:
                return colors_cfg[idx_for_fallback % len(colors_cfg)]

        # 4: fall back to matplotlib's default color cycle
        return default_colors[idx_for_fallback % len(default_colors)]

    # ---------- Gather and annotate runs (no helper) ----------
    #
    # For each "models_dics" (group), iterate over labels (keys),
    # collect all models under that label, aggregate their diffusion_errors,
    # and pick a best model via best_val_loss.
    #
    for i, models_dics in enumerate(models_dics_list):
        marker = marker_styles[i % len(marker_styles)]
        linestyle = line_styles[i % len(line_styles)]

        for label_key, runs_dict in models_dics.items():
            label = str(label_key)

            # Collect all diffusion_errors from runs under this label
            raw_errs = []
            models_for_label = []

            for m in runs_dict.values():
                if hasattr(m, "diffusion_errors"):
                    arr = np.array(m.diffusion_errors, dtype=float)
                    if arr.size > 0:
                        raw_errs.append(arr)
                        models_for_label.append(m)

            if len(raw_errs) == 0:
                # Nothing to plot for this label
                continue

            # Pad sequences with NaNs to equal length
            max_len = max(len(e) for e in raw_errs)
            errs = np.array(
                [
                    np.pad(e, (0, max_len - len(e)), constant_values=np.nan)
                    for e in raw_errs
                ]
            )

            # Pointwise stats across runs
            running_min = np.nanmin(errs, axis=0)
            running_max = np.nanmax(errs, axis=0)
            running_mean = np.nanmean(errs, axis=0)

            # Epoch / iteration axis, scaled by print frequency
            epochs = np.arange(max_len, dtype=float) * sf
            # Avoid zero on log-axis
            epochs = np.where(epochs == 0, 1e-1, epochs)

            color = label_to_color.get(label, None)
            if color is None:
                color = get_color_for_label(label, len(label_to_color))
                label_to_color[label] = color

            # ---- Find best model index using best_val_loss ----
            best_epoch_idx = None
            if len(models_for_label) > 0 and hasattr(models_for_label[0], "best_val_loss"):
                # Collect best_val_loss for each model in this label group
                best_val_losses = np.array(
                    [m.best_val_loss for m in models_for_label],
                    dtype=float,
                )
                # Index of model with minimal best_val_loss
                best_model_idx = int(np.argmin(best_val_losses))
                best_model = models_for_label[best_model_idx]

                # Map its last_improved (iteration) to an index into the aggregated curve
                if hasattr(best_model, "last_improved"):
                    idx_float = best_model.last_improved / sf
                    best_epoch_idx = int(
                        np.clip(np.round(idx_float), 0, max_len - 1)
                    )

            run = {
                "label": label,
                "epochs": epochs,
                "mean_vals": running_mean,
                "min_vals": running_min,
                "max_vals": running_max,
                "color": color,
                "markerstyle": marker,
                "linestyle": linestyle,
                "best_epoch_idx": best_epoch_idx,
            }

            all_run_data.append(run)

    # (Optional) sort runs by some metric if desired
    # all_run_data.sort(key=lambda x: x["hticks"])

    # ---------- Figure with broken x-axis ----------
    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        sharey=True,
        figsize=figsize,
        gridspec_kw={"width_ratios": [1, 5], "wspace": 0.05},
    )

    # ---------- Plot lines and best-epoch markers ----------
    for run in all_run_data:
        x = run["epochs"]
        y_mean = run["mean_vals"]
        y_min = run["min_vals"]
        y_max = run["max_vals"]

        color = run["color"]
        linestyle = run.get("linestyle", "-")
        markerstyle = run.get("markerstyle", "o")

        mask1 = x <= break_x
        mask2 = x > break_x

        # central line = mean; shaded min–max band
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

        if fill:
            ax1.fill_between(
                x[mask1], y_min[mask1], y_max[mask1], alpha=0.2, color=color
            )
            ax2.fill_between(
                x[mask2], y_min[mask2], y_max[mask2], alpha=0.2, color=color
            )

        # Best-model marker: use best_epoch_idx determined from best_val_loss
        best_idx = run.get("best_epoch_idx", None)
        if best_idx is not None and 0 <= best_idx < len(x):
            ax2.scatter(
                x[best_idx],
                y_mean[best_idx],
                facecolors=color,
                edgecolors="black",
                linewidths=0.8,
                s=40,
                zorder=5,
                marker=markerstyle,
            )

    # ---------- Axes formatting ----------
    for ax in (ax1, ax2):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_facecolor("white")

    ax2.set_xlabel("epoch (log)", fontsize=xfont)
    ax1.set_xlim(left=x_min, right=break_x)
    ax2.set_xlim(left=break_x, right=x_max)
    ax1.set_ylabel(r"Diffusion MSE [mm$^4$ day$^{-2}$]", fontsize=yfont)

    ax1.spines["right"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax1.tick_params(labelright=False)
    ax2.tick_params(labelleft=False)

    ax1.yaxis.set_tick_params(labelsize=ytick_font)
    ax1.xaxis.set_tick_params(labelsize=xtick_font)
    ax2.xaxis.set_tick_params(labelsize=xtick_font)

    # ---------- Broken-axis diagonals ----------
    d = 0.015
    kwargs = dict(transform=ax1.transAxes, color="k", clip_on=False)
    ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)
    ax1.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
    kwargs.update(transform=ax2.transAxes)
    ax2.plot((-d, +d), (-d, +d), **kwargs)
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)

    # ---------- Legend for lines (unique labels) ----------
    legend_title = legend.get("title", "$N_D$")

    sorted_labels = sorted(label_to_color.keys())
    line_handles = [
        Line2D(
            [0],
            [0],
            color=label_to_color[lbl],
            lw=line_width,
        )
        for lbl in sorted_labels
    ]

    line_legend_kwargs = dict(
        handles=line_handles,
        labels=sorted_labels,
        title=legend_title,
        fontsize=legend["fontsize"],
        title_fontsize=legend["fontsize"],
        ncols=legend["ncols"],
    )

    if legend["panel"] == 1:
        line_legend = ax1.legend(
            bbox_to_anchor=legend["loc"],
            **line_legend_kwargs,
        )
        ax1.add_artist(line_legend)
    else:
        line_legend = ax2.legend(
            bbox_to_anchor=legend["loc"],
            **line_legend_kwargs,
        )
        ax2.add_artist(line_legend)

    # ---------- Separate legend for ES markers ----------
    marker_handles = [
        Line2D(
            [0],
            [0],
            marker=mk,
            linestyle="",
            markerfacecolor="white",
            markeredgecolor="black",
            markeredgewidth=1.5,
            markersize=8,
        )
        for (label_es, mk) in es_entries
    ]
    marker_labels = [lab for (lab, _) in es_entries]

    marker_legend = ax2.legend(
        handles=marker_handles,
        labels=marker_labels,
        bbox_to_anchor=legend["loc_upd"],
        fontsize=legend["fontsize"],
        title="ES",
        title_fontsize=legend["fontsize"],
    )
    ax2.add_artist(marker_legend)

    fig.tight_layout()

    if name:
        plt.savefig(name, dpi=100, bbox_inches="tight", facecolor="None")
        print("saved plot:", name)
    plt.show()

    return {
        "name": name,
        "legend": legend,
        "fill": fill,
        "xaxis": xaxis,
        "num_runs": len(all_run_data),
    }
