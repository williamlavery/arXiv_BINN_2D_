import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .prepare_model_loss import prepare_model_run_data


# -------------------------------------------------------------------------
# Configuration helpers
# -------------------------------------------------------------------------
DEFAULT_SETTINGS = {
    "xaxis": {"min": 1, "max": 1e5, "break": 1e3},
    "legend": {
        "panel": 1,          # 1 → ax1, 2 → ax2
        "loc": (0.05, 0.95),
        "loc_upd": (0.35, 0.95),
        "fontsize": 10,
        "ncols": 1,
        "title": "$N_u$",   # legend title for lines
    },
    "name": "running_min_val_loss_broken_xaxis_loglog.png",
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


def _merge_plot_settings(plot_settings):
    """Merge user-supplied plot_settings with DEFAULT_SETTINGS."""
    plot_settings = plot_settings or {}
    settings = {**DEFAULT_SETTINGS, **plot_settings}

    xaxis = {**DEFAULT_SETTINGS["xaxis"], **settings.get("xaxis", {})}
    legend = {**DEFAULT_SETTINGS["legend"], **settings.get("legend", {})}
    name = settings.get("name", DEFAULT_SETTINGS["name"])
    fill = settings.get("fill", DEFAULT_SETTINGS["fill"])
    line_width = settings.get("line_width", DEFAULT_SETTINGS["line_width"])
    figsize = settings.get("figsize", DEFAULT_SETTINGS["figsize"])

    line_width_on_axis = {
        **DEFAULT_SETTINGS["line_widths_on_axis"],
        **settings.get("line_widths_on_axis", {}),
    }
    line_lengths = {
        **DEFAULT_SETTINGS["line_lengths"],
        **settings.get("line_lengths", {}),
    }
    fontsizes = {
        **DEFAULT_SETTINGS["fontsizes"],
        **settings.get("fontsizes", {}),
    }
    es_entries = settings.get("es_entries", DEFAULT_SETTINGS.get("es_entries", []))

    return {
        "settings": settings,
        "xaxis": xaxis,
        "legend": legend,
        "name": name,
        "fill": fill,
        "line_width": line_width,
        "figsize": figsize,
        "line_width_on_axis": line_width_on_axis,
        "line_lengths": line_lengths,
        "fontsizes": fontsizes,
        "es_entries": es_entries,
    }


# -------------------------------------------------------------------------
# Data preparation helpers
# -------------------------------------------------------------------------
def _prepare_run_data_with_styles(models_dics_list, plot_params, marker_styles, line_styles):
    """
    Prepare run data from all model dictionaries and attach marker/linestyle.
    Also tag each run with an 'es_index' so we can map to es_entries.

    Returns:
        run_data_all (list[dict])
        label_to_color (dict[str, color])
    """
    run_data_all = []
    label_to_color = {}

    for i, models_dics in enumerate(models_dics_list):
        marker = marker_styles[i % len(marker_styles)]
        linestyle = line_styles[i % len(line_styles)]

        run_data, _ = prepare_model_run_data(models_dics, plot_params=plot_params)

        for run in run_data:
            run["markerstyle"] = marker
            run["linestyle"] = linestyle
            # ES index per group: models_dics_list[i] ↔ es_entries[i]
            run["es_index"] = i

            label = run["label"]
            if label not in label_to_color:
                label_to_color[label] = run["color"]

        run_data_all.extend(run_data)

    # If you ever want sorting by performance, uncomment:
    # run_data_all.sort(key=lambda x: x["hticks"])

    return run_data_all, label_to_color


# -------------------------------------------------------------------------
# Plotting helpers
# -------------------------------------------------------------------------
def _create_broken_x_axes(figsize):
    """Create a figure with two subplots sharing the y-axis (broken x-axis layout)."""
    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        sharey=True,
        figsize=figsize,
        gridspec_kw={"width_ratios": [1, 5], "wspace": 0.05},
    )
    return fig, ax1, ax2


def _marker_for_run(run, es_entries):
    """
    Choose marker for the best-epoch scatter based on es_entries and group index.

    models_dics_list[i] is associated with es_entries[i], so each run carries
    an 'es_index' telling us which ES setting it belongs to.
    """
    es_idx = run.get("es_index", None)
    if es_idx is not None and 0 <= es_idx < len(es_entries):
        # es_entries[es_idx] = (label_es, marker)
        return es_entries[es_idx][1]
    # Fallback to the group marker if out of range
    return run.get("markerstyle", "o")


def _plot_runs_on_broken_axes(ax1, ax2, run_data_all, break_x, fill, line_width, es_entries):
    """Plot all runs on the two axes, with optional fill between min and max."""
    for run in run_data_all:
        x = np.where(run["epochs"] == 0, 1e-1, run["epochs"])
        y_min = run["running_min_min"]
        color = run["color"]
        linestyle = run.get("linestyle", "-")

        mask1 = x <= break_x
        mask2 = x > break_x

        # No labels here → legend is built manually
        ax1.plot(
            x[mask1],
            y_min[mask1],
            color=color,
            linestyle=linestyle,
            lw=line_width,
        )
        ax2.plot(
            x[mask2],
            y_min[mask2],
            color=color,
            linestyle=linestyle,
            lw=line_width,
        )

        # Best-model marker (black outline), using ES-based marker
        best_idx = run["best_epoch_seed_idx"][0]
        if 0 <= best_idx < len(x):
            marker = _marker_for_run(run, es_entries)
            ax2.scatter(
                x[best_idx],
                y_min[best_idx],
                facecolors=color,
                edgecolors="black",
                linewidths=0.8,
                s=40,
                zorder=5,
                marker=marker,
            )

        if fill:
            y_max = run["running_min_max"]
            ax1.fill_between(
                x[mask1], y_min[mask1], y_max[mask1], alpha=0.2, color=color
            )
            ax2.fill_between(
                x[mask2], y_min[mask2], y_max[mask2], alpha=0.2, color=color
            )


def _format_axes(ax1, ax2, xaxis, fontsizes):
    """Set axis scales, limits, labels, and tick fonts."""
    x_min = xaxis["min"]
    x_max = xaxis["max"]
    break_x = xaxis["break"]

    xfont = fontsizes["xaxis"]
    xtick_font = fontsizes["xtick_labels"]
    yfont = fontsizes["yaxis"]
    ytick_font = fontsizes["ytick_labels"]

    for ax in (ax1, ax2):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_facecolor("white")

    ax2.set_xlabel("epoch (log)", fontsize=xfont)
    ax1.set_xlim(left=x_min, right=break_x)
    ax2.set_xlim(left=break_x, right=x_max)
    ax1.set_ylabel("running min val loss [a.u]", fontsize=yfont)

    ax1.spines["right"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax1.tick_params(labelright=False)
    ax2.tick_params(labelleft=False)

    ax1.yaxis.set_tick_params(labelsize=ytick_font)
    ax1.xaxis.set_tick_params(labelsize=xtick_font)
    ax2.xaxis.set_tick_params(labelsize=xtick_font)


def _add_broken_axis_diagonals(ax1, ax2):
    """Draw diagonal lines to indicate the broken x-axis."""
    d = 0.015
    kwargs = dict(transform=ax1.transAxes, color="k", clip_on=False)
    ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)
    ax1.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
    kwargs.update(transform=ax2.transAxes)
    ax2.plot((-d, +d), (-d, +d), **kwargs)
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)


# -------------------------------------------------------------------------
# Legend helpers
# -------------------------------------------------------------------------
def _add_line_legend(ax1, ax2, label_to_color, line_width, legend_cfg):
    """Add legend for colored lines (one entry per label)."""
    legend_title = legend_cfg.get("title", "N_u")
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
        fontsize=legend_cfg["fontsize"],
        title_fontsize=legend_cfg["fontsize"],
        ncols=legend_cfg["ncols"],
    )

    if legend_cfg["panel"] == 1:
        line_legend = ax1.legend(
            bbox_to_anchor=legend_cfg["loc"],
            **line_legend_kwargs,
        )
        ax1.add_artist(line_legend)
    else:
        line_legend = ax2.legend(
            bbox_to_anchor=legend_cfg["loc"],
            **line_legend_kwargs,
        )
        ax2.add_artist(line_legend)


def _add_es_marker_legend(ax2, es_entries, legend_cfg):
    """Add separate legend for ES marker shapes."""
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
        bbox_to_anchor=legend_cfg["loc_upd"],
        fontsize=legend_cfg["fontsize"],
        title="ES",
        title_fontsize=legend_cfg["fontsize"],
    )
    ax2.add_artist(marker_legend)


# -------------------------------------------------------------------------
# Main public function: VAL ONLY (unchanged)
# -------------------------------------------------------------------------
def plot_running_min_val_loss_broken_x_log_lst(
    models_dics_list, plot_params=None, plot_settings=None
):
    """
    Plot running-min validation loss for multiple experiment groups.

    Each entry in `models_dics_list` is treated as a separate group and
    assigned a distinct marker and linestyle, while colors come from
    `plot_params` / configuration keys. All runs are combined and sorted by
    final performance.

    The early-stopping entries (es_entries) are used both:
      * for the ES marker legend, and
      * to select the scatter marker on the best epoch of each group
        via positional mapping: models_dics_list[i] ↔ es_entries[i].
    """
    # --- Merge and unpack settings ---
    cfg = _merge_plot_settings(plot_settings)
    settings = cfg["settings"]
    xaxis = cfg["xaxis"]
    legend = cfg["legend"]
    name = cfg["name"]
    fill = cfg["fill"]
    line_width = cfg["line_width"]
    figsize = cfg["figsize"]
    line_width_on_axis = cfg["line_width_on_axis"]
    line_lengths = cfg["line_lengths"]
    fontsizes = cfg["fontsizes"]
    es_entries = cfg["es_entries"]

    # Unused but kept for backward compatibility
    hlength = line_lengths["hlength"]
    vlength_factor = line_lengths["vlength_factor"]
    hwidth = line_width_on_axis["hwidth"]
    vwidth = line_width_on_axis["vwidth"]
    _ = (hlength, vlength_factor, hwidth, vwidth)  # keep linters quiet

    break_x = xaxis["break"]
    x_min = xaxis["min"]
    x_max = xaxis["max"]
    _ = (x_min, x_max)  # used indirectly via _format_axes

    marker_styles = ["o", "s", "D", "^", "v", "P", "*", "X"]
    line_styles = ["-", "--", "-.", ":"]

    # --- Prepare run data with marker/linestyle & ES index annotations ---
    run_data_all, label_to_color = _prepare_run_data_with_styles(
        models_dics_list, plot_params, marker_styles, line_styles
    )

    # --- Create figure and axes ---
    fig, ax1, ax2 = _create_broken_x_axes(figsize)

    # --- Plot runs ---
    _plot_runs_on_broken_axes(
        ax1, ax2, run_data_all, break_x=break_x, fill=fill, line_width=line_width, es_entries=es_entries
    )

    # --- Format axes & diagonals ---
    _format_axes(ax1, ax2, xaxis, fontsizes)
    _add_broken_axis_diagonals(ax1, ax2)

    # --- Legends ---
    _add_line_legend(ax1, ax2, label_to_color, line_width, legend)
    _add_es_marker_legend(ax2, es_entries, legend)

    fig.tight_layout()

    # --- Save & show ---
    if name:
        plt.savefig(name, dpi=100, bbox_inches="tight", facecolor="None")
        print("saved plot:", name)
    plt.show()

    return {
        "name": name,
        "legend": legend,
        "fill": fill,
        "xaxis": xaxis,
        "num_runs": len(run_data_all),
    }


# -------------------------------------------------------------------------
# NEW: VAL + TRAIN, with purple training curves
# -------------------------------------------------------------------------
def plot_running_min_val_and_train_loss_broken_x_log_lst(
    models_dics_list, plot_params=None, plot_settings=None
):
    """
    Plot running-min validation *and* training loss for multiple experiment groups.

    - Validation curves use the original color scheme (per configuration).
    - Training curves are drawn in aesthetic purple tones with dashed lines.
    - Best epochs (validation and training) are marked with scatter points.

    Training quantities are read from the additional keys produced by
    `prepare_model_run_data`, such as:
        'train_epochs', 'train_running_min_min', 'train_running_min_max',
        'train_best_epoch_seed_idx', ...
    """
    # --- Merge and unpack settings ---
    cfg = _merge_plot_settings(plot_settings)
    settings = cfg["settings"]
    xaxis = cfg["xaxis"]
    legend = cfg["legend"]
    base_name = cfg["name"]
    fill = cfg["fill"]
    line_width = cfg["line_width"]
    figsize = cfg["figsize"]
    line_width_on_axis = cfg["line_width_on_axis"]
    line_lengths = cfg["line_lengths"]
    fontsizes = cfg["fontsizes"]
    es_entries = cfg["es_entries"]

    # Choose an output name distinct from the val-only plot if default is used
    if ("name" not in (plot_settings or {})) and "val_loss" in base_name:
        name = base_name.replace("val_loss", "val_and_train_loss")
    else:
        name = base_name

    # Unused but kept for backward compatibility
    hlength = line_lengths["hlength"]
    vlength_factor = line_lengths["vlength_factor"]
    hwidth = line_width_on_axis["hwidth"]
    vwidth = line_width_on_axis["vwidth"]
    _ = (hlength, vlength_factor, hwidth, vwidth)  # keep linters quiet

    break_x = xaxis["break"]
    x_min = xaxis["min"]
    x_max = xaxis["max"]
    _ = (x_min, x_max)  # used indirectly via _format_axes

    marker_styles = ["o", "s", "D", "^", "v", "P", "*", "X"]
    line_styles = ["-", "--", "-.", ":"]

    # --- Prepare run data with marker/linestyle & ES index annotations ---
    run_data_all, label_to_color = _prepare_run_data_with_styles(
        models_dics_list, plot_params, marker_styles, line_styles
    )

    # --- Create figure and axes ---
    fig, ax1, ax2 = _create_broken_x_axes(figsize)

    # --- First: plot validation curves as in the original function ---
    _plot_runs_on_broken_axes(
        ax1, ax2, run_data_all, break_x=break_x, fill=fill, line_width=line_width, es_entries=es_entries
    )

    # --- Then: overlay training curves in purple tones ---
    num_runs = max(1, len(run_data_all))
    purple_cmap = cm.get_cmap("Purples", num_runs + 2)

    for idx, run in enumerate(run_data_all):
        train_epochs = np.asarray(run.get("train_epochs", []))
        train_y_min = np.asarray(run.get("train_running_min_min", []))
        train_y_max = np.asarray(run.get("train_running_min_max", []))

        if train_epochs.size == 0 or train_y_min.size == 0:
            continue

        x_train = np.where(train_epochs == 0, 1e-1, train_epochs)
        color_train = purple_cmap(idx + 2)  # skip the very lightest tones
        linestyle_train = "--"

        mask1 = x_train <= break_x
        mask2 = x_train > break_x

        ax1.plot(
            x_train[mask1],
            train_y_min[mask1],
            color=color_train,
            linestyle=linestyle_train,
            lw=line_width * 0.9,
            alpha=0.95,
        )
        ax2.plot(
            x_train[mask2],
            train_y_min[mask2],
            color=color_train,
            linestyle=linestyle_train,
            lw=line_width * 0.9,
            alpha=0.95,
        )

        if fill and train_y_max.size == train_y_min.size:
            ax1.fill_between(
                x_train[mask1],
                train_y_min[mask1],
                train_y_max[mask1],
                alpha=0.12,
                color=color_train,
            )
            ax2.fill_between(
                x_train[mask2],
                train_y_min[mask2],
                train_y_max[mask2],
                alpha=0.12,
                color=color_train,
            )

        # Mark training best epoch with a slightly smaller purple marker
        train_best_idx = run.get("train_best_epoch_seed_idx", [0, 0])[0]
        if 0 <= train_best_idx < len(x_train):
            marker = _marker_for_run(run, es_entries)
            ax2.scatter(
                x_train[train_best_idx],
                train_y_min[train_best_idx],
                facecolors=color_train,
                edgecolors="black",
                linewidths=0.6,
                s=30,
                zorder=6,
                marker=marker,
            )

    # --- Format axes & diagonals ---
    _format_axes(ax1, ax2, xaxis, fontsizes)
    # Overwrite y-label to reflect val + train
    ax1.set_ylabel(
        "running min loss (val & train) [a.u]", fontsize=fontsizes["yaxis"]
    )
    _add_broken_axis_diagonals(ax1, ax2)

    # --- Legends ---
    # 1) line legend: which color ↔ config (based on validation colors)
    _add_line_legend(ax1, ax2, label_to_color, line_width, legend)

    # 2) ES marker legend
    _add_es_marker_legend(ax2, es_entries, legend)

    # 3) style legend: val vs train
    style_handles = [
        Line2D(
            [0],
            [0],
            color="black",
            linestyle="-",
            lw=line_width,
        ),
        Line2D(
            [0],
            [0],
            color=purple_cmap(num_runs + 1),
            linestyle="--",
            lw=line_width,
        ),
    ]
    style_labels = ["validation (color-coded)", "training (purple, dashed)"]

    style_legend = ax2.legend(
        handles=style_handles,
        labels=style_labels,
        loc="lower left",
        fontsize=legend["fontsize"],
        frameon=True,
    )
    ax2.add_artist(style_legend)

    fig.tight_layout()

    # --- Save & show ---
    if name:
        plt.savefig(name, dpi=100, bbox_inches="tight", facecolor="None")
        print("saved plot:", name)
    plt.show()

    return {
        "name": name,
        "legend": legend,
        "fill": fill,
        "xaxis": xaxis,
        "num_runs": len(run_data_all),
    }
