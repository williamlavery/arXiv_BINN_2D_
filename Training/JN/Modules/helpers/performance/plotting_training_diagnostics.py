"""
plotting_training_diagnostics.py

Plotting utilities for training diagnostics and experiment comparisons.

This module is part of a four-file utilities package:

- plotting_training_diagnostics.py
    * Training-diagnostics plots across runs and experiment groups:
      - running-min validation loss with broken log-x axes (single and lists),
      - running-min diffusion MSE analogues,
      - epoch and total-time bar charts,
      - final loss and diffusion-MSE comparison plots.

This file focuses purely on plotting and imports only from the preparation
and core utility modules, avoiding circular dependencies.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Patch
import sys

from .run_data_preparation import (
    prepare_model_run_data,
    prepare_diff_run_data,
    prepare_timing_data_control,
    prepare_diffusion_mse_data,
)


def plot_running_min_val_loss_broken_x_log(
    dn_models_dics, plot_params=None, plot_settings=None
):
    """
    Plot running-min validation loss with a broken log-x axis.

    The epoch axis is split at a chosen `break` point. For each configuration,
    the running-min loss curve is drawn and optionally filled between its
    min and max over repeats. Best-epoch markers and summary horizontal/vertical
    lines are also drawn.

    Parameters
    ----------
    dn_models_dics : dict
        Nested dict {config_key: {seed: modelWrapper}}.
    plot_params : dict, optional
        Styling overrides for each config key.
    plot_settings : dict, optional
        High-level plotting configuration; keys include:
        - 'xaxis': {'min', 'max', 'break'}
        - 'legend': {'panel', 'loc', 'fontsize', 'ncols'}
        - 'name': output filename
        - 'fill': bool
        - 'line_lengths': {'hlength', 'vlength_factor'}
        - 'line_widths_on_axis': {'hwidth', 'vwidth'}
        - 'line_width'
        - 'fontsizes': {'xaxis', 'xtick_labels', 'yaxis', 'ytick_labels'}

    Returns
    -------
    dict
        Summary of used plot settings (name, legend, fill, xaxis, num_runs).
    """
    default_settings = {
        "xaxis": {"min": 1, "max": 1e5, "break": 1e3},
        "legend": {"panel": 1, "loc": "lower left", "fontsize": 10, "ncols": 1},
        "name": "running_min_val_loss_broken_xaxis_loglog.png",
        "fill": True,
        "line_lengths": {"hlength": 10000, "vlength_factor": 2.0},
        "line_width": 1.5,
        "line_widths_on_axis": {"hwidth": 1, "vwidth": 1},
        "fontsizes": {
            "xaxis": 12,
            "xtick_labels": 10,
            "yaxis": 12,
            "ytick_labels": 10,
        },
    }

    plot_settings = plot_settings or {}
    settings = {**default_settings, **plot_settings}
    xaxis = {**default_settings["xaxis"], **settings.get("xaxis", {})}
    legend = {**default_settings["legend"], **settings.get("legend", {})}
    name = settings.get("name", default_settings["name"])
    fill = settings.get("fill", default_settings["fill"])
    line_width = settings.get("line_width", default_settings["line_width"])
    line_width_on_axis = {
        **default_settings["line_widths_on_axis"],
        **settings.get("line_widths_on_axis", {}),
    }
    line_lengths = {
        **default_settings["line_lengths"],
        **settings.get("line_lengths", {}),
    }
    fontsizes = {**default_settings["fontsizes"], **settings.get("fontsizes", {})}

    xfont = fontsizes["xaxis"]
    xtick_font = fontsizes["xtick_labels"]
    yfont = fontsizes["yaxis"]
    ytick_font = fontsizes["ytick_labels"]

    hlength = line_lengths["hlength"]
    hwidth = line_width_on_axis["hwidth"]
    vwidth = line_width_on_axis["vwidth"]
    vlength_factor = line_lengths["vlength_factor"]

    break_x = xaxis["break"]
    x_min = xaxis["min"]
    x_max = xaxis["max"]

    run_data, _ = prepare_model_run_data(dn_models_dics, plot_params=plot_params)
    run_data.sort(key=lambda x: x["hticks"])

    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        sharey=True,
        figsize=(8, 5),
        gridspec_kw={"width_ratios": [1, 5], "wspace": 0.05},
    )

    for run in run_data:
        x = np.where(run["epochs"] == 0, 1e-1, run["epochs"])
        y_min = run["running_min_min"]
        color = run["color"]

        linestyle = run.get("linestyle", "-")
        markerstyle = run.get("markerstyle", "-")

        mask1 = x <= break_x
        mask2 = x > break_x

        ax1.plot(
            x[mask1],
            y_min[mask1],
            color=color,
            label=run["label"],
            linestyle=linestyle,
            lw=line_width,
        )
        ax2.plot(
            x[mask2],
            y_min[mask2],
            color=color,
            label=run["label"],
            linestyle=linestyle,
            lw=line_width,
        )
        ax2.scatter(
            x[run["best_epoch_idx"]],
            y_min[run["best_epoch_idx"]],
            color=color,
            s=20,
            zorder=5,
            marker=markerstyle,
        )

        if fill:
            y_max = run["running_min_max"]
            ax1.fill_between(
                x[mask1], y_min[mask1], y_max[mask1], alpha=0.2, color=color
            )
            ax2.fill_between(
                x[mask2], y_min[mask2], y_max[mask2], alpha=0.2, color=color
            )

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

    d = 0.015
    kwargs = dict(transform=ax1.transAxes, color="k", clip_on=False)
    ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)
    ax1.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
    kwargs.update(transform=ax2.transAxes)
    ax2.plot((-d, +d), (-d, +d), **kwargs)
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)

    if legend["panel"] == 1:
        ax1.legend(
            fontsize=legend["fontsize"],
            loc=legend["loc"],
            ncols=legend["ncols"],
        )
    else:
        ax2.legend(
            fontsize=legend["fontsize"],
            loc=legend["loc"],
            ncols=legend["ncols"],
        )

    ax2_right = ax2.twinx()
    y_min, y_max = ax1.get_ylim()
    y_top = y_min * vlength_factor

    ax2_right.tick_params(
        axis="y",
        which="both",
        left=False,
        right=False,
        labelleft=False,
        labelright=False,
    )

    for run in run_data:
        linestyle = run.get("linestyle", "-")
        color = run["color"]

        ax2.hlines(
            y=run["hticks"],
            xmin=x_max - hlength,
            xmax=x_max,
            color=color,
            linestyle=linestyle,
            linewidth=hwidth,
            zorder=10,
        )

        best_epochs_lst = run["best_epochs_lst"]
        to_plot = np.min(best_epochs_lst)

        if to_plot < break_x:
            ax1.vlines(
                to_plot, y_min, y_top, color=color, linestyle=linestyle, lw=vwidth
            )
        ax2.vlines(
            to_plot, y_min, y_top, color=color, linestyle=linestyle, lw=vwidth
        )

    ax2.set_ylim(y_min)
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
        "num_runs": len(run_data),
    }


def plot_running_min_MSE_diff_loss_broken_x_log(
    dn_models_dics, plot_params=None, plot_settings=None
):
    """
    Plot running-min diffusion MSE with broken log-x axis.

    This is analogous to `plot_running_min_val_loss_broken_x_log` but uses
    diffusion-error histories prepared by `prepare_diff_run_data`.

    Parameters
    ----------
    dn_models_dics : dict
        Nested dict of model wrappers.
    plot_params : dict, optional
        Styling overrides for each config key.
    plot_settings : dict, optional
        Same structure as in `plot_running_min_val_loss_broken_x_log`.

    Returns
    -------
    None
    """
    default_settings = {
        "xaxis": {"min": 1, "max": 1e5, "break": 1e3},
        "legend": {"panel": 1, "loc": "lower left", "fontsize": 10, "ncols": 1},
        "name": "running_min_val_loss_broken_xaxis_loglog.png",
        "fill": True,
        "line_lengths": {"hlength": 10000, "vlength_factor": 2.0},
        "line_width": 1.5,
        "markersize": 4,
        "line_widths_on_axis": {"hwidth": 1, "vwidth": 1},
        "fontsizes": {
            "xaxis": 12,
            "xtick_labels": 10,
            "yaxis": 12,
            "ytick_labels": 10,
        },
    }

    plot_settings = plot_settings or {}
    settings = {**default_settings, **plot_settings}
    xaxis = {**default_settings["xaxis"], **settings.get("xaxis", {})}
    legend = {**default_settings["legend"], **settings.get("legend", {})}
    name = settings.get("name", default_settings["name"])
    fill = settings.get("fill", default_settings["fill"])
    line_width = settings.get("line_width", default_settings["line_width"])
    line_lengths = {
        **default_settings["line_lengths"],
        **settings.get("line_lengths", {}),
    }
    hlength = line_lengths["hlength"]
    vlength_factor = line_lengths["vlength_factor"]
    markersize = settings.get("markersize", 4)

    line_width_on_axis = {
        **default_settings["line_widths_on_axis"],
        **settings.get("line_widths_on_axis", {}),
    }
    fontsizes = {**default_settings["fontsizes"], **settings.get("fontsizes", {})}
    xfont = fontsizes["xaxis"]
    xtick_font = fontsizes["xtick_labels"]
    yfont = fontsizes["yaxis"]
    ytick_font = fontsizes["ytick_labels"]
    hwidth = line_width_on_axis["hwidth"]
    vwidth = line_width_on_axis["vwidth"]

    break_x = xaxis["break"]
    x_min = xaxis["min"]
    x_max = xaxis["max"]

    run_data = prepare_diff_run_data(dn_models_dics, plot_params)
    run_data.sort(key=lambda x: x["hticks"])

    sample_model = list(list(dn_models_dics.values())[0].values())[0]
    sf = (
        sample_model.print_freq
        if len(sample_model.train_loss_list) > len(sample_model.epoch_times)
        else 1
    )

    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        sharey=True,
        figsize=(8, 5),
        gridspec_kw={"width_ratios": [1, 5], "wspace": 0.05},
    )

    for run in run_data:
        x = np.where(run["epochs"] * sf == 0, 1e-1, run["epochs"] * sf)
        y_mean = run["mean_vals"]
        y_min = run["min_vals"]
        y_max = run["max_vals"]
        color = run["color"]

        markerstyle = run.get("markerstyle", "-")
        linestyle = run.get("linestyle", "-")

        mask1 = x <= break_x
        mask2 = x > break_x

        ax1.plot(
            x[mask1],
            y_mean[mask1],
            color=color,
            label=run["label"],
            lw=line_width,
            linestyle=linestyle,
            marker=markerstyle,
            markersize=markersize,
        )
        ax2.plot(
            x[mask2],
            y_mean[mask2],
            color=color,
            label=run["label"],
            lw=line_width,
            linestyle=linestyle,
            marker=markerstyle,
            markersize=markersize,
        )
        if fill:
            ax1.fill_between(
                x[mask1], y_min[mask1], y_max[mask1], alpha=0.2, color=color
            )
            ax2.fill_between(
                x[mask2], y_min[mask2], y_max[mask2], alpha=0.2, color=color
            )

    for ax in (ax1, ax2):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_facecolor("white")

    ax1.yaxis.set_tick_params(labelsize=ytick_font)
    ax1.xaxis.set_tick_params(labelsize=xtick_font)
    ax2.xaxis.set_tick_params(labelsize=xtick_font)

    ax2.set_xlabel("epoch (log)", fontsize=xfont)
    ax1.set_xlim(left=x_min, right=break_x)
    ax2.set_xlim(left=break_x, right=x_max)
    ax1.set_ylabel(r"Diffusion MSE [mm$^4$ day$^{-2}$]", fontsize=yfont)

    ax1.spines["right"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax1.tick_params(labelright=False)
    ax2.tick_params(labelleft=False)

    d = 0.015
    kwargs = dict(transform=ax1.transAxes, color="k", clip_on=False)
    ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)
    ax1.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
    kwargs.update(transform=ax2.transAxes)
    ax2.plot((-d, +d), (-d, +d), **kwargs)
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)

    if legend["panel"] == 1:
        ax1.legend(
            fontsize=legend["fontsize"],
            loc=legend["loc"],
            ncols=legend["ncols"],
        )
    else:
        ax2.legend(
            fontsize=legend["fontsize"],
            loc=legend["loc"],
            ncols=legend["ncols"],
        )

    ax2_right = ax2.twinx()
    y_min_val, y_max_val = ax1.get_ylim()
    y_top = y_min_val * vlength_factor

    ax2_right.set_yscale("log")
    ax2_right.set_ylim(ax2.get_ylim())
    ax2_right.tick_params(
        axis="y",
        which="both",
        left=False,
        right=False,
        labelleft=False,
        labelright=False,
    )

    for run in run_data:
        linestyle = run.get("linestyle", "-")
        color = run["color"]

        ax2.hlines(
            y=run["hticks"],
            xmin=x_max - hlength,
            xmax=x_max,
            color=color,
            linestyle=linestyle,
            linewidth=hwidth,
            zorder=10,
        )

        best_epochs_lst = run["best_epochs_lst"]
        to_plot = np.min(best_epochs_lst)

        if to_plot < break_x:
            ax1.vlines(
                to_plot, y_min_val, y_top, color=color, linestyle=linestyle, lw=vwidth
            )
        ax2.vlines(
            to_plot, y_min_val, y_top, color=color, linestyle=linestyle, lw=vwidth
        )

    ax2.set_ylim(y_min_val)
    fig.tight_layout()

    if name:
        plt.savefig(name, dpi=100, bbox_inches="tight", facecolor="None")
        print("saved plot:", name)
    plt.show()


def plot_epoch_and_total_times(dn_models_dics, plot_params=None, name=None):
    """
    Plot average epoch time (with error bars) and total run time (bars).

    Parameters
    ----------
    dn_models_dics : dict
        Nested dict of model wrappers used to compute timing.
    plot_params : dict, optional
        Styling overrides for each config key.
    name : str, optional
        Output filename (PNG). If None, figure is not saved.

    Returns
    -------
    None
    """
    if name is None:
        name = "case11_bar.png"

    _, raw_timing_data = prepare_model_run_data(dn_models_dics, plot_params=plot_params)

    x = np.arange(len(raw_timing_data))
    avg_times = [d["avg_epoch_time"] for d in raw_timing_data]
    std_times = [d["std_epoch_time"] for d in raw_timing_data]
    run_time_means = [d["run_time_mean"] for d in raw_timing_data]
    run_time_stds = [d["run_time_std"] for d in raw_timing_data]
    scatter_colors = [d["color"] for d in raw_timing_data]
    bar_labels = [d["label"] for d in raw_timing_data]

    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax2 = ax1.twinx()

    ax1.errorbar(
        x,
        avg_times,
        yerr=std_times,
        fmt="o",
        capsize=4,
        color="black",
        label="Avg Epoch Time",
    )
    ax1.set_ylabel("Avg Epoch Time [s]", fontsize=11, color="black")
    ax1.tick_params(axis="y", labelcolor="black")

    for i in range(len(x)):
        ax2.bar(
            x[i],
            run_time_means[i],
            yerr=run_time_stds[i],
            capsize=5,
            color=scatter_colors[i],
            edgecolor="gray",
            alpha=0.4,
            error_kw=dict(ecolor="gray", alpha=1, linewidth=1.5),
        )

    ax2.set_ylabel("Total Run Time [s]", fontsize=11, color=scatter_colors[0])
    ax2.tick_params(axis="y", labelcolor=scatter_colors[0])

    ax1.set_xticks(x)
    ax1.set_xticklabels(bar_labels, fontsize=11)

    fig.tight_layout()
    if name:
        plt.savefig(name, dpi=100, bbox_inches="tight", facecolor="None")
        print("saved plot:", name)
    plt.show()


def plot_epoch_and_total_times_control(
    dn_models_dics_c, dn_models_dics, plot_params=None, name=None
):
    """
    Plot epoch and total training times using separate control models.

    Timing statistics are computed from `dn_models_dics_c` while epoch counts
    come from `dn_models_dics`.

    Parameters
    ----------
    dn_models_dics_c : dict
        Control model wrappers for timing.
    dn_models_dics : dict
        Main model wrappers (epoch counts).
    plot_params : dict, optional
        Styling overrides.
    name : str, optional
        Output filename (PNG). If None, figure is not saved.

    Returns
    -------
    None
    """
    raw_timing_data = prepare_timing_data_control(
        dn_models_dics_c, dn_models_dics, plot_params=plot_params
    )

    x = np.arange(len(raw_timing_data))
    avg_times = [d["avg_epoch_time"] for d in raw_timing_data]
    std_times = [d["std_epoch_time"] for d in raw_timing_data]
    run_time_means = [d["run_time_mean"] for d in raw_timing_data]
    run_time_stds = [d["run_time_std"] for d in raw_timing_data]
    scatter_colors = [d["color"] for d in raw_timing_data]
    bar_labels = [d["label"] for d in raw_timing_data]

    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax2 = ax1.twinx()

    ax1.errorbar(
        x,
        avg_times,
        yerr=std_times,
        fmt="o",
        capsize=4,
        color="black",
        label="Avg Epoch Time",
    )
    ax1.set_ylabel("Avg Epoch Time [s]", fontsize=11, color="black")
    ax1.tick_params(axis="y", labelcolor="black")

    for i in range(len(x)):
        ax2.bar(
            x[i],
            run_time_means[i],
            yerr=run_time_stds[i],
            capsize=5,
            color=scatter_colors[i],
            edgecolor="gray",
            alpha=0.4,
            error_kw=dict(ecolor="gray", alpha=1, linewidth=1.5),
        )

    ax2.set_ylabel("Total Run Time [s]", fontsize=11, color=scatter_colors[0])
    ax2.tick_params(axis="y", labelcolor=scatter_colors[0])

    ax1.set_xticks(x)
    ax1.set_xticklabels(bar_labels, fontsize=11)

    fig.tight_layout()
    if name:
        plt.savefig(name, dpi=100, bbox_inches="tight", facecolor="None")
        print("saved plot:", name)
    plt.show()


def plot_running_min_val_loss_broken_x_log_lst(
    models_dics_list, plot_params=None, plot_settings=None
):
    """
    Plot running-min validation loss for multiple experiment groups.

    Each entry in `models_dics_list` is treated as a separate group and
    assigned a distinct marker and linestyle, while colors come from
    `plot_params` / configuration keys. All runs are combined and sorted by
    final performance.

    Parameters
    ----------
    models_dics_list : list
        List of nested dicts, each of the form {config_key: {seed: modelWrapper}}.
    plot_params : dict, optional
        Styling overrides for each config key.
    plot_settings : dict, optional
        Similar to `plot_running_min_val_loss_broken_x_log` with additional
        'figsize'.

    Returns
    -------
    dict
        Summary of used settings and number of runs.
    """
    default_settings = {
        "xaxis": {"min": 1, "max": 1e5, "break": 1e3},
        "legend": {
            "panel": 1,
            "loc": (0.05, 0.95),
            "fontsize": 10,
            "ncols": 1,
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

    xfont = fontsizes["xaxis"]
    xtick_font = fontsizes["xtick_labels"]
    yfont = fontsizes["yaxis"]
    ytick_font = fontsizes["ytick_labels"]

    hlength = line_lengths["hlength"]
    hwidth = line_width_on_axis["hwidth"]
    vwidth = line_width_on_axis["vwidth"]
    vlength_factor = line_lengths["vlength_factor"]

    break_x = xaxis["break"]
    x_min = xaxis["min"]
    x_max = xaxis["max"]

    marker_styles = ["o", "s", "D", "^", "v", "P", "*", "X"]
    line_styles = ["-", "--"]

    run_data_all = []

    for i, dn_models_dics in enumerate(models_dics_list):
        marker = marker_styles[i % len(marker_styles)]
        linestyle = line_styles[i % len(line_styles)]

        run_data, _ = prepare_model_run_data(dn_models_dics, plot_params=plot_params)

        for run in run_data:
            run["markerstyle"] = marker
            run["linestyle"] = linestyle

        run_data_all.extend(run_data)

    run_data_all.sort(key=lambda x: x["hticks"])

    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        sharey=True,
        figsize=figsize,
        gridspec_kw={"width_ratios": [1, 5], "wspace": 0.05},
    )

    for run in run_data_all:
        x = np.where(run["epochs"] == 0, 1e-1, run["epochs"])
        y_min = run["running_min_min"]
        color = run["color"]
        linestyle = run.get("linestyle", "-")
        markerstyle = run.get("markerstyle", "o")

        mask1 = x <= break_x
        mask2 = x > break_x

        label = run["label"] if linestyle == "-" else None

        ax1.plot(
            x[mask1],
            y_min[mask1],
            color=color,
            label=label,
            linestyle=linestyle,
            lw=line_width,
        )
        ax2.plot(
            x[mask2],
            y_min[mask2],
            color=color,
            label=label,
            linestyle=linestyle,
            lw=line_width,
        )

        ax2.scatter(
            x[run["best_epoch_idx"]],
            y_min[run["best_epoch_idx"]],
            color=color,
            s=20,
            zorder=5,
            marker=markerstyle,
        )

        if fill:
            y_max = run["running_min_max"]
            ax1.fill_between(
                x[mask1], y_min[mask1], y_max[mask1], alpha=0.2, color=color
            )
            ax2.fill_between(
                x[mask2], y_min[mask2], y_max[mask2], alpha=0.2, color=color
            )

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

    d = 0.015
    kwargs = dict(transform=ax1.transAxes, color="k", clip_on=False)
    ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)
    ax1.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
    kwargs.update(transform=ax2.transAxes)
    ax2.plot((-d, +d), (-d, +d), **kwargs)
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)

    if legend["panel"] == 1:
        ax1.legend(
            fontsize=legend["fontsize"],
            bbox_to_anchor=legend["loc"],
            ncols=legend["ncols"],
        )
    else:
        ax2.legend(
            fontsize=legend["fontsize"],
            bbox_to_anchor=legend["loc"],
            ncols=legend["ncols"],
        )

    ax2_right = ax2.twinx()
    y_min_axis, y_max_axis = ax1.get_ylim()
    y_top = y_min_axis * vlength_factor
    ax2_right.tick_params(
        axis="y",
        which="both",
        left=False,
        right=False,
        labelleft=False,
        labelright=False,
    )

    for run in run_data_all:
        linestyle = run.get("linestyle", "-")
        color = run["color"]

        if linestyle == "-":
            ax2.hlines(
                y=run["hticks"],
                xmin=x_max - hlength,
                xmax=x_max,
                color=color,
                linestyle=linestyle,
                linewidth=hwidth,
                zorder=10,
            )

            best_epochs_lst = run["best_epochs_lst"]
            to_plot = np.min(best_epochs_lst)

            if to_plot < break_x:
                ax1.vlines(
                    to_plot,
                    y_min_axis,
                    y_top,
                    color=color,
                    linestyle=linestyle,
                    lw=vwidth,
                )
            ax2.vlines(
                to_plot,
                y_min_axis,
                y_top,
                color=color,
                linestyle=linestyle,
                lw=vwidth,
            )

    ax2.set_ylim(y_min_axis)
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
        "num_runs": len(run_data_all),
    }


def plot_diff_mse_lst(
    models_dics_list, label_list=None, plot_params=None, name=None
):
    """
    Compare diffusion MSE across multiple experiment groups.

    Each group is represented with a distinct hatch pattern, and MSE is
    shown on a log scale with error bars.

    Parameters
    ----------
    models_dics_list : list
        List of nested dicts for each experiment group.
    label_list : list, optional
        Human-readable labels for each group.
    plot_params : dict, optional
        Styling overrides for each config key.
    name : str, optional
        Output filename (PNG). If None, figure is not saved.

    Returns
    -------
    None
    """
    if label_list is None:
        label_list = [f"Group {i+1}" for i in range(len(models_dics_list))]

    hatch_patterns = ("", "\\", "|", "-", "+", "x", "o", "O", ".", "*")

    grouped_data = []
    all_labels_set = set()

    for dn_models_dics in models_dics_list:
        mse_data = prepare_diffusion_mse_data(dn_models_dics, plot_params=plot_params)
        label_to_data = {d["label"]: d for d in mse_data}
        grouped_data.append(label_to_data)
        all_labels_set.update(label_to_data.keys())

    all_labels = sorted(list(all_labels_set), key=float)
    num_groups = len(models_dics_list)
    num_bars = len(all_labels)
    bar_width = 0.8 / num_groups
    group_offsets = np.linspace(-0.4 + bar_width / 2, 0.4 - bar_width / 2, num_groups)

    fig, ax1 = plt.subplots(figsize=(10, 5))

    for i, label in enumerate(all_labels):
        for j, group_dict in enumerate(grouped_data):
            if label not in group_dict:
                continue

            d = group_dict[label]
            x_pos = i + group_offsets[j]

            ax1.bar(
                x_pos,
                d["avg_mse"],
                yerr=d["std_mse"],
                width=bar_width,
                capsize=5,
                color=d.get("color", "gray"),
                edgecolor="gray",
                hatch=hatch_patterns[j % len(hatch_patterns)],
                alpha=0.5,
                error_kw=dict(ecolor="gray", alpha=1, linewidth=1.5),
            )

    custom_handles = [
        Patch(
            facecolor="lightgray",
            edgecolor="gray",
            hatch=hatch_patterns[i % len(hatch_patterns)],
            label=label_list[i],
        )
        for i in range(num_groups)
    ]

    ax1.legend(
        custom_handles,
        label_list,
        loc="center",
        bbox_to_anchor=(0.25, 0.85),
        fontsize=16,
    )

    ax1.set_ylabel("Diffusion MSE [day$^{-2}$ mm$^{4}$]", fontsize=14, color="k")
    ax1.tick_params(axis="y", labelcolor="k", labelsize=14)

    ax1.set_xticks(np.arange(num_bars))
    ax1.set_xticklabels(all_labels, rotation=45, ha="right", fontsize=14)

    ax1.set_yscale("log")

    fig.tight_layout()
    if name:
        plt.savefig(name, dpi=100, bbox_inches="tight", facecolor="None")
        print("saved plot:", name)
    plt.show()


def plot_running_min_MSE_diff_loss_broken_x_log_lst(
    models_dics_list, plot_params=None, plot_settings=None
):
    """
    Plot running-min diffusion MSE for multiple experiment groups.

    Each group is given a different marker and linestyle, analogous to
    `plot_running_min_val_loss_broken_x_log_lst`, but using diffusion-error
    histories prepared by `prepare_diff_run_data`.

    Parameters
    ----------
    models_dics_list : list
        List of nested dicts for each experiment group.
    plot_params : dict, optional
        Styling overrides.
    plot_settings : dict, optional
        Same structure as in `plot_running_min_MSE_diff_loss_broken_x_log`.

    Returns
    -------
    None
    """
    default_settings = {
        "xaxis": {"min": 1, "max": 1e5, "break": 1e3},
        "legend": {"panel": 1, "loc": "lower left", "fontsize": 10, "ncols": 1},
        "name": "running_min_diff_loss_broken_xaxis_loglog.png",
        "fill": True,
        "line_lengths": {"hlength": 10000, "vlength_factor": 2.0},
        "line_width": 1.5,
        "markersize": 4,
        "line_widths_on_axis": {"hwidth": 1, "vwidth": 1},
        "fontsizes": {
            "xaxis": 12,
            "xtick_labels": 10,
            "yaxis": 12,
            "ytick_labels": 10,
        },
    }

    plot_settings = plot_settings or {}
    settings = {**default_settings, **plot_settings}
    xaxis = {**default_settings["xaxis"], **settings.get("xaxis", {})}
    legend = {**default_settings["legend"], **settings.get("legend", {})}
    name = settings.get("name", default_settings["name"])
    fill = settings.get("fill", default_settings["fill"])
    line_width = settings.get("line_width", default_settings["line_width"])
    line_lengths = {
        **default_settings["line_lengths"],
        **settings.get("line_lengths", {}),
    }
    hlength = line_lengths["hlength"]
    vlength_factor = line_lengths["vlength_factor"]
    markersize = settings.get("markersize", 4)
    line_width_on_axis = {
        **default_settings["line_widths_on_axis"],
        **settings.get("line_widths_on_axis", {}),
    }
    fontsizes = {**default_settings["fontsizes"], **settings.get("fontsizes", {})}

    xfont = fontsizes["xaxis"]
    xtick_font = fontsizes["xtick_labels"]
    yfont = fontsizes["yaxis"]
    ytick_font = fontsizes["ytick_labels"]
    hwidth = line_width_on_axis["hwidth"]
    vwidth = line_width_on_axis["vwidth"]

    break_x = xaxis["break"]
    x_min = xaxis["min"]
    x_max = xaxis["max"]

    marker_styles = ["o", "s", "D", "^", "v", "P", "*", "X"]
    line_styles = ["-", "--", "-.", ":"]

    all_run_data = []
    for i, dn_models_dics in enumerate(models_dics_list):
        group_runs = prepare_diff_run_data(dn_models_dics, plot_params)
        marker = marker_styles[i % len(marker_styles)]
        linestyle = line_styles[i % len(line_styles)]
        for run in group_runs:
            run["markerstyle"] = marker
            run["linestyle"] = linestyle
        all_run_data.extend(group_runs)

    all_run_data.sort(key=lambda x: x["hticks"])

    sample_model = list(list(models_dics_list[0].values())[0].values())[0]
    sf = (
        sample_model.print_freq
        if len(sample_model.train_loss_list) > len(sample_model.epoch_times)
        else 1
    )

    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        sharey=True,
        figsize=(8, 5),
        gridspec_kw={"width_ratios": [1, 5], "wspace": 0.05},
    )

    for run in all_run_data:
        x = np.where(run["epochs"] * sf == 0, 1e-1, run["epochs"] * sf)
        y_mean = run["mean_vals"]
        y_min = run["min_vals"]
        y_max = run["max_vals"]
        color = run["color"]
        markerstyle = run.get("markerstyle", "-")
        linestyle = run.get("linestyle", "-")

        mask1 = x <= break_x
        mask2 = x > break_x

        label = run["label"] if linestyle == "-" else None

        ax1.plot(
            x[mask1],
            y_mean[mask1],
            color=color,
            label=label,
            lw=line_width,
            linestyle=linestyle,
            marker=markerstyle,
            markersize=markersize,
        )
        ax2.plot(
            x[mask2],
            y_mean[mask2],
            color=color,
            label=label,
            lw=line_width,
            linestyle=linestyle,
            marker=markerstyle,
            markersize=markersize,
        )

        if fill:
            ax1.fill_between(
                x[mask1], y_min[mask1], y_max[mask1], alpha=0.2, color=color
            )
            ax2.fill_between(
                x[mask2], y_min[mask2], y_max[mask2], alpha=0.2, color=color
            )

    for ax in (ax1, ax2):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_facecolor("white")

    ax1.yaxis.set_tick_params(labelsize=ytick_font)
    ax1.xaxis.set_tick_params(labelsize=xtick_font)
    ax2.xaxis.set_tick_params(labelsize=xtick_font)

    ax2.set_xlabel("epoch (log)", fontsize=xfont)
    ax1.set_xlim(left=x_min, right=break_x)
    ax2.set_xlim(left=break_x, right=x_max)
    ax1.set_ylabel(r"Diffusion MSE [mm$^4$ day$^{-2}$]", fontsize=yfont)

    ax1.spines["right"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax1.tick_params(labelright=False)
    ax2.tick_params(labelleft=False)

    d = 0.015
    kwargs = dict(transform=ax1.transAxes, color="k", clip_on=False)
    ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)
    ax1.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
    kwargs.update(transform=ax2.transAxes)
    ax2.plot((-d, +d), (-d, +d), **kwargs)
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)

    if legend["panel"] == 1:
        ax1.legend(
            fontsize=legend["fontsize"],
            loc=legend["loc"],
            ncols=legend["ncols"],
        )
    else:
        ax2.legend(
            fontsize=legend["fontsize"],
            loc=legend["loc"],
            ncols=legend["ncols"],
        )

    ax2_right = ax2.twinx()
    y_min_plot, y_max_plot = ax1.get_ylim()
    y_top = y_min_plot * vlength_factor
    ax2_right.set_yscale("log")
    ax2_right.set_ylim(ax2.get_ylim())
    ax2_right.tick_params(
        axis="y",
        which="both",
        left=False,
        right=False,
        labelleft=False,
        labelright=False,
    )

    for run in all_run_data:
        linestyle = run.get("linestyle", "-")
        color = run["color"]

        if linestyle == "-":
            best_epochs_lst = run["best_epochs_lst"]
            to_plot = np.min(best_epochs_lst)

            if to_plot < break_x:
                ax1.vlines(
                    to_plot,
                    y_min_plot,
                    y_top,
                    color=color,
                    linestyle=linestyle,
                    lw=vwidth,
                )
            ax2.vlines(
                to_plot,
                y_min_plot,
                y_top,
                color=color,
                linestyle=linestyle,
                lw=vwidth,
            )

            ax2.hlines(
                run["hticks"],
                x_max - hlength,
                x_max,
                color=color,
                linestyle=linestyle,
                linewidth=hwidth,
                zorder=10,
            )

        if linestyle in ("-", "--"):
            x_final = run["epochs"][-1] * sf
            if x_final <= break_x:
                ax1.vlines(
                    x_final,
                    y_min_plot,
                    y_max_plot,
                    color=color,
                    linestyle=linestyle,
                    lw=1,
                    alpha=0.75,
                )
            else:
                ax2.vlines(
                    x_final,
                    y_min_plot,
                    y_max_plot,
                    color=color,
                    linestyle=linestyle,
                    lw=1,
                    alpha=0.75,
                )

    ax2.set_ylim(y_min_plot)
    fig.tight_layout()

    if name:
        plt.savefig(name, dpi=100, bbox_inches="tight", facecolor="None")
        print("saved plot:", name)
    plt.show()

