import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Patch
import sys

from .run_data_preparation import prepare_model_run_data

def plot_running_min_val_loss_broken_x_log_lst(
    models_dics_list, plot_params=None, plot_settings=None
):
    """
    Plot running-min validation loss for multiple experiment groups.

    Each entry in `models_dics_list` is treated as a separate group and
    assigned a distinct marker and linestyle, while colors come from
    `plot_params` / configuration keys. All runs are combined and sorted by
    final performance.

    Returns
    -------
    dict
        Summary of used settings and number of runs.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.lines import Line2D

    default_settings = {
        "xaxis": {"min": 1, "max": 1e5, "break": 1e3},
        "legend": {
            "panel": 1,          # 1 → ax1, 2 → ax2
            "loc": (0.05, 0.95),
            "loc_upd": (0.35, 0.95),
            "fontsize": 10,
            "ncols": 1,
            "title": "$N_u$",      # legend title for lines
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
    hwidth = line_width_on_axis["hwidth"]
    vwidth = line_width_on_axis["vwidth"]
    vlength_factor = line_lengths["vlength_factor"]

    break_x = xaxis["break"]
    x_min = xaxis["min"]
    x_max = xaxis["max"]

    marker_styles = ["o", "s", "D", "^", "v", "P", "*", "X"]
    line_styles = ["-", "--"]

    run_data_all = []
    label_to_color = {}  # for clean legend construction

    # ---------- Prepare and annotate runs ----------
    for i, models_dics in enumerate(models_dics_list):
        marker = marker_styles[i % len(marker_styles)]
        linestyle = line_styles[i % len(line_styles)]

        run_data, _ = prepare_model_run_data(models_dics, plot_params=plot_params)

        for run in run_data:
            run["markerstyle"] = marker
            run["linestyle"] = linestyle

            # keep a single color per label for legend
            label = run["label"]
            if label not in label_to_color:
                label_to_color[label] = run["color"]

        run_data_all.extend(run_data)

    # Sort by hticks (performance)
    run_data_all#.sort(key=lambda x: x["hticks"])

    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        sharey=True,
        figsize=figsize,
        gridspec_kw={"width_ratios": [1, 5], "wspace": 0.05},
    )

    # ---------- Plot lines and best-epoch markers ----------
    for run in run_data_all:
        x = np.where(run["epochs"] == 0, 1e-1, run["epochs"])
        y_min = run["running_min_min"]
        color = run["color"]
        linestyle = run.get("linestyle", "-")
        markerstyle = run.get("markerstyle", "o")

        mask1 = x <= break_x
        mask2 = x > break_x

        # No labels here → we build legend manually, avoiding duplicates
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

        # Best-model marker (black outline)
        best_idx = run["best_epoch_idx"]
        ax2.scatter(
            x[best_idx],
            y_min[best_idx],
            facecolors=color,
            edgecolors="black",
            linewidths=0.8,
            s=40,
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

    # ---------- Axes formatting ----------
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

    # ---------- Broken-axis diagonals ----------
    d = 0.015
    kwargs = dict(transform=ax1.transAxes, color="k", clip_on=False)
    ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)
    ax1.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
    kwargs.update(transform=ax2.transAxes)
    ax2.plot((-d, +d), (-d, +d), **kwargs)
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)

    # ---------- Legend for lines (unique labels) ----------
    legend_title = legend.get("title", "N_u")

    # Order legend entries in a stable way (sorted by label)
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
    # Marker shapes for ES values; legend only, white fill, black outline
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

    # Put ES legend on the right panel (separate from line legend)
    marker_legend = ax2.legend(
        handles=marker_handles,
        labels=marker_labels,
       # loc="upper right",
        bbox_to_anchor=legend["loc_upd"],
        #bbox_to_anchor=(0.98, 0.98),
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
        "num_runs": len(run_data_all),
    }


