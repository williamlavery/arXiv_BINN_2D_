import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Patch
import sys

from .prepare_model_loss import prepare_model_run_data

def plot_final_loss_lst(
    models_dics_list,
    label_list=None,
    plot_params=None,
    name=None,
    bbox_to_anchor=(0.5, 0.5),
    legend_size=16,
    y_label="Validation loss [a.u.]",
    hatch_patterns=(
        "", "////", "\\\\\\\\", "xxxx", "++++", "....", "ooo", "***"
    ),
    figsize=(10, 5),
    colors=None,
    hatch_linewidth=1,
    legend_title="ES",
    x_label=r"$N_u$",
    x_label_fontsize=14,
):
    """
    Bars use:
      - consistent fill colors across groups (same label → same color)
      - black hatch patterns per group (different texture per group)
      - log y-axis
      - spacing between bars of same label

    Error bars span min–max across repeats (using min_best_model_loss / max_best_model_loss
    if available, otherwise falling back to mean ± std_best_model_loss).
    """

    import matplotlib as mpl
    from matplotlib.patches import Patch

    # ---------- Blue palette for *labels* ----------
    if colors is None:
        colors = [
            "#0033A0",
            "#1E90FF",
            "#6699CC",
            "#A4C8E1",
            "#D6EAF8",
        ]

    # Label list is group names, not x-axis labels
    if label_list is None:
        label_list = [f"Group {i+1}" for i in range(len(models_dics_list))]

    # ---------- Gather Data ----------
    grouped_data = []
    all_labels = []

    for models_dics in models_dics_list:
        run_data, _ = prepare_model_run_data(models_dics, plot_params=plot_params)
        label_to_data = {d["label"]: d for d in run_data}
        grouped_data.append(label_to_data)

        for lbl in label_to_data.keys():
            if lbl not in all_labels:
                all_labels.append(lbl)

    print("All x-axis labels found (ordered):", all_labels)


    # x-axis labels sorted numerically
    num_labels = len(all_labels)
    num_groups = len(models_dics_list)

    # ---------- Assign consistent colors per label ----------
    label_color_map = {
        label: colors[i % len(colors)]
        for i, label in enumerate(all_labels)
    }

    # ---------- Bar Geometry ----------
    group_width = 0.8
    slot_width = group_width / num_groups
    bar_width = slot_width * 0.8
    group_offsets = (
        np.arange(num_groups) * slot_width - group_width / 2 + slot_width / 2
    )

    fig, ax1 = plt.subplots(figsize=figsize)

    # ---------- Hatch settings ----------
    old_hatch_lw = mpl.rcParams["hatch.linewidth"]
    old_hatch_color = mpl.rcParams["hatch.color"]

    mpl.rcParams["hatch.linewidth"] = hatch_linewidth
    mpl.rcParams["hatch.color"] = "gray"

    try:
        # ---------- Plot bars ----------
        for i, label in enumerate(all_labels):
            label_color = label_color_map[label]

            for j, group_dict in enumerate(grouped_data):
                if label not in group_dict:
                    continue

                d = group_dict[label]

                mean_val = d["mean_best_model_loss"]

                # Prefer explicit min/max if present; otherwise fall back to std
                if "min_best_model_loss" in d and "max_best_model_loss" in d:
                    min_val = d["min_best_model_loss"]
                    max_val = d["max_best_model_loss"]
                else:
                    min_val = mean_val - d["std_best_model_loss"]
                    max_val = mean_val + d["std_best_model_loss"]

                # Asymmetric error bars: [lower, upper] relative to mean
                lower_err = mean_val - min_val
                upper_err = max_val - mean_val
                yerr = np.array([[lower_err], [upper_err]])

                # hatch distinguishes the *group*
                hatch = hatch_patterns[j % len(hatch_patterns)]
                x_pos = i + group_offsets[j]

                ax1.bar(
                    x_pos,
                    mean_val,
                    yerr=yerr,
                    width=bar_width,
                    capsize=5,
                    facecolor=label_color,    # consistent across groups
                    alpha=0.7,
                    edgecolor="black",
                    hatch=hatch,              # group distinction
                    linewidth=1.5,
                    error_kw=dict(
                        ecolor="k",
                        linewidth=3,
                    ),
                )

        # ---------- Legend ----------
        # Shows only groups (hatch patterns)
        custom_handles = [
            Patch(
                facecolor="white",
                edgecolor="black",
                hatch=hatch_patterns[i % len(hatch_patterns)],
                linewidth=1.5,
                label=label_list[i],
            )
            for i in range(num_groups)
        ]

        ax1.legend(
            custom_handles,
            label_list,
            loc="upper left",
            bbox_to_anchor=bbox_to_anchor,
            fontsize=legend_size,
            title_fontsize=legend_size,
            title=legend_title,
        )

        # ---------- Axes ----------
        ax1.set_ylabel(y_label, fontsize=14)
        ax1.set_yscale("log")

        ax1.tick_params(axis="y", which="major", length=10)
        ax1.tick_params(axis="y", which="minor", length=5)
        ax1.tick_params(axis="y", labelsize=14)

        ax1.set_xticks(np.arange(num_labels))
        ax1.set_xticklabels(all_labels, rotation=0, ha="center", fontsize=14)

        ax1.set_xlabel(x_label, fontsize=x_label_fontsize)

        fig.tight_layout()

        if name:
            plt.savefig(name, dpi=100, bbox_inches="tight", facecolor="None")
            print("saved plot:", name)

        plt.show()

    finally:
        mpl.rcParams["hatch.linewidth"] = old_hatch_lw
        mpl.rcParams["hatch.color"] = old_hatch_color
