import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Patch
import sys

from .prepare_model_loss import prepare_model_run_data

def plot_loss_and_time_dual_axis(
    models_dics_list,
    label_list=None,
    plot_params=None,
    name=None,
    figsize=(10, 5),
    legend_title="ES",
    y_label="Validation loss [a.u.]",
    y2_label="Total Run Time [s]",
    hatch_patterns=("", "////", "\\\\\\\\", "xxxx", "++++", "....", "ooo", "***"),
    colors=None,
    hatch_linewidth=1,
    x_label=r"$N_u$",
    x_label_fontsize=14,
    legend_size=16,
    legend_bool = False,
    # NEW PARAMETERS:
    intra_spacing=0.15,   # spacing between loss & time bars inside a group
    inter_spacing=0.80,    # spacing between groups (labels)
    alpha = 0.3
):
    """
    Combined dual-axis bar plot where:

        - Left y-axis  = Loss (bars)
        - Right y-axis = Time (bars)

    You can adjust:
        intra_spacing : distance between the LOSS and TIME bars within each group
        inter_spacing : spacing between different x-axis groups

    Example:
        |Loss| |Time|     (big gap)     |Loss| |Time|
    """

    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from matplotlib.patches import Patch

    # ---------------- Colors per x-label ----------------
    if colors is None:
        colors = ["#0033A0", "#1E90FF", "#6699CC", "#A4C8E1", "#D6EAF8"]

    if label_list is None:
        label_list = [f"Group {i+1}" for i in range(len(models_dics_list))]

    # ---------------- Gather Data ----------------
    grouped_loss = []
    grouped_time = []
    all_labels = []

    for models_dics in models_dics_list:
        run_data, raw_data = prepare_model_run_data(models_dics, plot_params=plot_params)

        loss_map = {d["label"]: d for d in run_data}
        time_map = {d["label"]: d for d in raw_data}

        grouped_loss.append(loss_map)
        grouped_time.append(time_map)

        for lbl in loss_map.keys():
            if lbl not in all_labels:
                all_labels.append(lbl)

    num_groups = len(models_dics_list)
    num_labels = len(all_labels)

    label_color_map = {lbl: colors[i % len(colors)] for i, lbl in enumerate(all_labels)}

    # ---------------- Group Geometry ----------------
    # The base x-spacing between LABEL GROUPS:
    x_positions = np.arange(num_labels) * inter_spacing

    # Within each label group, each experimental GROUP gets a slot:
    slot_width = 0.35   # width allocated to one experimental group within a label
    bar_width = slot_width * 0.40

    # Loss/Time inside each slot: offset left/right by intra-spacing
    loss_offset  = -intra_spacing / 2
    time_offset  = +intra_spacing / 2

    # ---------------- Plot Setup ----------------
    fig, ax1 = plt.subplots(figsize=figsize)
    ax2 = ax1.twinx()

    old_hlw = mpl.rcParams["hatch.linewidth"]
    old_hc = mpl.rcParams["hatch.color"]
    mpl.rcParams["hatch.linewidth"] = hatch_linewidth
    mpl.rcParams["hatch.color"] = "gray"

    try:
        # ---------------- Bars ----------------
        for li, label in enumerate(all_labels):
            base_x = x_positions[li]
            hatch = hatch_patterns[li % len(hatch_patterns)]

            for gi in range(num_groups):
                if label not in grouped_loss[gi] or label not in grouped_time[gi]:
                    continue

                loss_d = grouped_loss[gi][label]
                time_d = grouped_time[gi][label]

                # --- Loss stats ---
                m_loss = loss_d["mean_best_model_loss"]
                min_l = loss_d.get("min_best_model_loss", m_loss - loss_d["std_best_model_loss"])
                max_l = loss_d.get("max_best_model_loss", m_loss + loss_d["std_best_model_loss"])
                yerr_loss = [[m_loss - min_l], [max_l - m_loss]]

                # --- Time stats ---
                m_time = time_d["run_time_mean"]
                min_t = time_d.get("run_time_min", m_time - time_d["run_time_std"])
                max_t = time_d.get("run_time_max", m_time + time_d["run_time_std"])
                yerr_time = [[m_time - min_t], [max_t - m_time]]

                #hatch = hatch_patterns[gi % len(hatch_patterns)]
                color = label_color_map[label]

                # slot for group gi
                slot_x = base_x + gi * slot_width

                # --- LOSS BAR --- (left y-axis)
                ax1.bar(
                    slot_x + loss_offset,
                    m_loss,
                    width=bar_width,
                    yerr=np.array(yerr_loss),
                    capsize=5,
                    facecolor=color,
                    alpha=0.75,
                    edgecolor="black",
                    hatch=hatch,
                    linewidth=1.5,
                )

                # --- TIME BAR --- (right y-axis)
                ax2.bar(
                    slot_x + time_offset,
                    m_time,
                    width=bar_width,
                    yerr=np.array(yerr_time),
                    capsize=5,
                    facecolor=color,
                    alpha=alpha,
                    edgecolor="black",
                    hatch=hatch,
                    linewidth=1.5,
                )

        # ---------------- Legend ----------------
        handles = [
            Patch(facecolor="white", edgecolor="black",
                  hatch=hatch_patterns[i % len(hatch_patterns)],
                  label=label_list[i], linewidth=1.5)
            for i in range(num_groups)
        ]
        if legend_bool:
            ax1.legend(handles, label_list, title=legend_title, fontsize=legend_size)

        # ---------------- Axes ----------------
        ax1.set_ylabel(y_label, fontsize=14)
        ax2.set_ylabel(y2_label, fontsize=14, color="gray")

        ax1.set_yscale("log")
        ax2.set_yscale("log")

        ax1.set_xticks(x_positions)
        ax1.set_xticklabels(all_labels, fontsize=14)

        ax1.set_xlabel(x_label, fontsize=x_label_fontsize)

        fig.tight_layout()

        if name:
            plt.savefig(name, dpi=120, bbox_inches="tight")
            print("Saved plot:", name)

        plt.show()

    finally:
        mpl.rcParams["hatch.linewidth"] = old_hlw
        mpl.rcParams["hatch.color"] = old_hc
