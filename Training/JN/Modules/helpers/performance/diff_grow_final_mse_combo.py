
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Patch
import sys

from .diffusion.prepare_diff_mse import prepare_diff_run_data
from .growth.prepare_grow_mse import prepare_grow_run_data

def plot_dual_mse_lst(
    grow_models_dics_list,
    diff_models_dics_list,
    label_list=None,
    plot_params=None,
    name=None,
    bbox_to_anchor=(0.5, 0.5),
    legend_size=16,
    # now: left = diffusion, right = growth
    y_label_left=r"Diffusion MSE [mm$^4$ day$^{-2}$]",
    y_label_right=r"Growth MSE [mm$^4$ day$^{-2}$]",
    hatch_patterns=(
        "", "////", "\\\\\\\\", "xxxx", "++++", "....", "ooo", "***"
    ),
    figsize=(10, 5),
    diffusion_colors=None,
    growth_colors=None,
    hatch_linewidth=1,
    legend_title="ES",
    legend_bool=True,
    x_label=r"$N_u$",
    x_label_fontsize=14,
    # --- PE ARGS (applied to growth axis, now on ax2) ---
    pe_values=None,
    pe_label_list=None,
    pe_legend_title="MPE",
    pe_legend_ncols=1,
    pe_bbox_to_anchor_offset=(0.0, -0.15),
    pe_linestyles=("--", "-.", ":"),
    # --- Spacing params ---
    group_width=0.8,          # total width for all ES groups at one x-label
    intra_metric_gap_factor=0.3,  # relative gap between growth & diffusion in a group

    diffusion_ref = [None, None],
    growth_ref = [None, None]
):
    """
    Dual-axis bar plot with inter- and intra-group spacing:

      - For each x-label (e.g. ES/Nu), we have a total 'group_width' (<= 1).
      - That is divided among ES groups (entries in *_models_dics_list).
      - For each ES group, we plot two bars:
          * Diffusion  (left)  on ax1
          * Growth     (right) on ax2
        with an intra-group gap between them.

    Both y-axes are log scale.

    Requires:
      - prepare_grow_run_data, prepare_diff_run_data
      - Each returns list of dicts with keys:
          'label', 'avg_mse', and either
          ('min_mse','max_mse') or 'std_mse'.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib as mpl
    from matplotlib.patches import Patch

    # ---------- Palettes ----------
    if growth_colors is None:
        growth_colors = [
            "#00441B",
            "#1A9850",
            "#66C2A4",
            "#B2E2E2",
            "#E5F5F9",
        ][::-1]

    if diffusion_colors is None:
        diffusion_colors = [
            "#7F2704",
            "#D95F0E",
            "#FE9929",
            "#FEC44F",
            "#FEE391",
        ][::-1]

    # ---------- Label list / ES groups ----------
    if label_list is None:
        n_groups = max(len(grow_models_dics_list), len(diff_models_dics_list))
        label_list = [f"Group {i+1}" for i in range(n_groups)]

    # ---------- Gather Data ----------
    # Growth
    grow_grouped = []
    grow_labels = []
    for models_dics in grow_models_dics_list:
        mse_data = prepare_grow_run_data(models_dics, plot_params=plot_params["diff"])
        label_to_data = {d["label"]: d for d in mse_data}
        grow_grouped.append(label_to_data)
        for lbl in label_to_data.keys():
            if lbl not in grow_labels:
                grow_labels.append(lbl)

    # Diffusion
    diff_grouped = []
    diff_labels = []
    for models_dics in diff_models_dics_list:
        mse_data = prepare_diff_run_data(models_dics, plot_params=plot_params["grow"])
        label_to_data = {d["label"]: d for d in mse_data}
        diff_grouped.append(label_to_data)
        for lbl in label_to_data.keys():
            if lbl not in diff_labels:
                diff_labels.append(lbl)

    # Union of labels (preserve order)
    all_labels = grow_labels[:]
    for lbl in diff_labels:
        if lbl not in all_labels:
            all_labels.append(lbl)

    print("All x-axis labels found (ordered):", all_labels)

    num_labels = len(all_labels)
    num_grow_groups = len(grow_models_dics_list)
    num_diff_groups = len(diff_models_dics_list)
    num_groups = max(num_grow_groups, num_diff_groups)

    # ---------- Color maps (per x-label) ----------
    grow_label_color_map = {
        label: growth_colors[i % len(growth_colors)]
        for i, label in enumerate(all_labels)
    }
    diff_label_color_map = {
        label: diffusion_colors[i % len(diffusion_colors)]
        for i, label in enumerate(all_labels)
    }

    # ---------- Bar Geometry with inter + intra spacing ----------
    group_block_width = group_width / num_groups

    metric_bar_width = group_block_width * 0.35
    metric_gap = group_block_width * intra_metric_gap_factor
    total_inner = 2 * metric_bar_width + metric_gap
    if total_inner > group_block_width:
        scale = group_block_width / total_inner
        metric_bar_width *= scale
        metric_gap *= scale

    group_offsets = (
        (np.arange(num_groups) - (num_groups - 1) / 2.0) * group_block_width
    )

    half_sep = (metric_bar_width + metric_gap) / 2.0

    fig, ax1 = plt.subplots(figsize=figsize)   # left: diffusion
    ax2 = ax1.twinx()                          # right: growth

    # ---------- Hatch settings ----------
    old_hatch_lw = mpl.rcParams["hatch.linewidth"]
    old_hatch_color = mpl.rcParams["hatch.color"]
    mpl.rcParams["hatch.linewidth"] = hatch_linewidth
    mpl.rcParams["hatch.color"] = "gray"

    all_mse_values = []

    try:
        # ---------- DIFFUSION bars (ax1, left within each ES block) ----------
        for i, label in enumerate(all_labels):
            hatch = hatch_patterns[i % len(hatch_patterns)]
            label_color = diff_label_color_map[label]

            for j, group_dict in enumerate(diff_grouped):
                if j >= num_groups:
                    continue
                if label not in group_dict:
                    continue

                d = group_dict[label]

                x_center_group = i + group_offsets[j]
                x_pos_diff = x_center_group - half_sep  # left bar

                center = d["avg_mse"]
                if "min_mse" in d and "max_mse" in d:
                    min_val = d["min_mse"]
                    max_val = d["max_mse"]
                else:
                    min_val = center - d["std_mse"]
                    max_val = center + d["std_mse"]

                lower_err = center - min_val
                upper_err = max_val - center
                yerr = np.array([[lower_err], [upper_err]])

                all_mse_values.extend([min_val, max_val])

                ax1.bar(
                    x_pos_diff,
                    center,
                    yerr=yerr,
                    width=metric_bar_width,
                    capsize=5,
                    facecolor=label_color,
                    alpha=0.5,
                    edgecolor="black",
                    hatch=hatch,
                    linewidth=1.5,
                    error_kw=dict(
                        ecolor="k",
                        linewidth=3,
                    ),
                    zorder=3,
                )

        # ---------- GROWTH bars (ax2, right within each ES block) ----------
        for i, label in enumerate(all_labels):
            label_color = grow_label_color_map[label]
            hatch = hatch_patterns[i % len(hatch_patterns)]

            for j, group_dict in enumerate(grow_grouped):
                if j >= num_groups:
                    continue
                if label not in group_dict:
                    continue

                d = group_dict[label]

                x_center_group = i + group_offsets[j]
                x_pos_grow = x_center_group + half_sep  # right bar

                center = d["avg_mse"]
                if "min_mse" in d and "max_mse" in d:
                    min_val = d["min_mse"]
                    max_val = d["max_mse"]
                else:
                    min_val = center - d["std_mse"]
                    max_val = center + d["std_mse"]

                lower_err = center - min_val
                upper_err = max_val - center
                yerr = np.array([[lower_err], [upper_err]])

                all_mse_values.extend([min_val, max_val])

                ax2.bar(
                    x_pos_grow,
                    center,
                    yerr=yerr,
                    width=metric_bar_width,
                    capsize=5,
                    facecolor=label_color,
                    alpha=0.5,
                    edgecolor="black",
                    hatch=hatch,
                    linewidth=1.5,
                    error_kw=dict(
                        ecolor="k",
                        linewidth=3,
                    ),
                    zorder=2,
                )

        # ---------- ES Legend (hatch patterns) ----------
        es_handles = [
            Patch(
                facecolor="white",
                edgecolor="black",
                hatch=hatch_patterns[i % len(hatch_patterns)],
                linewidth=1.5,
                label=label_list[i],
            )
            for i in range(num_groups)
        ]
        if legend_bool:
            es_legend = ax1.legend(
                es_handles,
                label_list,
                bbox_to_anchor=bbox_to_anchor,
                fontsize=legend_size,
                title_fontsize=legend_size,
                title=legend_title,
            )
            ax1.add_artist(es_legend)

        # ---------- Horizontal PE lines on GROWTH axis (now ax2) ----------
        pe_handles = []
        if pe_values is not None:
            import numpy as np
            if not isinstance(pe_values, (list, tuple, np.ndarray)):
                pe_values = [pe_values]

            if pe_label_list is None:
                pe_label_list = [f"PE {i+1}" for i in range(len(pe_values))]

            for y_val, lbl, pe_ls in zip(pe_values, pe_label_list, pe_linestyles):
                line = ax2.axhline(
                    y=y_val,
                    color="red",
                    linestyle=pe_ls,
                    linewidth=1.5,
                    label=lbl,
                )
                pe_handles.append(line)

            pe_bbox = (
                bbox_to_anchor[0] + pe_bbox_to_anchor_offset[0],
                bbox_to_anchor[1] + pe_bbox_to_anchor_offset[1],
            )
            if legend_bool:
                ax2.legend(
                    pe_handles,
                    pe_label_list,
                    bbox_to_anchor=pe_bbox,
                    fontsize=legend_size,
                    title_fontsize=legend_size,
                    title=pe_legend_title,
                    ncols=pe_legend_ncols,
                )

        # ---------- Axes formatting ----------
        ax1.set_ylabel(y_label_left, fontsize=14)   # diffusion
        ax2.set_ylabel(y_label_right, fontsize=14)  # growth

        ax1.set_yscale("log")
        ax2.set_yscale("log")

        ax1.tick_params(axis="y", which="major", length=10)
        ax1.tick_params(axis="y", which="minor", length=5)
        ax1.tick_params(axis="y", labelsize=14)
        ax2.tick_params(axis="y", which="major", length=10)
        ax2.tick_params(axis="y", which="minor", length=5)
        ax2.tick_params(axis="y", labelsize=14)

        # x ticks at integer label centers
        ax1.set_xticks(np.arange(num_labels))
        ax1.set_xticklabels(all_labels, rotation=0, ha="center", fontsize=14)
        ax1.set_xlabel(x_label, fontsize=x_label_fontsize)

        # ---------- Reference lines and combined legend ----------
        handles = []
        labels = []

        # diffusion on ax1 (left)
        if diffusion_ref != [None, None]:
            h1 = ax1.axhline(
                diffusion_ref[0],
                color=diffusion_ref[1],
                label=r"$D(\cdot)$",
                zorder=4,
                linestyle =":"
            )
            handles.append(h1)
            labels.append(r"$D(\cdot)$")

        # growth on ax2 (right)
        if growth_ref != [None, None]:
            h2 = ax2.axhline(
                growth_ref[0],
                color=growth_ref[1],
                label=r"$G(\cdot)$",
                zorder=4,
                linestyle ="--"
            )
            handles.append(h2)
            labels.append(r"$G(\cdot)$")

        # Create one legend containing both, with white background on top
        if handles:
            lg = ax1.legend(
                handles,
                labels,
                title=r"best $1$D$+t$",
                ncols=2,
                loc="best",
                frameon=True,
                facecolor="white",
                framealpha=1.0,
                fontsize = 12
            )
            lg.set_zorder(10)

        fig.tight_layout()

        if name:
            plt.savefig(name, dpi=100, bbox_inches="tight", facecolor="None")
            print("saved plot:", name)

        plt.show()

    finally:
        mpl.rcParams["hatch.linewidth"] = old_hatch_lw
        mpl.rcParams["hatch.color"] = old_hatch_color
