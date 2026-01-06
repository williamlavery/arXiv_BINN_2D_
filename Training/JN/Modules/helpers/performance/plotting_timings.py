import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Patch
import sys
from .density.prepare_model_loss import prepare_model_run_data

def prepare_timing_data_control(models_dics_c, models_dics, plot_params=None):
    """
    Aggregate timing statistics when training-time comes from a control model.

    Parameters
    ----------
    models_dics_c : dict
        Nested dict of control model wrappers (for timing).
    models_dics : dict
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
    model_keys = list(models_dics.keys())
    dic_values_c = list(models_dics_c.values())
    dic_values = list(models_dics.values())

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

def plot_total_run_times_lst(
    models_dics_list,
    label_list=None,
    plot_params=None,
    name=None,
    hatch_patterns=(
        "", "////", "\\\\\\\\", "xxxx", "++++", "....", "ooo", "***"
    ),
    figsize=(7, 5),
    colors=None,
    hatch_linewidth=1,
    legend_pos=(0.02, 0.98),
    legend_fontsize=16,
    legend_title="ES",
    legend_ncols=1,
    legend_bool = True,
    y_label="Total Run Time [s]",
    x_label=r"$N_u$",
    x_label_fontsize=14,
):
    """
    Compare TOTAL training times across multiple experiment groups.

    Styling matches the new plot_final_loss_lst:
      - consistent fill colors across labels (same x-label → same color)
      - black hatch patterns per group (different texture per group)
      - log y-axis

    Error bars span min–max across repeats (using run_time_min / run_time_max if
    available, otherwise falling back to run_time_mean ± run_time_std).
    """

    import matplotlib as mpl
    from matplotlib.patches import Patch

    # ---------- Blue-ish palette for *labels* ----------
    if colors is None:
        colors = [
            "#0033A0",
            "#1E90FF",
            "#6699CC",
            "#A4C8E1",
            "#D6EAF8",
        ]

    # Legend group labels
    if label_list is None:
        label_list = [f"Group {i+1}" for i in range(len(models_dics_list))]

    # ---------- Gather data ----------

    grouped_data = []
    all_labels = []   # ordered list, no duplicates

    for models_dics in models_dics_list:
        _, raw_data = prepare_model_run_data(models_dics, plot_params=plot_params)
        label_to_data = {d["label"]: d for d in raw_data}
        grouped_data.append(label_to_data)

        # preserve the order of first appearance
        for lbl in label_to_data.keys():
            if lbl not in all_labels:
                all_labels.append(lbl)

    print("All x-axis labels found (ordered):", all_labels)


    num_labels = len(all_labels)
    num_groups = len(models_dics_list)

    # ---------- Assign consistent colors per label ----------
    label_color_map = {
        label: colors[i % len(colors)]
        for i, label in enumerate(all_labels)
    }

    # ---------- Bar geometry ----------
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
        # ---------- Plot bars (total run time) ----------
        for i, label in enumerate(all_labels):
            label_color = label_color_map[label]

            for j, group_dict in enumerate(grouped_data):
                if label not in group_dict:
                    continue

                d = group_dict[label]
                x_pos = i + group_offsets[j]
                hatch = hatch_patterns[j % len(hatch_patterns)]

                center = d["run_time_mean"]

                # Prefer explicit min/max if present; otherwise fall back to mean ± std
                if "run_time_min" in d and "run_time_max" in d:
                    min_val = d["run_time_min"]
                    max_val = d["run_time_max"]
                else:
                    min_val = center - d["run_time_std"]
                    max_val = center + d["run_time_std"]

                lower_err = center - min_val
                upper_err = max_val - center
                yerr = np.array([[lower_err], [upper_err]])

                ax1.bar(
                    x_pos,
                    center,
                    yerr=yerr,
                    width=bar_width,
                    capsize=5,
                    facecolor=label_color,       # consistent per x-label
                    alpha=0.7,
                    edgecolor="black",
                    hatch=hatch,                  # distinguishes group
                    linewidth=1.5,
                    error_kw=dict(
                        ecolor="k",
                        linewidth=3,
                    ),
                )

        # ---------- Legend (groups only) ----------
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
        if legend_bool:
            ax1.legend(
                custom_handles,
                label_list,
                bbox_to_anchor=legend_pos,
                fontsize=legend_fontsize,
                ncols=legend_ncols,
                title_fontsize=legend_fontsize,
                title=legend_title,
            )

        # ---------- Axes ----------
        ax1.set_ylabel(y_label, fontsize=14)
        ax1.set_yscale("log")

        ax1.tick_params(axis="y", which="major", length=10, labelsize=14)
        ax1.tick_params(axis="y", which="minor", length=5)

        ax1.set_xticks(np.arange(num_labels))
        ax1.set_xticklabels(all_labels, rotation=0, ha="center", fontsize=14)

        ax1.set_xlabel(x_label, fontsize=x_label_fontsize)

        fig.tight_layout()

        if name:
            plt.savefig(name, dpi=100, bbox_inches="tight", facecolor="None")
            print("saved plot:", name)

        plt.show()

    finally:
        # restore rcParams
        mpl.rcParams["hatch.linewidth"] = old_hatch_lw
        mpl.rcParams["hatch.color"] = old_hatch_color
