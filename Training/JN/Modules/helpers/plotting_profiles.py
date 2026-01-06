"""
plotting_profiles.py

Plotting utilities for physical profiles (u and D) and per-architecture repeats.

This module is part of a four-file utilities package:

- plotting_profiles.py
    * Plots operating in “physical space”, including:
      - `plot_initial_condition_1d`: 1D initial/terminal u(x, t) profiles.
      - `plot_eval2`: histogram of u plus model prediction vs. truth at t=0 and T.
      - `plot_repeats_u` / `plot_repeats_D`: per-architecture repeats for
        u-loss and diffusion error.
      - `plot_eval_D_multi`: ensemble diffusion profiles D(u) vs. truth.
      - `plot_repeats_times_u` / `_control`: per-architecture total training times.


This file focuses on plots that mainly depend on the spatial field u(x, t)
and diffusion D(u).
"""

import numpy as np
import torch, sys
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Patch
import matplotlib.ticker as ticker
import itertools

from .utils import to_torch, hist_properties


import numpy as np
import matplotlib.pyplot as plt

def plot_initial_condition_2d(dataobj, tidx=0, 
                              x1_idx=None, x2_idx=None,
                              filename=None, K_orig=1.7e3, tol=1e-6):
    """
    Plot 2D initial-condition profiles for u(x1, x2, t) as a heat map
    at a fixed time slice, and also plot fixed x1 and x2 1D slices.

    Parameters
    ----------
    dataobj : object
        Must contain:
        - u : noisy u(x1, x2, t), shape (Nx1, Nx2, Nt)
        - u_clean : clean reference array, same shape
        - x1 : 1D grid (len Nx1)
        - x2 : 1D grid (len Nx2)
        - t  : 1D time grid (len Nt)
    tidx : int
        Time index to visualize.
    x1_idx, x2_idx : int or None
        Indices for slicing. If None → midpoints are used.
    filename : str or None
        If provided, figure is saved.
    K_orig : float
        Scaling factor for u.

    Returns
    -------
    None
    """

    u = dataobj.u
    x1 = dataobj.x1
    x2 = dataobj.x2
    t = dataobj.t

    Nx1, Nx2, Nt = u.shape

    # Validate time index
    if not (0 <= tidx < Nt):
        raise ValueError(f"tidx={tidx} out of range for Nt={Nt}")

    # Default slice positions = midpoints
    if x1_idx is None:
        x1_idx = Nx1 // 2
    if x2_idx is None:
        x2_idx = Nx2 // 2

    # Extract slices
    u_slice_2d = u[:, :, tidx] * K_orig
    u_slice_x1 = u[x1_idx, :, tidx] * K_orig  # varies over x2
    u_slice_x2 = u[:, x2_idx, tidx] * K_orig  # varies over x1

    # --- CREATE FIGURE WITH 3 SUBPLOTS ---
    fig, axs = plt.subplots(1, 3, figsize=(14, 4))

    # -------------------------
    # 1. Heat map
    # -------------------------
    extent = [x1.min(), x1.max(), x2.min(), x2.max()]
    im = axs[0].imshow(
        u_slice_2d.T,
        origin="lower",
        extent=extent,
        aspect="equal",
        interpolation="nearest",
        cmap="viridis",
    )
    axs[0].set_title(f"Heat map at t = {tidx}/{Nt-1} T")
    axs[0].set_xlabel("x₁ [mm]")
    axs[0].set_ylabel("x₂ [mm]")
    fig.colorbar(im, ax=axs[0], label=r"cell density [cells mm$^{-2}$]")

    # -------------------------
    # 2. Slice at fixed x1 = x1[x1_idx]
    # -------------------------
    axs[1].plot(x2, u_slice_x1, color="black")
    axs[1].set_title(f"Slice at x₁ = {x1[x1_idx]:.3f}")
    axs[1].set_xlabel("x₂ [mm]")
    axs[1].set_ylabel(r"cell density [cells mm$^{-2}$]")
    axs[1].grid(True, alpha=0.3)

    # -------------------------
    # 3. Slice at fixed x2 = x2[x2_idx]
    # -------------------------
    axs[2].plot(x1, u_slice_x2, color="black")
    axs[2].set_title(f"Slice at x₂ = {x2[x2_idx]:.3f}")
    axs[2].set_xlabel("x₁ [mm]")
    axs[2].set_ylabel(r"cell density [cells mm$^{-2}$]")
    axs[2].grid(True, alpha=0.3)

    plt.tight_layout()

    # Optional saving
    if filename:
        plt.savefig(f"{filename}.png", dpi=120, bbox_inches="tight", facecolor="None")

    # ───── Error metrics (space–time) with near-zero masking ────────────────
    u_clean = dataobj.u_clean

    err      = np.abs(u - u_clean)
    mse_val  = np.mean(err**2)
    abs_val  = np.mean(err)

    print("MSE between u and u_clean:", K_orig * mse_val)
    print("ABS between u and u_clean [cells]:", K_orig * abs_val)

    # Relative error only on sufficiently large reference values
    u_ref_abs = np.abs(u_clean)
    u_max     = float(np.max(u_ref_abs)) if u_ref_abs.size else 0.0
    u_min     = max(tol, 1e-6 * max(1.0, u_max))   # same spirit as in DATA_add_noise
    mask      = u_ref_abs >= u_min

    if np.any(mask):
        rel_err_mean = np.mean(err[mask] / u_ref_abs[mask])
        print(
            "ABS (%) between u and u_clean "
            f"(on |u_clean| ≥ {u_min:.3e}):",
            100 * rel_err_mean,
        )
    else:
        print(
            "ABS (%) between u and u_clean: not defined "
            "(all |u_clean| below threshold)"
        )

    plt.show()


def plot_eval2(
    modelWrappers_dic,
    dataobj,
    IC=0,
    device="cpu",
    num_bins=10,
    label="",
    clean=True,
    noisy=True,
    error=True,
    K=1700,
    save_name=None,
    colors=None,
):
    """
    Plot histogram of u-values (train vs. validation) and prediction vs. truth.

    First, a stacked bar histogram of train and validation u values is shown.
    Then, for each model wrapper, the predicted u(x, t) at t=0 and t=T is
    compared against both clean and noisy data, with optional error curves.

    Parameters
    ----------
    modelWrappers_dic : dict
        Dictionary {seed: modelWrapper} (or similar) for a single IC.
    dataobj : object
        Data container with attributes x, u, u_clean, t, inputs.
    IC : int, optional
        Index of the initial condition (for labeling only), by default 0.
    device : str, optional
        Device used when running the model, by default "cpu".
    num_bins : int, optional
        Number of bins in the u-histogram, by default 10.
    label : str, optional
        Base label for this dataset, used in titles, by default ''.
    clean : bool, optional
        If True, plot clean curves, by default True.
    noisy : bool, optional
        If True, plot noisy curves, by default True.
    error : bool, optional
        If True, plot absolute error on a second axis, by default True.
    K : float, optional
        Scaling factor applied to u-values, by default 1700.
    save_name : str, optional
        Filename for saving the prediction plot (PNG). If None, no file is saved.
    colors : list, optional
        Custom color list. If None, a default purple palette is used.

    Returns
    -------
    None
    """
    if not colors:
        colors = ["#9013FE", "#5E239D", "#B580FF", "#E5D5FF"]
    markers = ["o", "s", "D", "^", "p", "<", ">", "p", "*", "h", "x", "+"]
    grayscale_colors = ["#111111", "#444444", "#888888", "#BBBBBB", "#EEEEEE"]

    # All models for same IC will have same data distribution -> use first model
    modelWrapper = list(modelWrappers_dic.values())[0]
    u_val = np.array([k for (_, _), k in zip(modelWrapper.x_val, modelWrapper.y_val)])
    u_train = np.array(
        [k for (_, _), k in zip(modelWrapper.x_train, modelWrapper.y_train)]
    )

    # Get data
    x = dataobj.x
    u = dataobj.u * K
    u_clean = dataobj.u_clean * K
    t = dataobj.t
    Nt, Nx = len(dataobj.t), len(dataobj.x)

    # Histogram properties
    model = modelWrapper.model
    model.eval()
    h_properties = hist_properties(dataobj, num_bins)

    hist = h_properties["hist"]
    bin_edges = h_properties["bin_edges"] * K
    bin_centers = h_properties["bin_centers"] * K

    # Compute separate histograms for train/val
    hist_train, _ = np.histogram(u_train * K, bins=bin_edges)
    hist_val, _ = np.histogram(u_val * K, bins=bin_edges)

    # Plot stacked bar chart
    fig, ax = plt.subplots(figsize=(5, 4))
    bar_width = bin_edges[1] - bin_edges[0]

    ax.bar(
        bin_centers,
        hist_val,
        width=bar_width,
        alpha=0.7,
        label="Validation",
        color=colors[1],
        edgecolor="black",
        linewidth=0.5,
    )
    ax.bar(
        bin_centers,
        hist_train,
        width=bar_width,
        alpha=0.7,
        bottom=hist_val,
        label="Train",
        color=colors[0],
        edgecolor="black",
        linewidth=0.5,
    )

    ax.set_xlabel("cell density [cells mm$^{-2}]$", fontsize=11)
    ax.set_ylabel("frequency", fontsize=11)
    ax.legend()
    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------------------
    # Prediction vs. truth at t=0 and t=T
    # ------------------------------------------------------------------
    for modelWrapper_indx in modelWrappers_dic.keys():
        modelWrapper = modelWrappers_dic[modelWrapper_indx]

        x_val = modelWrapper.x_val
        y_val = modelWrapper.y_val
        x_train = modelWrapper.x_train
        y_train = modelWrapper.y_train

        model = modelWrapper.model

        with torch.no_grad():
            u_pred_flat = model(to_torch(dataobj.inputs, device)) * K
            u_pred = u_pred_flat.reshape(Nx, Nt).cpu().numpy()

        fig, ax1 = plt.subplots(figsize=(5, 4))

        ax1.plot(
            x,
            u_pred[:, 0],
            "-",
            lw=2,
            alpha=0.7,
            color=colors[0],
            label=r"$\hat{u}_{dn}(0,x)$",
        )
        ax1.plot(
            x,
            u_pred[:, -1],
            "-",
            lw=2,
            alpha=0.7,
            color=colors[1],
            label=r"$\hat{u}_{dn}(T,x)$",
        )

        if error:
            ax2 = ax1.twinx()
            ax2.set_ylabel("abs. error [cells mm$^{-2}]$", fontsize=11)
            ax2.set_yscale("log")

        abs_e = np.abs(u_pred - u_clean)
        mse = np.mean(abs_e)

        # Clean reference
        if clean:
            ax1.plot(
                x,
                u_clean[:, 0],
                label="t=0",
                markersize=4,
                color=grayscale_colors[0],
                marker=markers[0],
                lw=0,
                markeredgecolor="black",
                markeredgewidth=0.5,
            )
            ax1.plot(
                x,
                u_clean[:, -1],
                label="t=T",
                markersize=4,
                color=grayscale_colors[4],
                marker=markers[4],
                lw=0,
                markeredgecolor="black",
                markeredgewidth=0.5,
            )

        if noisy:
            ax1.plot(x, u[:, 0], label="t=0", color=grayscale_colors[0], lw=1)
            ax1.plot(x, u[:, -1], label="t=T", color=grayscale_colors[3], lw=1)

        if error:
            ax2.plot(
                x,
                abs_e[:, 0],
                label="err(x,0)",
                markersize=4,
                color="#8b0000",
                marker=markers[0],
                markeredgecolor="black",
                markeredgewidth=0.5,
                alpha=0.7,
                lw=1,
            )
            ax2.plot(
                x,
                abs_e[:, -1],
                label="err(x,T)",
                markersize=4,
                color="#dd0505",
                marker=markers[4],
                markeredgecolor="black",
                markeredgewidth=0.5,
                alpha=0.7,
                lw=1,
            )

        ax1.set_xlabel("x [mm]", fontsize=11)
        ax1.set_ylabel("cell density [cells mm$^{-2}]$", fontsize=11)

        lines1, labels1 = ax1.get_legend_handles_labels()
        if error:
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(
                lines1 + lines2,
                labels1 + labels2,
                loc="lower right",
                fontsize=8,
                ncol=1,
            )
        else:
            ax1.legend(lines1, labels1, loc="lower right", fontsize=8, ncol=1)

        ax1.set_facecolor("white")
        plt.tight_layout()

        if save_name:
            plt.savefig(save_name, dpi=100, bbox_inches="tight", facecolor="None")


def plot_repeats_u(
    binn_models_dics,
    base_colors,
    filename=None,
    bbox_to_anchor=(1.02, 1.0),
    legend_fontsize=10,
):
    """
    Plot best validation loss for all repeats across widths and depths (u-models).

    Bars are grouped by width and hatched by depth; each repeat is overlaid
    on top of the same x-position with a tiny label (R1, R2, ...).

    Parameters
    ----------
    binn_models_dics : dict
        Nested dict {depth: {width: {seed: modelWrapper}}}.
    base_colors : list
        Base color for each width.
    filename : str, optional
        Output filename (PNG). If None, the figure is not saved.
    bbox_to_anchor : tuple, optional
        Legend anchor position, by default (1.02, 1.0).
    legend_fontsize : int, optional
        Fontsize for legend labels, by default 10.

    Returns
    -------
    None
    """
    hatch_styles = ["", "//", ".."]

    depths = list(binn_models_dics.keys())
    widths = list(binn_models_dics[depths[0]].keys())

    def collect_metric(binn_models, metric_attr):
        """
        Return a dict {(width, depth): [metric_per_repeat, ...]}.

        Parameters
        ----------
        binn_models : dict
            Nested dict {depth: {width: {seed: modelWrapper}}}.
        metric_attr : str
            Name of attribute to read from each modelWrapper.

        Returns
        -------
        dict
            Mapping (width, depth) to list of metric values over repeats.
        """
        out = {}
        for d in depths:
            for w in widths:
                repeats = binn_models[d][w]
                out[(w, d)] = [getattr(m, metric_attr) for m in repeats.values()]
        return out

    data = collect_metric(binn_models_dics, "best_val_loss")

    bar_width = 0.15
    intra_gap, inter_gap = 0.02, 0.30
    current_x = 0.0

    max_repeats = max(len(v) for v in data.values())
    label_cmap = cm.get_cmap("tab10", max_repeats)

    xtick_positions = []
    xtick_labels = []

    plt.figure(figsize=(7, 5))

    for (w, base_c) in zip(widths, base_colors):
        group_x_positions = []

        for i_d, d in enumerate(depths):
            vals = data[(w, d)]

            for i_r, v in enumerate(vals):
                alpha = 0.9 - 0.15 * i_r
                plt.bar(
                    current_x,
                    v,
                    width=bar_width,
                    color=base_c,
                    alpha=max(alpha, 0.25),
                    hatch=hatch_styles[i_d],
                    edgecolor="k",
                )

                plt.text(
                    current_x,
                    v,
                    f"R{i_r + 1}",
                    ha="center",
                    va="bottom",
                    fontsize=6,
                    color="black",
                    bbox=dict(
                        facecolor="white",
                        edgecolor="none",
                        pad=0.3,
                        alpha=0.9,
                    ),
                )

            group_x_positions.append(current_x)
            current_x += bar_width + (
                intra_gap if i_d < len(depths) - 1 else inter_gap
            )

        center_x = np.mean(group_x_positions)
        xtick_positions.append(center_x)
        xtick_labels.append(f"{w}")

    plt.xticks(xtick_positions, xtick_labels)
    plt.yscale("log")
    plt.ylabel(r"Validation loss [a.u.]")

    depth_patches = [
        Patch(
            facecolor="white",
            edgecolor="k",
            hatch=hatch_styles[i],
            label=rf"NN$_\mathrm{{D}}$ = {d}",
        )
        for i, d in enumerate(depths)
    ]
    plt.legend(
        handles=depth_patches,
        loc="upper left",
        bbox_to_anchor=bbox_to_anchor,
        fontsize=legend_fontsize,
    )

    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()

    if filename:
        plt.savefig(f"{filename}", dpi=100, bbox_inches="tight", facecolor="None")
    plt.show()


def plot_repeats_D(
    binn_models_dics,
    base_colors,
    filename=None,
    bbox_to_anchor=(1.02, 1.0),
    legend_fontsize=10,
):
    """
    Plot best diffusion error for all repeats across widths and depths (D-models).

    Bars are grouped by width and hatched by depth; each repeat is shown as a bar.

    Parameters
    ----------
    binn_models_dics : dict
        Nested dict {depth: {width: {seed: modelWrapper}}}.
    base_colors : list
        Base color for each width.
    filename : str, optional
        Output filename (PNG). If None, the figure is not saved.
    bbox_to_anchor : tuple, optional
        Legend position, by default (1.02, 1.0).
    legend_fontsize : int, optional
        Font size for legend text.

    Returns
    -------
    None
    """
    depths = list(binn_models_dics.keys())
    widths = list(binn_models_dics[depths[0]].keys())

    hatch_styles = ["", "//", ".."]

    def collect_metric(binn_models, metric_attr):
        """
        Return a dict {(width, depth): [metric_per_repeat, ...]}.

        See `plot_repeats_u` for structure.
        """
        out = {}
        for d in depths:
            for w in widths:
                repeats = binn_models[d][w]
                out[(w, d)] = [getattr(m, metric_attr) for m in repeats.values()]
        return out

    data = collect_metric(binn_models_dics, "best_diffusion_error")

    bar_width = 0.15
    intra_gap, inter_gap = 0.02, 0.30
    current_x = 0.0

    xtick_positions = []
    xtick_labels = []

    plt.figure(figsize=(7, 5))

    for (w, base_c) in zip(widths, base_colors):
        group_x_positions = []

        for i_d, d in enumerate(depths):
            vals = data[(w, d)]

            for i_r, v in enumerate(vals):
                alpha = 0.9 - 0.15 * i_r
                plt.bar(
                    current_x,
                    v,
                    width=bar_width,
                    color=base_c,
                    alpha=max(alpha, 0.25),
                    hatch=hatch_styles[i_d],
                    edgecolor="k",
                )

                plt.text(
                    current_x,
                    v,
                    f"R{i_r + 1}",
                    ha="center",
                    va="bottom",
                    fontsize=6,
                    color="black",
                    bbox=dict(
                        facecolor="white",
                        edgecolor="none",
                        pad=0.3,
                        alpha=0.9,
                    ),
                )

            group_x_positions.append(current_x)
            current_x += bar_width + (
                intra_gap if i_d < len(depths) - 1 else inter_gap
            )

        center_x = np.mean(group_x_positions)
        xtick_positions.append(center_x)
        xtick_labels.append(f"{w}")

    plt.xticks(xtick_positions, xtick_labels)
    plt.yscale("log")
    plt.ylabel(r"Diffusion MSE [day$^{-2}$ mm$^{4}$]")

    depth_patches = [
        Patch(
            facecolor="white",
            edgecolor="k",
            hatch=hatch_styles[i],
            label=rf"NN$_\mathrm{{D}}$ = {d}",
        )
        for i, d in enumerate(depths)
    ]

    plt.legend(
        handles=depth_patches,
        loc="upper left",
        bbox_to_anchor=bbox_to_anchor,
        fontsize=legend_fontsize,
    )

    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    if filename:
        plt.savefig(f"{filename}.png", dpi=100, bbox_inches="tight", facecolor="None")
    plt.show()


def plot_eval_D_multi(
    modelWrapper_dics,
    dataobj,
    D_sym_true_lst,
    colors,
    labels,
    Dnum=1,
    device="cpu",
    num_bins=50,
    K=1700,
    name=None,
    fill=True,
    legend_pos=(0.5, 0.5),
    linestyles=itertools.cycle(
        ["-", ":", "-.", (0, (3, 1, 1, 1)), (0, (1, 1))]
    ),
    errs=None,
):
    """
    Plot multiple learned diffusion profiles D(u) vs. their true counterparts.

    For each experiment group, the ensemble of predicted D(u) over repeats
    is averaged and plotted, with optional min/max shading. True diffusion
    curves and percentile lines of u are also shown.

    Parameters
    ----------
    modelWrapper_dics : dict
        Dictionary of {group_key: {seed: modelWrapper}}.
    dataobj : object
        Data container with attribute `u` for computing u percentiles.
    D_sym_true_lst : list
        List of symbolic or numerical true D(u) functions/arrays
        corresponding to each D_true in the first wrapper of each group.
    colors : list
        Colors to use for each group (truncated to len(labels)).
    labels : list
        Labels for each group (same length as `modelWrapper_dics`).
    Dnum : int, optional
        Index number for D in the label (D_1, D_2, ...), by default 1.
    device : str, optional
        Device for potential model evaluation (not heavily used here), by default "cpu".
    num_bins : int, optional
        Number of bins for u histogram for percentile marking, by default 50.
    K : float, optional
        Scaling factor applied to u-values, by default 1700.
    name : str, optional
        Output filename (PNG). If None, figure is not saved.
    fill : bool, optional
        If True, show min/max shading for ensemble, by default True.
    legend_pos : tuple, optional
        Legend anchor point in axes coordinates, by default (0.5, 0.5).
    linestyles : iterator, optional
        Cycle of linestyles for each group.
    errs : list, optional
        If provided as [err_up, err_low], plot ±percent bands around true D.

    Returns
    -------
    None
    """
    if errs:
        [err_up, err_low] = errs

    colors = colors[: len(labels)]
    h_properties = hist_properties(dataobj, num_bins)
    low_u = h_properties["low_count"]
    high_u = h_properties["high_count"]

    sample_model = list(list(modelWrapper_dics.values())[0].values())[0].model
    u_vals_torch = sample_model.u_vals_torch
    u_vals_np = sample_model.u_vals.flatten()

    results = []
    D_true_lst = []

    # Collect ensemble statistics per group
    for wrapper_dic, color, label in zip(
        modelWrapper_dics.values(), colors, labels
    ):
        diffusion_errors = []
        D_ensemble = []

        for i, wrapper in enumerate(wrapper_dic.values()):
            D_true_check = list(wrapper.model.D_true)
            if i == 0 and D_true_check not in D_true_lst:
                D_true_lst.append(D_true_check)

            model = wrapper.model
            model.eval()

            with torch.no_grad():
                diff_pred = (
                    model.D_scale * model.diffusion(model.u_vals_torch).flatten()
                )
                diffusion_error = (model.D_true_torch - diff_pred) ** 2
                diffusion_errors.append(diffusion_error.unsqueeze(0))
                D_ensemble.append(diff_pred.unsqueeze(0))

        D_ensemble = torch.cat(D_ensemble, dim=0)
        D_mean = torch.mean(D_ensemble, dim=0)
        D_min = torch.min(D_ensemble, dim=0).values
        D_max = torch.max(D_ensemble, dim=0).values

        diffusion_errors = torch.cat(diffusion_errors, dim=0)
        mse = torch.mean(diffusion_errors).item()

        results.append(
            (mse, label, color, D_mean.numpy(), D_min.numpy(), D_max.numpy())
        )

    fig, ax = plt.subplots(figsize=(7, 5))

    for j, (mse, label, color, D_mean, D_min, D_max) in enumerate(results):
        ls = next(linestyles)
        ax.plot(
            u_vals_np * K,
            D_mean,
            lw=2,
            color=color,
            linestyle=ls,
            label=label,
        )

        if fill:
            ax.fill_between(
                u_vals_np * K, D_min, D_max, alpha=0.4, color=color
            )

    # Plot true D(u) and optional error bands
    for i, (D_true, D_sym) in enumerate(zip(D_true_lst, D_sym_true_lst)):
        if i == 0:
            ax.plot(
                u_vals_np * K,
                D_true,
                "--",
                lw=2,
                color="k",
                label=f"$D_{Dnum}(u)$",
                zorder=3,
            )

            if errs is not None:
                ax.plot(
                    u_vals_np * K,
                    np.array(D_true) * (1 + err_up / 100),
                    "--",
                    lw=1,
                    color="r",
                    label=f"{err_up}%",
                    zorder=3,
                )
                ax.plot(
                    u_vals_np * K,
                    np.array(D_true) * (1 - err_low / 100),
                    "-.",
                    lw=1,
                    color="r",
                    label=f"{err_low}%",
                    zorder=3,
                )
        else:
            ax.plot(u_vals_np * K, D_true, "--", lw=2, color="k", zorder=3)

    ax.axvline(
        low_u * K, color="#666666", ls="-.", lw=1, label="5% $u$-perc."
    )
    ax.axvline(
        high_u * K, color="#BBBBBB", ls="--", lw=1, label="95% $u$-perc."
    )
    ax.set_xlabel("cell density [cells mm$^{-2}]$", fontsize=11)
    ax.set_ylabel(r"cell diffusion [mm$^2$ days$^{-1}$]", fontsize=11)
    ax.set_facecolor("white")
    ax.legend(
        loc="center",
        bbox_to_anchor=legend_pos,
        ncols=2,
        fontsize=12,
    )

    plt.tight_layout()
    if name:
        plt.savefig(name, dpi=100, bbox_inches="tight", facecolor="None")
        print("saved plot:", name)
    plt.show()


def plot_repeats_times_u(binn_models_dics, base_colors, filename=None):
    """
    Plot total training time for u-models across widths and depths.

    For each (width, depth) pair, the mean and standard deviation of total
    training time (after clipping epoch times at the 95th percentile) are
    shown as a bar with error bar on a log y-axis.

    Parameters
    ----------
    binn_models_dics : dict
        Nested dict {depth: {width: {seed: modelWrapper}}}.
    base_colors : list
        Colors for each width.
    filename : str, optional
        Output filename (PNG). If None, figure is not saved.

    Returns
    -------
    None
    """
    hatch_styles = ["", "//", ".."]

    depths = list(binn_models_dics.keys())
    widths = list(binn_models_dics[depths[0]].keys())

    def mean_95(epoch_times):
        arr = np.array(epoch_times, dtype=np.float64)
        p95 = np.percentile(arr, 95)
        return np.mean(arr[arr <= p95])

    def collect_metric(binn_models, metric_attr1, metric_attr2):
        out = {}
        for d in depths:
            for w in widths:
                repeats = binn_models[d][w]
                out[(w, d)] = [
                    mean_95(getattr(m, metric_attr1))
                    * len(getattr(m, metric_attr2))
                    for m in repeats.values()
                ]
        return out

    data = collect_metric(
        binn_models_dics,
        metric_attr1="epoch_times",
        metric_attr2="train_loss_list",
    )

    bar_width = 0.15
    intra_gap, inter_gap = 0.02, 0.10
    current_x = 0.0

    xtick_positions = []
    xtick_labels = []

    plt.figure(figsize=(7, 5))

    for (w, base_c) in zip(widths, base_colors):
        group_x_positions = []

        for i_d, d in enumerate(depths):
            vals = data[(w, d)]
            mean_val = np.mean(vals)
            std_val = np.std(vals)

            plt.bar(
                current_x,
                mean_val,
                yerr=std_val,
                width=bar_width,
                color=base_c,
                hatch=hatch_styles[i_d],
                edgecolor="k",
                error_kw=dict(ecolor="gray", lw=2),
                capsize=4,
            )

            group_x_positions.append(current_x)
            current_x += bar_width + (
                intra_gap if i_d < len(depths) - 1 else inter_gap
            )

        center_x = np.mean(group_x_positions)
        xtick_positions.append(center_x)
        xtick_labels.append(f"{w}")

    plt.xticks([], [])
    plt.ylabel(r"Total training time [s]", fontsize=14)

    depth_patches = [
        Patch(
            facecolor="white",
            edgecolor="k",
            hatch=hatch_styles[i],
            label=rf"NN$_\mathrm{{D}}$ = {d}",
        )
        for i, d in enumerate(depths)
    ]
    plt.legend(handles=depth_patches, loc="upper left", fontsize=16)

    plt.yscale("log")

    ax = plt.gca()
    ax.yaxis.set_minor_locator(
        ticker.LogLocator(base=10, subs="auto", numticks=100)
    )
    ax.yaxis.set_minor_formatter(ticker.FormatStrFormatter("%.0f"))
    ax.tick_params(axis="y", which="minor", labelsize=11)
    ax.tick_params(axis="y", which="major", labelsize=12)

    plt.tight_layout()

    if filename:
        plt.savefig(f"{filename}.png", dpi=100, bbox_inches="tight", facecolor="None")
    plt.show()


def plot_repeats_times_u_control(
    binn_models_dics_control, binn_models_dics, base_colors, filename=None
):
    """
    Plot total training time using control models for epoch-time measurements.

    Uses epoch times from `binn_models_dics_control` but epoch counts from
    `binn_models_dics`. Otherwise identical to `plot_repeats_times_u`.

    Parameters
    ----------
    binn_models_dics_control : dict
        Nested dict of control models {depth: {width: {seed: modelWrapper}}}.
    binn_models_dics : dict
        Nested dict of main models {depth: {width: {seed: modelWrapper}}}.
    base_colors : list
        Colors per width.
    filename : str, optional
        Output filename (PNG). If None, figure is not saved.

    Returns
    -------
    None
    """
    hatch_styles = ["", "//", ".."]

    depths = list(binn_models_dics.keys())
    widths = list(binn_models_dics[depths[0]].keys())

    def mean_95(epoch_times):
        arr = np.array(epoch_times, dtype=np.float64)
        p95 = np.percentile(arr, 95)
        return np.mean(arr[arr <= p95])

    def collect_metric(binn_models_control, binn_models, metric_attr1, metric_attr2):
        out = {}
        for d in depths:
            for w in widths:
                repeats_control = binn_models_control[d][w]
                repeats = binn_models[d][w]
                out[(w, d)] = [
                    mean_95(getattr(m_control, metric_attr1))
                    * len(getattr(m, metric_attr2))
                    for m_control, m in zip(
                        repeats_control.values(), repeats.values()
                    )
                ]
        return out

    data = collect_metric(
        binn_models_dics_control,
        binn_models_dics,
        metric_attr1="epoch_times",
        metric_attr2="train_loss_list",
    )

    bar_width = 0.15
    intra_gap, inter_gap = 0.02, 0.10
    current_x = 0.0

    xtick_positions = []
    xtick_labels = []

    plt.figure(figsize=(7, 5))

    for (w, base_c) in zip(widths, base_colors):
        group_x_positions = []

        for i_d, d in enumerate(depths):
            vals = data[(w, d)]
            mean_val = np.mean(vals)
            std_val = np.std(vals)

            plt.bar(
                current_x,
                mean_val,
                yerr=std_val,
                width=bar_width,
                color=base_c,
                hatch=hatch_styles[i_d],
                edgecolor="k",
                error_kw=dict(ecolor="gray", lw=2),
                capsize=4,
            )

            group_x_positions.append(current_x)
            current_x += bar_width + (
                intra_gap if i_d < len(depths) - 1 else inter_gap
            )

        center_x = np.mean(group_x_positions)
        xtick_positions.append(center_x)
        xtick_labels.append(f"{w}")

    plt.xticks([], [])
    plt.ylabel(r"Total training time [s]", fontsize=14)

    depth_patches = [
        Patch(
            facecolor="white",
            edgecolor="k",
            hatch=hatch_styles[i],
            label=rf"NN$_\mathrm{{D}}$ = {d}",
        )
        for i, d in enumerate(depths)
    ]
    plt.legend(handles=depth_patches, loc="upper left", fontsize=16)

    plt.yscale("log")

    ax = plt.gca()
    ax.yaxis.set_minor_locator(
        ticker.LogLocator(base=10, subs="auto", numticks=100)
    )
    ax.yaxis.set_minor_formatter(ticker.FormatStrFormatter("%.0f"))
    ax.tick_params(axis="y", which="minor", labelsize=11)
    ax.tick_params(axis="y", which="major", labelsize=12)

    plt.tight_layout()

    if filename:
        plt.savefig(f"{filename}.png", dpi=100, bbox_inches="tight", facecolor="None")
    plt.show()
