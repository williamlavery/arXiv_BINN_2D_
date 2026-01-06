


import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patches as mpatches
import matplotlib.cm as cm


def diffusion_func(u):
    """Constant diffusion function D(u)."""
    return 0.02472 * np.ones_like(u)

def growth_func(u):
    """Zero growth function G(u)."""
    return 0.0 * u

def ic1(x):
    """Smooth cosine bump initial condition: 0 at edges, 1 in middle."""
    mean_x = np.mean(x)
    L = np.max(x) - np.min(x)
    return 0.5 - 0.5 * np.cos(2 * np.pi * (x - mean_x) / L)

def ic2(x):
    """Smooth cosine bump initial condition: 0 at edges, 1 in middle."""
    mean_x = np.mean(x)
    L = np.max(x) - np.min(x)
    return 0.5 + 0.5 * np.cos(2 * np.pi * (x - mean_x) / L)

def RDEq(x, t, u0, clear, PDE_sim_old, PDE_RHS_1D):
    """
    Thin wrapper around your 1-D PDE solver.
    PDE_sim_old and PDE_RHS_1D are injected so there is no circular import.
    """
    return PDE_sim_old(
        PDE_RHS_1D, u0, x, t,
        diffusion_func, growth_func, clear=clear
    )

def TVplotter(data_obj,
              PDE_sim_old,
              PDE_RHS_1D, 
              add_noise, 
              ic_num=1, 
              seed=42,
              bins = 7,
              tnum=5,
              xnum=10,
              noise_lvl = 0,
              save_path_intro = "pngs/",
              dpi=100):
    # pick which IC
    ic_func = ic1 if ic_num == 1 else ic2

    # ------------------------------------------------------------------
    # Problem setup
    # ------------------------------------------------------------------
    x = np.linspace(np.min(data_obj.x), np.max(data_obj.x), xnum)
    t = np.linspace(np.min(data_obj.t), np.max(data_obj.t), tnum)

    K = data_obj.K
    gamma = data_obj.gamma

    # PDE solve
    u_clean = RDEq(x=x, t=t, u0=K * ic_func(x), clear=False,
                   PDE_sim_old=PDE_sim_old, PDE_RHS_1D=PDE_RHS_1D)

    cell_density = add_noise(u_clean, gamma=gamma, noise_lvl=noise_lvl,
                             seed=seed, t_start_idx=0)

    # ensure same t-grid used for plotting
    t = np.linspace(0, 4, 5)

    # --------------------------------------
    # Pixel grid edges
    # --------------------------------------
    dx = x[1] - x[0]
    dt = t[1] - t[0]
    x_edges = np.concatenate(([x[0] - dx/2], x + dx/2))
    t_edges = np.concatenate(([t[0] - dt/2], t + dt/2))

    num_x, num_t = len(x), len(t)
    total_cells  = num_x * num_t
    num_purple   = int(0.2 * total_cells)

    # reproducible purple-cell sampling
    random.seed(seed)
    purple_cells = random.sample(
        [(i, j) for i in range(num_x) for j in range(num_t)],
        num_purple
    )

    # ==============================================================
    # PLOT 1: Heat map
    # ==============================================================
    fig1, ax_img = plt.subplots(figsize=(7, 5))
    im = ax_img.imshow(
        np.flip(cell_density).T,
        cmap='viridis',
        extent=[x_edges[0], x_edges[-1], t_edges[0], t_edges[-1]],
        aspect='auto'
    )

    # ticks present but no tick labels
    ax_img.set_xticks(x)
    ax_img.set_yticks(t)
    ax_img.set_xticklabels([])      # ← **remove tick labels on x**
    ax_img.tick_params(axis='y', labelsize=14)

    ax_img.grid(which='minor', color='white', linewidth=0.4)
    ax_img.set_xlabel('x', fontsize=18)
    ax_img.set_ylabel('t', fontsize=18)

    # draw validation cells
    for i, j in purple_cells:
        rect = patches.Rectangle(
            (x_edges[i], t_edges[j]), dx, dt,
            facecolor='lightgray'
        )
        ax_img.add_patch(rect)

    plt.colorbar(im, ax=ax_img).set_label('cell density [a.u.]', fontsize=14)

    plt.tight_layout()
    if save_path_intro:
        plt.savefig(f"{save_path_intro}/ic{ic_num}_{seed}_heatmap.png", dpi=dpi)
    plt.show()

    # ==============================================================
    # PLOT 2: histogram
    # ==============================================================
    fig2, ax_hist = plt.subplots(figsize=(5, 4))
    all_vals    = cell_density.ravel()
    purple_vals = np.array([cell_density[i, j] for i, j in purple_cells])

    counts_all, edges = np.histogram(all_vals,    bins=bins)
    counts_val, _     = np.histogram(purple_vals, bins=edges)

    bin_centres = (edges[:-1] + edges[1:]) / 2
    vmin, vmax  = im.get_clim()
    norm = plt.Normalize(vmin, vmax)

    for k in range(len(bin_centres)):
        ax_hist.bar(
            edges[k], counts_all[k],
            align='edge',
            width=np.diff(edges)[k],
            color=cm.viridis(norm(bin_centres[k])),
            edgecolor='white'
        )

    ax_hist.bar(
        edges[:-1], counts_val,
        align='edge',
        width=np.diff(edges),
        color='lightgray', edgecolor='white'
    )

    ax_hist.set_xlabel('cell density [a.u.]', fontsize=16)   # ← updated label
    ax_hist.set_ylabel('count', fontsize=16)
    ax_hist.set_xlim(0, 1)

    legend_items = [
        mpatches.Patch(facecolor=cm.viridis(0.6), label='training cells'),
        mpatches.Patch(facecolor='lightgray', label='validation cells')
    ]
    ax_hist.legend(handles=legend_items, frameon=False, fontsize=12)

    plt.tight_layout()
    if save_path_intro:
        plt.savefig(f"{save_path_intro}/ic{ic_num}_{seed}_histogram.png", dpi=dpi)
    plt.show()

def TVplotter_lines_and_hist(
    data_obj,
    PDE_sim_old,
    PDE_RHS_1D,
    add_noise,
    ic_num=1,
    seed=42,
    bins=7,
    tnum=5,
    xnum=10,
    noise_lvl=0,
    save_path_intro="pngs/",
    dpi=100,
    color_scheme="green",  # "green" or "blue"
    marker_size=8          # NEW: master marker size
):
    # pick which IC
    ic_func = ic1 if ic_num == 1 else ic2

    # ------------------------------------------------------------------
    # Problem setup
    # ------------------------------------------------------------------
    x = np.linspace(np.min(data_obj.x), np.max(data_obj.x), xnum)
    t = np.linspace(np.min(data_obj.t), np.max(data_obj.t), tnum)

    K = data_obj.K
    gamma = data_obj.gamma

    # PDE solve
    u_clean = RDEq(
        x=x,
        t=t,
        u0=K * ic_func(x),
        clear=False,
        PDE_sim_old=PDE_sim_old,
        PDE_RHS_1D=PDE_RHS_1D
    )

    cell_density = add_noise(
        u_clean,
        gamma=gamma,
        noise_lvl=noise_lvl,
        seed=seed,
        t_start_idx=0
    )

    t = np.linspace(0, 4, tnum)

    # --------------------------------------
    # Validation “purple” cells (now red)
    # --------------------------------------
    num_x, num_t = len(x), len(t)
    total_cells = num_x * num_t
    num_purple = int(0.2 * total_cells)

    random.seed(seed)
    purple_cells = random.sample(
        [(i, j) for i in range(num_x) for j in range(num_t)],
        num_purple
    )

    # ==============================================================
    # PLOT 1: Line plot over x for different time slices
    # ==============================================================

    fig1, ax_line = plt.subplots(figsize=(7, 5))

    # choose cmap
    if color_scheme.lower() == "green":
        cmap = plt.cm.Greens
    elif color_scheme.lower() == "blue":
        cmap = plt.cm.Blues
    else:
        raise ValueError("color_scheme must be 'green' or 'blue'")

    colors = cmap(np.linspace(0.4, 0.9, num_t))

    # line plot with consistent marker size
    for j, (tj, c) in enumerate(zip(t, colors)):
        ax_line.plot(
            x,
            cell_density[:, j],
            color=c,
            linewidth=0.5,
            marker='o',
            markersize=marker_size,        # unified size
            markerfacecolor=c,
            markeredgecolor='black',
            markeredgewidth=0.5,
            label=f"t = {tj:.0f} days"
        )

    # overlay validation points as red hollow markers
    first_val_point = True
    for i, j in purple_cells:
        label = "validation points" if first_val_point else "_nolegend_"
        ax_line.scatter(
            x[i],
            cell_density[i, j],
            facecolors='none',
            edgecolors='red',
            lw=2,
            s=marker_size**2,              # unified size (scatter uses area)
            zorder=3,
            label=label
        )
        first_val_point = False

    ax_line.set_xlabel("x [mm]", fontsize=18)
    ax_line.set_ylabel("cell density [a.u.]", fontsize=18)
    ax_line.tick_params(axis="both", labelsize=14)
    ax_line.legend(frameon=False, fontsize=12)

    plt.tight_layout()
    if save_path_intro:
        plt.savefig(f"{save_path_intro}/ic{ic_num}_{seed}_lineplot.png", dpi=dpi)
    plt.show()

    # ==============================================================
    # PLOT 2: histogram
    # ==============================================================

    fig2, ax_hist = plt.subplots(figsize=(7, 5))

    all_vals = cell_density.ravel()
    purple_vals = np.array([cell_density[i, j] for i, j in purple_cells])

    # Force bin edges to fully span [0,1]
    edges = np.linspace(0, 1, bins + 1)

    counts_all, _ = np.histogram(all_vals, bins=edges)
    counts_val, _ = np.histogram(purple_vals, bins=edges)

    # nice mid-tone colors
    base_color = plt.cm.Greens(0.6) if color_scheme == "green" else plt.cm.Blues(0.6)
    val_color  = plt.cm.Reds(0.55)

    # training cells histogram
    ax_hist.bar(
        edges[:-1],
        counts_all,
        align="edge",
        width=np.diff(edges),
        color=base_color,
        edgecolor="black",
        linewidth=0.4,
        alpha=0.85,
        label="training cells"
    )

    # validation overlay
    ax_hist.bar(
        edges[:-1],
        counts_val,
        align="edge",
        width=np.diff(edges),
        color="r",
        edgecolor="black",
        linewidth=0.4,
        alpha=0.9,
        label="validation cells"
    )

    ax_hist.set_xlabel("cell density [a.u.]", fontsize=16)
    ax_hist.set_ylabel("count", fontsize=16)
    ax_hist.set_xlim(0, 1)

    ax_hist.legend(frameon=False, fontsize=12)

    plt.tight_layout()
    if save_path_intro:
        plt.savefig(f"{save_path_intro}/ic{ic_num}_{seed}_histogram.png", dpi=dpi)
    plt.show()

