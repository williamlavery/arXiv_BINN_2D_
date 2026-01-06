import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.ticker import ScalarFormatter
import numpy as np
from matplotlib.colors import LogNorm
import torch
from matplotlib.patches import Patch
#from __future__ import division, print_function
#from gplearn.genetic import SymbolicRegressor
from sympy import symbols, sympify, lambdify,latex
import sympy as sp
#from pysr import PySRRegressor
from scipy.optimize import minimize
import ast
import inspect
import itertools

def to_torch(ndarray, device):
    
    """
    Converts numpy array to torch.
    """
    arr = torch.tensor(ndarray, dtype=torch.float)
    arr.requires_grad_(True)
    arr = arr.to(device)
    return arr

def hist_properties(dataobj, num_bins_data_plot = 100, low = 5, high=95):
    
    # Compute valid u range based on histogram over the full u
    u = dataobj.u
    u_flat = u.flatten()

    hist, bin_edges = torch.histogram(torch.tensor(u_flat), bins=num_bins_data_plot)
    hist_np = hist.numpy()

    low_count_thresh = np.percentile(hist, low)
    high_count_thresh = np.percentile(hist, high)
    valid_bins = (hist >= low_count_thresh) & (hist <= high_count_thresh)
    bin_indices = torch.bucketize(torch.tensor(u_flat), bin_edges[1:-1])
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    return {"hist": hist,
            "bin_edges": bin_edges,
            "bin_indices":bin_indices,
            "bin_centers": bin_centers, 
            "low_count_thresh":low_count_thresh,
            "low_count": np.percentile(u, low),
            "high_count_thresh":high_count_thresh,
            "high_count": np.percentile(u, high)}


def prepare_model_run_data(dn_models_dics, plot_params=None):
    import numpy as np

    if plot_params is None:
        plot_params = {}

    run_data = []
    raw_timing_data = []
    model_keys = list(dn_models_dics.keys())
    dic_values = list(dn_models_dics.values())



    for i, (key, model_dic) in enumerate(zip(model_keys, dic_values)):
        modelWrappers = list(model_dic.values())

        sample_model = modelWrappers[0]
        #sf = sample_model.print_freq if len(sample_model.train_loss_list)- 3 > len(sample_model.epoch_times) else 1
        sf = 1
        # Extract individual plot parameters or use defaults
        color, linestyle, markerstyle, label = plot_params.get(
            key,
            ['C{}'.format(i), '-', '.', f'Run {i+1}']
        )
        val_losses_raw = []

        for m in modelWrappers:

            if len(m.val_loss_list):
                losses = np.array(m.val_loss_list, dtype=np.float64)
            else:
                losses = np.array(m.train_loss_list, dtype=np.float64)
            val_losses_raw.append(losses)

        epoch_times_arr = []
        for m in modelWrappers:
            arr = np.array(m.epoch_times, dtype=np.float64)
            p95 = np.percentile(arr, 80)
            clipped = arr[arr <= p95]
            epoch_times_arr.append(clipped)

        max_len_val = max(len(v) for v in val_losses_raw)
        val_losses = [
            np.pad(v, (0, max_len_val - len(v)), constant_values=np.nan)
            for v in val_losses_raw
        ]
        val_losses = np.vstack(val_losses)

        running_mins = np.full_like(val_losses, np.nan)
        for j in range(val_losses.shape[0]):
            v = val_losses[j]
            running_mins[j] = np.minimum.accumulate(np.nan_to_num(v, nan=np.inf))

        running_min_min = np.nanmin(running_mins, axis=0)
        
        running_min_max = np.nanmax(running_mins, axis=0)
        best_epoch_idx = np.nanargmin(running_min_min)
        epochs = np.arange(max_len_val)

        last_valid_vals = np.array([
            row[~np.isnan(row)][-1] if np.any(~np.isnan(row)) else np.nan
            for row in val_losses
        ])
        mean_final = np.nanmean(last_valid_vals)
        std_final = np.nanstd(last_valid_vals)

        hticks = []
        best_epochs_lst = []

        for idx, m in enumerate(modelWrappers):
            best_val_epoch  = np.argmin(np.abs(val_losses_raw[idx]-m.best_val_loss) )
            best_epochs_lst.append(best_val_epoch)
            hticks.append(m.best_val_loss)
        hticks = np.mean(hticks)

        final_loss = [m.best_val_loss for m in modelWrappers]


        label_with_stats = f"{label}"# ({mean_final:.2e} ± {std_final:.1e})"

        total_num_epochs = [len(m.epoch_times) for m in modelWrappers]

        mean_epoch_times = [np.mean(times) for times in epoch_times_arr]
        avg_epoch_time = np.mean(mean_epoch_times)
        std_epoch_time = np.std(mean_epoch_times)

        total_time_arr = [np.mean(times) * epoch_num * sf for times,epoch_num in zip(epoch_times_arr, total_num_epochs)]
        run_time_mean = np.mean(total_time_arr)
        run_time_std = np.std(total_time_arr)

        mean_str = f"{mean_final:.2e}"
        std_str = f"{std_final:.1e}"
        label_with_stats = f"{label} ({mean_str} ± {std_str})"

        run_data.append({
            'label': label,
            'mean_final_loss': np.mean(final_loss),
            'std_final_loss': np.std(final_loss),
            'mean_final': mean_final,
            'hticks': hticks,
            'best_epochs_lst': best_epochs_lst,
            'epochs': epochs,
            'running_min_min': running_min_min,
            'running_min_max': running_min_max,
            'best_epoch_idx': best_epoch_idx,
            'label_with_stats': label_with_stats,
            'label': label,
            'color': color,
            'linestyle': linestyle,
            'markerstyle': markerstyle
        })

        raw_timing_data.append({
            'label': label,
            'avg_epoch_time': avg_epoch_time,
            'std_epoch_time': std_epoch_time,
            'run_time_mean': run_time_mean,
            'run_time_std': run_time_std,
            'color': color
        })

    return run_data, raw_timing_data






def plot_initial_condition_1d(dataobj,
                               filename=None,
                               K_orig=1.7e3):
    """
    Plots 1D initial condition profiles using a fixed purple color palette and marker styles,
    then saves the plot as a transparent PNG.

    Parameters:
        dataobj: An object with attributes u, x, K, t
        filename (str): Output filename (without extension)
        K_orig (float): Scaling constant for u-values
    """
    u = dataobj.u
    x = dataobj.x
    t = dataobj.t

    markers = ['o', 's', 'D', '^', 'p', '<', '>', 'p', '*', 'h', 'x', '+']
    colors = ['#2E003E', '#5E239D', '#9013FE', '#B580FF', '#E5D5FF']
    grayscale_colors = ['#111111', '#444444', '#888888', '#BBBBBB', '#EEEEEE']

    t_labels = ["0"] + [f"{int(i)}/{u.shape[-1]-1} T" for i in range(1,u.shape[-1]-1)] + ["T"]
    fig, ax = plt.subplots(figsize=(5, 4))

    for tidx in range(u.shape[-1]):
        color = grayscale_colors[tidx % len(grayscale_colors)]
        marker = markers[tidx % len(markers)]
        ax.plot(
            x, u[:, tidx] * K_orig,
            label=f"t = {t_labels [tidx]}",
            markersize=4,
            color=color,
            marker=marker,
            lw=0,
            markeredgecolor='black',
            markeredgewidth=0.5
        )

    ax.set_xlabel('x [mm]')
    ax.set_ylabel(r'cell density [cells mm$^{-2}]$')
    ax.set_facecolor('white') 
    ax.legend(fontsize=9)
    plt.tight_layout()

    if filename:
        plt.savefig(f"{filename}.png",  dpi=100, bbox_inches='tight', facecolor='None')

    err = np.abs(u - dataobj.u_clean)

    print("MSE between u and u_clean:", K_orig*np.mean(err**2))
    print("ABS between u and u_clean [cells]:", K_orig*np.mean(err))
    print("ABS (%) between u and u_clean:", 100*np.mean(err/dataobj.u_clean))
    plt.show()

def plot_eval2(modelWrappers_dic, 
                    dataobj,
                    IC = 0,
                    device = "cpu",
                    num_bins = 10,
                    label='',
                    clean=True,
                    noisy=True,
                    error=True,
                    K=1700,
                    save_name=None,
                    colors=None):
    
    if not colors:
        colors = ['#9013FE', '#5E239D','#B580FF', '#E5D5FF']
    markers = ['o', 's', 'D', '^', 'p', '<', '>', 'p', '*', 'h', 'x', '+']
    grayscale_colors = ['#111111', '#444444', '#888888', '#BBBBBB', '#EEEEEE']

    # All models for same IC will have same data distribution.
    # So we extract the first model

    modelWrapper = list(modelWrappers_dic.values())[0]

    u_val = np.array([k for (_, _), k in zip(modelWrapper.x_val, modelWrapper.y_val)])
    u_train = np.array([k for (_, _), k in zip(modelWrapper.x_train, modelWrapper.y_train)])


    model = modelWrapper.model

    # Get data
    x = dataobj.x
    u = dataobj.u*K
    u_clean = dataobj.u_clean*K
    t = dataobj.t
    Nt, Nx = len(dataobj.t), len(dataobj.x)

    # Flattened u values
    u_flat = u.flatten()


    # Histogram properties
    model.eval()
    h_properties = hist_properties(dataobj, num_bins)

    hist = h_properties["hist"]
    bin_edges = h_properties["bin_edges"]*1700
    bin_centers = h_properties["bin_centers"]*1700
    low_count_thresh = h_properties["low_count_thresh"]*1700
    high_count_thresh = h_properties["high_count_thresh"]*1700

    # Compute separate histograms for train/val
    hist_train, _ = np.histogram(u_train*1700, bins=bin_edges)
    hist_val, _ = np.histogram(u_val*1700, bins=bin_edges)

    # Plot stacked bar chart
    fig, ax = plt.subplots(figsize=(5, 4))

    bar_width = bin_edges[1] - bin_edges[0]

    # Plot validation bars first (on bottom)
    ax.bar(
        bin_centers, hist_val, width=bar_width, alpha=0.7,
        label="Validation", color=colors[1],
        edgecolor='black', linewidth=0.5
    )

    # Then plot train bars on top of validation
    ax.bar(
        bin_centers, hist_train, width=bar_width, alpha=0.7,
        bottom=hist_val, label="Train", color=colors[0],
        edgecolor='black', linewidth=0.5
    )

    # Add threshold lines
    #ax.axhline(low_count_thresh, color='r', linestyle='--', label="5% freq")
    #ax.axhline(high_count_thresh, color='g', linestyle='--', label="95% freq")

    # Labeling
    #ax.set_title(f"u Value Histogram (Train vs Validation) – Dataset {IC}")
    ax.set_xlabel("cell density [cells mm$^{-2}]$", fontsize=11)
    ax.set_ylabel("frequency", fontsize=11)
    ax.legend()
    #ax.grid(True, ls=":", lw=0.4)

    plt.tight_layout()
    plt.show()

    #=============
    for modelWrapper_indx in modelWrappers_dic.keys():

        modelWrapper = modelWrappers_dic[modelWrapper_indx]

        x_val = modelWrapper.x_val
        y_val = modelWrapper.y_val
        x_train = modelWrapper.x_train
        y_train = modelWrapper.y_train

        # Compute training and validation u values
        u_val = np.array([k for (_, _), k in zip(x_val, y_val)])
        u_train = np.array([k for (_, _), k in zip(x_train, y_train)])

        model = modelWrapper.model

        with torch.no_grad():
            
            u_pred_flat = model(to_torch(dataobj.inputs, device))*K  # Use direct net call (no ID input)
            u_pred = u_pred_flat.reshape(Nx, Nt).cpu().numpy()

        

  
        fig, ax1 = plt.subplots(figsize=(5, 4))

        # === First subplot: Predictions and truth at t=0 and t=T ===


        ax1.plot(x, u_pred[:, 0], "-", lw=2, alpha=0.7, color=colors[0], label=r"$\hat{u}_{dn}(0,x)$")
        ax1.plot(x, u_pred[:, -1], "-", lw=2, alpha=0.7, color=colors[1], label=r"$\hat{u}_{dn}(T,x)$")

        if error:
            ax2 = ax1.twinx()
            ax2.set_ylabel("abs. error [cells mm$^{-2}]$", fontsize=11)
            ax2.set_yscale('log')


        # --- Primary axis (cell density)

        abs_e = np.abs(u_pred - u_clean)
        mse = np.mean(abs_e)
        ax1.plot(
            x, u_clean[:, 0],
            label=f"t=0",
            markersize=4,
            color=grayscale_colors[0],
            marker=markers[0],
            lw=0,
            markeredgecolor='black',
            markeredgewidth=0.5
        )

        ax1.plot(
            x, u_clean[:, -1],
            label=f"t=T",
            markersize=4,
            color=grayscale_colors[4],
            marker=markers[4],
            lw=0,
            markeredgecolor='black',
            markeredgewidth=0.5
        )


        if noisy:
            ax1.plot(
            x, u[:, 0],
            label=f"t=0",
            color=grayscale_colors[0],
            lw=1
            )

            ax1.plot(
                x, u[:, -1],
                label=f"t=T",
                color=grayscale_colors[3],
                lw=1
            )

        
        if error:
            # --- Secondary axis ()
            ax2.plot(
                x, abs_e[:, 0],
                label=f"err(x,0)",
                markersize=4,
                color='#8b0000',
                marker=markers[0],
                markeredgecolor='black',
                markeredgewidth=0.5,
                alpha=0.7,
                lw=1,
            )

            ax2.plot(
                x, abs_e[:, -1],
                label=f"err(x,T)",
                markersize=4,
                color="#dd0505",
                marker=markers[4],
                markeredgecolor='black',
                markeredgewidth=0.5,
                alpha=0.7,
                lw=1,
            )


        # --- Axis labels
        ax1.set_xlabel("x [mm]", fontsize=11)
        ax1.set_ylabel("cell density [cells mm$^{-2}]$", fontsize=11)

        # --- Optional stats box
        textstr = f"MSE = {mse:.3e}"
        props = dict(boxstyle="round", facecolor="white", alpha=0.85, lw=0.1)
        # ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=11, verticalalignment="top", bbox=props)

        
        # --- Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        if error:
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower right', fontsize=8, ncol=1)
        else:
            ax1.legend(lines1, labels1, loc='lower right', fontsize=8, ncol=1)

        ax.set_facecolor('white') 
        plt.tight_layout()

        if save_name:
            plt.savefig(save_name,  dpi=100, bbox_inches='tight', facecolor='None')
        
def plot_running_min_val_loss_broken_x_log(dn_models_dics, 
                                           plot_params=None,
                                           plot_settings=None):
    import matplotlib.pyplot as plt
    import numpy as np

    # Default plot settings
    default_settings = {
        'xaxis': {'min': 1, 'max': 1e5, 'break': 1e3},
        'legend': {'panel': 1, 'loc': 'lower left', 'fontsize': 10, 'ncols':1},
        'name': 'running_min_val_loss_broken_xaxis_loglog.png',
        'fill': True,
        'line_lengths': {'hlength': 10000, 'vlength_factor': 2.0},
        'line_widths_on_axis': {'hwidth': 1, 'vwidth': 1},
        'line_width':1.5,
        'fontsizes': {'xaxis':12, 'xtick_labels':10, 'yaxis':12, 'ytick_labels':10}
    }

    # Merge user-provided settings with defaults
    plot_settings = plot_settings or {}
    settings = {**default_settings, **plot_settings}
    xaxis = {**default_settings['xaxis'], **settings.get('xaxis', {})}
    legend = {**default_settings['legend'], **settings.get('legend', {})}
    name = settings.get('name', default_settings['name'])
    fill = settings.get('fill', default_settings['fill'])
    line_width = settings.get('line_width', default_settings['line_width'])
    line_width_on_axis = {**default_settings['line_widths_on_axis'], **settings.get('line_widths_on_axis', {})}
    line_lengths = {**default_settings['line_lengths'], **settings.get('line_lengths', {})}
    fontsizes = {**default_settings['fontsizes'], **settings.get('fontsizes', {})}

    xfont = fontsizes['xaxis']
    xtick_font = fontsizes['xtick_labels']
    yfont = fontsizes['yaxis']
    ytick_font = fontsizes['ytick_labels']

    hlength = line_lengths['hlength']             # for horizontal lines (x-range)
    hwidth = line_width_on_axis["hwidth"]
    vwidth = line_width_on_axis["vwidth"]
    vlength_factor = line_lengths['vlength_factor']  # for vertical lines (multiplier on y_base)
    
    

    break_x = xaxis['break']
    x_min = xaxis['min']
    x_max = xaxis['max']

    # Prepare run data
    run_data, _ = prepare_model_run_data(dn_models_dics, plot_params=plot_params)
    run_data.sort(key=lambda x: x['hticks'])  # Sort by final loss

    # Set up broken-axis plot
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(8, 5),
                                   gridspec_kw={'width_ratios': [1, 5], 'wspace': 0.05})

    for run in run_data:
        x = np.where(run['epochs'] == 0, 1e-1, run['epochs'])  # avoid log(0)
        y_min = run['running_min_min']
        color = run['color']

        linestyle = run.get('linestyle', '-')
        markerstyle = run.get('markerstyle', '-')

        mask1 = x <= break_x
        mask2 = x > break_x

        ax1.plot(x[mask1], y_min[mask1], color=color, label=run['label'], linestyle=linestyle, lw=line_width)
        ax2.plot(x[mask2], y_min[mask2], color=color, label=run['label'], linestyle=linestyle, lw=line_width)
        ax2.scatter(x[run['best_epoch_idx']], y_min[run['best_epoch_idx']], color=color, s=20, zorder=5, marker=markerstyle)

        if fill:
            y_max = run['running_min_max']
            ax1.fill_between(x[mask1], y_min[mask1], y_max[mask1], alpha=0.2, color=color)
            ax2.fill_between(x[mask2], y_min[mask2], y_max[mask2], alpha=0.2, color=color)

    # Axis formatting
    for ax in [ax1, ax2]:
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_facecolor('white')

    ax2.set_xlabel("epoch (log)", fontsize=xfont)
    ax1.set_xlim(left=x_min, right=break_x)
    ax2.set_xlim(left=break_x, right=x_max)
    ax1.set_ylabel("running min val loss [a.u]", fontsize=yfont)

    # Axis break visuals
    ax1.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax1.tick_params(labelright=False)
    ax2.tick_params(labelleft=False)

    ax1.yaxis.set_tick_params(labelsize=ytick_font)
    ax1.xaxis.set_tick_params(labelsize=xtick_font)
    ax2.xaxis.set_tick_params(labelsize=xtick_font)

    d = .015
    kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
    ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)
    ax1.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
    kwargs.update(transform=ax2.transAxes)
    ax2.plot((-d, +d), (-d, +d), **kwargs)
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)

    # Legend
    if legend['panel'] == 1:
        ax1.legend(fontsize=legend['fontsize'], loc=legend['loc'], ncols=legend['ncols'])
    else:
        ax2.legend(fontsize=legend['fontsize'], loc=legend['loc'], ncols=legend['ncols'])


    ax2_right = ax2.twinx()
    y_min, y_max = ax1.get_ylim()
    y_top = y_min * vlength_factor

    # Optional: clean axis (no ticks/labels)
    ax2_right.tick_params(axis='y', which='both', left=False, right=False, labelleft=False, labelright=False)


    # Plot one colored circle per run at mean_final
    for run in run_data:

        linestyle = run.get('linestyle', '-')
        color = run['color']

        ax2.hlines(
            y=run['hticks'],
            xmin=x_max - hlength,
            xmax=x_max,
            color=color,
            linestyle=linestyle,
            linewidth=hwidth,
            zorder=10
        )

        best_epochs_lst = run['best_epochs_lst']
        to_plot = np.min(best_epochs_lst)

        if to_plot < break_x:
            ax1.vlines(to_plot, y_min, y_top,
                    color=color, linestyle=linestyle, lw=vwidth)
        ax2.vlines(to_plot, y_min, y_top,
                    color=color, linestyle=linestyle, lw=vwidth)

    ax2.set_ylim(y_min)
    fig.tight_layout()
    if name:
        plt.savefig(name, dpi=100, bbox_inches='tight', facecolor='None')
        print("saved plot:", name)
    plt.show()
    

    return {
        'name': name,
        'legend': legend,
        'fill': fill,
        'xaxis': xaxis,
        'num_runs': len(run_data)
    }

# ------------------------------------------------------------------
# 3.  Plot every repeat on top of each other
# ------------------------------------------------------------------


def plot_repeats_u(binn_models_dics, base_colors, filename=None,
                 bbox_to_anchor=(1.02, 1.0), legend_fontsize=10):

    hatch_styles = ['', '//', '..']

    depths = list(binn_models_dics.keys())
    widths = list(binn_models_dics[depths[0]].keys())

    # ------------------------------------------------------------------
    # 2.  Pull every repeat into a flat {(w,d): [values]} dictionary
    # ------------------------------------------------------------------
    def collect_metric(binn_models, metric_attr):
        """
        Return {(width, depth): [metric per repeat, …]} for the given attribute.
        """
        out = {}
        for d in depths:
            for w in widths:
                repeats = binn_models[d][w]            # dict {seed: model}
                out[(w, d)] = [getattr(m, metric_attr) for m in repeats.values()]
        return out
    data = collect_metric(binn_models_dics, 'best_val_loss')
    

    bar_width        = 0.15
    intra_gap, inter_gap = 0.02, 0.30
    current_x        = 0.0

    max_repeats      = max(len(v) for v in data.values())
    label_cmap       = cm.get_cmap('tab10', max_repeats)      # colour for tiny text tags

    xtick_positions = []
    xtick_labels = []

    plt.figure(figsize=(7, 5))

    for i_w, (w, base_c) in enumerate(zip(widths, base_colors)):
        group_x_positions = []

        for i_d, d in enumerate(depths):
            vals = data[(w, d)]

            # draw all repeats for this (width, depth) on the SAME x
            for i_r, v in enumerate(vals):
                alpha = 0.9 - 0.15 * i_r
                plt.bar(current_x,
                        v,
                        width=bar_width,
                        color=base_c,
                        alpha=max(alpha, 0.25),
                        hatch=hatch_styles[i_d],
                        edgecolor='k')

                # Tiny label
                plt.text(
                    current_x, v, f'R{i_r + 1}',
                    ha='center', va='bottom',
                    fontsize=6,
                    color='black',
                    bbox=dict(facecolor='white', edgecolor='none', pad=0.3, alpha=0.9)
                )

            group_x_positions.append(current_x)
            current_x += bar_width + (intra_gap if i_d < len(depths) - 1 else inter_gap)

        # Center the xtick at the middle of the group
        center_x = np.mean(group_x_positions)
        xtick_positions.append(center_x)
        #xtick_labels.append(rf"NN$_\mathrm{{u}}$={w}")
        xtick_labels.append(f"{w}")

    # Cosmetics
    plt.xticks(xtick_positions, xtick_labels)
    plt.yscale('log')
    yaxis_label=r"Validation loss [a.u.]"
    #yaxis_label=r"MSE [day$^{-2}$ mm$^{4}$]"
    plt.ylabel(yaxis_label)

    # Only depth in legend
    depth_patches = [Patch(facecolor='white', edgecolor='k',
                           hatch=hatch_styles[i],
                           label=rf'NN$_\mathrm{{D}}$ = {d}') for i, d in enumerate(depths)]
    plt.legend(handles=depth_patches, loc='upper left', bbox_to_anchor=bbox_to_anchor, fontsize=legend_fontsize)


    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    if filename:
        plt.savefig(f"{filename}",  dpi=100, bbox_inches='tight', facecolor='None')
    plt.show()

def plot_repeats_D(binn_models_dics, base_colors,filename=None,
                 bbox_to_anchor=(1.02, 1.0), legend_fontsize=10):


    depths = list(binn_models_dics.keys())
    widths = list(binn_models_dics[depths[0]].keys())

    hatch_styles = ['', '//', '..']
    
    # ------------------------------------------------------------------
    # 2.  Pull every repeat into a flat {(w,d): [values]} dictionary
    # ------------------------------------------------------------------
    def collect_metric(binn_models, metric_attr):
        """
        Return {(width, depth): [metric per repeat, …]} for the given attribute.
        """
        out = {}
        for d in depths:
            for w in widths:
                repeats = binn_models[d][w]            # dict {seed: model}
                out[(w, d)] = [getattr(m, metric_attr) for m in repeats.values()]
        return out

    data = collect_metric(binn_models_dics, 'best_diffusion_error')
    

    bar_width        = 0.15
    intra_gap, inter_gap = 0.02, 0.30
    current_x        = 0.0

    max_repeats      = max(len(v) for v in data.values())
    label_cmap       = cm.get_cmap('tab10', max_repeats)      # colour for tiny text tags

    xtick_positions = []
    xtick_labels = []

    plt.figure(figsize=(7, 5))

    for i_w, (w, base_c) in enumerate(zip(widths, base_colors)):
        group_x_positions = []

        for i_d, d in enumerate(depths):
            vals = data[(w, d)]

            # draw all repeats for this (width, depth) on the SAME x
            for i_r, v in enumerate(vals):
                alpha = 0.9 - 0.15 * i_r
                plt.bar(current_x,
                        v,
                        width=bar_width,
                        color=base_c,
                        alpha=max(alpha, 0.25),
                        hatch=hatch_styles[i_d],
                        edgecolor='k')

                # Tiny label
                plt.text(
                    current_x, v, f'R{i_r + 1}',
                    ha='center', va='bottom',
                    fontsize=6,
                    color='black',
                    bbox=dict(facecolor='white', edgecolor='none', pad=0.3, alpha=0.9)
                )

            group_x_positions.append(current_x)
            current_x += bar_width + (intra_gap if i_d < len(depths) - 1 else inter_gap)

        # Center the xtick at the middle of the group
        center_x = np.mean(group_x_positions)
        xtick_positions.append(center_x)
        #xtick_labels.append(rf"NN$_\mathrm{{u}}$={w}")
        xtick_labels.append(f"{w}")

    # Cosmetics
    plt.xticks(xtick_positions, xtick_labels)
    plt.yscale('log')
    yaxis_label=r"Diffusion MSE [day$^{-2}$ mm$^{4}$]"
    plt.ylabel(yaxis_label)

    # Only depth in legend
    depth_patches = [Patch(facecolor='white', edgecolor='k',
                           hatch=hatch_styles[i],
                           label=rf'NN$_\mathrm{{D}}$ = {d}') for i, d in enumerate(depths)]

    plt.legend(handles=depth_patches, loc='upper left', bbox_to_anchor=bbox_to_anchor, fontsize=legend_fontsize)


    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    if filename:
        plt.savefig(f"{filename}.png",  dpi=100, bbox_inches='tight', facecolor='None')
    plt.show()

def prepare_diff_run_data(dn_models_dics, plot_params=None):
    import numpy as np

    if plot_params is None:
        plot_params = {}

    run_data = []
    model_keys = list(dn_models_dics.keys())
    dic_values = list(dn_models_dics.values())

    # Determine scaling factor for epoch time based on print frequency
    sample_model = list(dic_values[0].values())[0]
    sf = sample_model.print_freq if len(sample_model.train_loss_list) > len(sample_model.epoch_times) else 1

    for i, (key, model_dic) in enumerate(zip(model_keys, dic_values)):
        model_wrappers = list(model_dic.values())

        # Use user-defined or default styling
        color, linestyle, markerstyle, label = plot_params.get(
            key, [f'C{i}', '-', '.', f'Run {i+1}']
        )

        best_epochs_lst = []
        diff_errs_raw = [[] for _ in range(len(model_wrappers))]
        diff_errs = []

        max_len = 0
        for m in model_wrappers:
            best_val_epoch = np.argmin(np.abs(np.array(m.val_loss_list) - m.best_val_loss))
            best_epochs_lst.append(best_val_epoch)
            max_len = max(max_len, best_val_epoch)
        max_len = max_len // sf + 1

        for j, m in enumerate(model_wrappers):
            try:
                total_num_epochs = len(m.train_loss_list)
                best_epochs_scaled = best_epochs_lst[j] // sf + 1
                padded = np.full(max_len, np.nan)
                for diff_err in m.diffusion_errors:
                    diff_errs_raw[j].append(torch.mean(diff_err[:, 0]).detach().cpu().numpy())
                padded[:best_epochs_scaled] = diff_errs_raw[j][:best_epochs_scaled]
                diff_errs.append(padded)
            except:
                diff_err = m.diffusion_errors[:best_epochs_lst[j] // sf]
                padded_diff_err = np.pad(np.array(diff_err), (0, max_len - len(diff_err)), constant_values=np.nan)
                diff_errs.append(padded_diff_err)

        diff_errs_arr = np.vstack(diff_errs)
        running_mins = np.minimum.accumulate(np.nan_to_num(diff_errs_arr, nan=np.inf), axis=1)
        running_min_min = np.nanmin(running_mins, axis=0)
        running_min_max = np.nanmax(running_mins, axis=0)
        best_epoch_idx = np.nanargmin(running_min_min)
        epochs = np.arange(max_len)

        mean_vals = np.nanmean(diff_errs_arr, axis=0)
        std_vals = np.nanstd(diff_errs_arr, axis=0)

        
        try:
            hticks = np.mean([m.best_diffusion_error for m in model_wrappers])
        except:
            hticks = []
            for i, m in enumerate(model_wrappers):
                hticks.append(diff_errs_arr[i][best_epochs_lst[i] // sf])
            hticks = np.mean(hticks)

        try:
            last_vals = np.array([mw.best_diffusion_error for mw in model_wrappers])
        except:
            last_vals = np.array([
                row[~np.isnan(row)][-1] if np.any(~np.isnan(row)) else np.nan
                for row in running_mins
            ])

        mean_final = np.nanmean(last_vals)
        std_final = np.nanstd(last_vals)
        label_with_stats = f"{label}"# ({mean_final:.2e} ± {std_final:.1e})"

        run_data.append({
            'mean_final': mean_final,
            'epochs': epochs,
            'mean_vals': mean_vals,
            'std_vals': std_vals,
            'hticks':hticks,
            'min_vals': np.nanmin(diff_errs_arr, axis=0),
            'max_vals': np.nanmax(diff_errs_arr, axis=0),
            'running_min_min': running_min_min,
            'running_min_max': running_min_max,
            'best_epoch_idx': best_epoch_idx,
            'best_epochs_lst': np.array(best_epochs_lst),
            'label': label_with_stats,
            'color': color,
            'linestyle': linestyle,
            'markerstyle': markerstyle
        })

    return run_data

def plot_running_min_MSE_diff_loss_broken_x_log(dn_models_dics, 
                                                plot_params=None,
                                                plot_settings=None):
    import matplotlib.pyplot as plt
    import numpy as np

    default_settings = {
        'xaxis': {'min': 1, 'max': 1e5, 'break': 1e3},
        'legend': {'panel': 1, 'loc': 'lower left', 'fontsize': 10, 'ncols':1},
        'name': 'running_min_val_loss_broken_xaxis_loglog.png',
        'fill': True,
        'line_lengths': {'hlength': 10000, 'vlength_factor': 2.0},
        'line_width':1.5,
        'markersize':4,
        'line_widths_on_axis': {'hwidth': 1, 'vwidth': 1},
        'fontsizes': {'xaxis':12, 'xtick_labels':10, 'yaxis':12, 'ytick_labels':10}
    }

    plot_settings = plot_settings or {}
    settings = {**default_settings, **plot_settings}
    xaxis = {**default_settings['xaxis'], **settings.get('xaxis', {})}
    legend = {**default_settings['legend'], **settings.get('legend', {})}
    name = settings.get('name', default_settings['name'])
    fill = settings.get('fill', default_settings['fill'])
    line_width = settings.get('line_width', default_settings['line_width'])
    line_lengths = {**default_settings['line_lengths'], **settings.get('line_lengths', {})}
    hlength = line_lengths['hlength']             # for horizontal lines (x-range)
    vlength_factor = line_lengths['vlength_factor']  # for vertical lines (multiplier on y_base)
    markersize = settings.get('markersize', 4)

    line_width_on_axis = {**default_settings['line_widths_on_axis'], **settings.get('line_widths_on_axis', {})}
    fontsizes = {**default_settings['fontsizes'], **settings.get('fontsizes', {})}
    xfont = fontsizes['xaxis']
    xtick_font = fontsizes['xtick_labels']
    yfont = fontsizes['yaxis']
    ytick_font = fontsizes['ytick_labels']
    hwidth = line_width_on_axis["hwidth"]
    vwidth = line_width_on_axis["vwidth"]


    break_x = xaxis['break']
    x_min = xaxis['min']
    x_max = xaxis['max']

    run_data = prepare_diff_run_data(dn_models_dics, plot_params)
    run_data.sort(key=lambda x: x['hticks'])

    sample_model = list(list(dn_models_dics.values())[0].values())[0]
    sf = sample_model.print_freq if len(sample_model.train_loss_list) > len(sample_model.epoch_times) else 1

    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(8, 5),
                                   gridspec_kw={'width_ratios': [1, 5], 'wspace': 0.05})

    for run in run_data:
        x = np.where(run['epochs'] * sf == 0, 1e-1, run['epochs'] * sf)
        y_mean = run['mean_vals']
        y_min = run['min_vals']
        y_max = run['max_vals']
        color = run['color']

        mask1 = x <= break_x
        mask2 = x > break_x
        
        markerstyle = run.get('markerstyle', '-')
        linestyle = run.get('linestyle', '-')

        ax1.plot(x[mask1], y_mean[mask1], color=color, label=run['label'], lw=line_width, linestyle=linestyle,marker=markerstyle, markersize=markersize)
       

        ax2.plot(x[mask2], y_mean[mask2], color=color, label=run['label'], lw=line_width, linestyle=linestyle,marker=markerstyle, markersize=markersize)
        if fill:
            ax1.fill_between(x[mask1], y_min[mask1], y_max[mask1], alpha=0.2, color=color)
            ax2.fill_between(x[mask2], y_min[mask2], y_max[mask2], alpha=0.2, color=color)

    for ax in [ax1, ax2]:
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_facecolor('white')

    ax1.yaxis.set_tick_params(labelsize=ytick_font)
    ax1.xaxis.set_tick_params(labelsize=xtick_font)
    ax2.xaxis.set_tick_params(labelsize=xtick_font)

    ax2.set_xlabel("epoch (log)", fontsize=xfont)
    ax1.set_xlim(left=x_min, right=break_x)
    ax2.set_xlim(left=break_x, right=x_max)
    ax1.set_ylabel(r"Diffusion MSE [mm$^4$ day$^{-2}$]", fontsize=yfont)

    ax1.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax1.tick_params(labelright=False)
    ax2.tick_params(labelleft=False)

    d = .015
    kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
    ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)
    ax1.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
    kwargs.update(transform=ax2.transAxes)
    ax2.plot((-d, +d), (-d, +d), **kwargs)
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)

    if legend['panel'] == 1:
        ax1.legend(fontsize=legend['fontsize'], loc=legend['loc'], ncols=legend['ncols'])
    else:
        ax2.legend(fontsize=legend['fontsize'], loc=legend['loc'], ncols=legend['ncols'])

    ax2_right = ax2.twinx()
    y_min_val, y_max_val = ax1.get_ylim()
    y_base = y_min_val
    y_top = y_min_val * 2

    ax2_right.set_yscale('log')
    ax2_right.set_ylim(ax2.get_ylim())
    ax2_right.tick_params(axis='y', which='both', left=False, right=False, labelleft=False, labelright=False)




    ax2_right = ax2.twinx()
    y_min, y_max = ax1.get_ylim()
    y_top = y_min * vlength_factor
    ax2_right.tick_params(axis='y', which='both', left=False, right=False, labelleft=False, labelright=False)

    # Plot one colored circle per run at mean_final
    for run in run_data:

        linestyle = run.get('linestyle', '-')
        color = run['color']


        ax2.hlines(
            y=run['hticks'],
            xmin=x_max - hlength,
            xmax=x_max,
            color=color,
            linestyle=linestyle,
            linewidth=hwidth,
            zorder=10
        )

        best_epochs_lst = run['best_epochs_lst']
        to_plot = np.min(best_epochs_lst)

        if to_plot < break_x:
            ax1.vlines(to_plot, y_min, y_top,
                    color=color, linestyle=linestyle, lw=vwidth)
        ax2.vlines(to_plot, y_min, y_top,
                    color=color, linestyle=linestyle, lw=vwidth)

    ax2.set_ylim(y_min)
    fig.tight_layout()

    if name:
        plt.savefig(name, dpi=100, bbox_inches='tight', facecolor='None')
        print("saved plot:", name)
    plt.show()

def plot_epoch_and_total_times(dn_models_dics, plot_params=None, name=None):

    import matplotlib.pyplot as plt
    import numpy as np

    if name==None:
        name = "case11_bar.png"

    _, raw_timing_data = prepare_model_run_data(dn_models_dics, plot_params=plot_params)

    x = np.arange(len(raw_timing_data))
    avg_times = [d['avg_epoch_time'] for d in raw_timing_data]
    std_times = [d['std_epoch_time'] for d in raw_timing_data]
    run_time_means = [d['run_time_mean'] for d in raw_timing_data]
    run_time_stds = [d['run_time_std'] for d in raw_timing_data]
    scatter_colors = [d['color'] for d in raw_timing_data]
    bar_labels = [d['label'] for d in raw_timing_data]

    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax2 = ax1.twinx()

    ax1.errorbar(x, avg_times, yerr=std_times, fmt='o', capsize=4, color='black', label='Avg Epoch Time')
    ax1.set_ylabel("Avg Epoch Time [s]", fontsize=11, color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    #ax1.set_ylim(bottom=0)

    for i in range(len(x)):
        ax2.bar(
            x[i],
            run_time_means[i],
            yerr=run_time_stds[i],
            capsize=5,
            color=scatter_colors[i],
            edgecolor='gray',
            alpha=0.4,
            error_kw=dict(ecolor="gray", alpha=1, linewidth=1.5)
        )

    ax2.set_ylabel("Total Run Time [s]", fontsize=11, color=scatter_colors[0])
    ax2.tick_params(axis='y', labelcolor=scatter_colors[0])
    #ax2.set_ylim(bottom=0)

    ax1.set_xticks(x)
    ax1.set_xticklabels(bar_labels, fontsize=11)

    fig.tight_layout()
    if name:
        plt.savefig(name, dpi=100, bbox_inches='tight', facecolor='None')
        print("saved plot:", name)
    plt.show()

def symbolic_from_function(func, var_name='u'):
    # Get source code
    source = inspect.getsource(func).strip()

    # Parse the function's AST
    tree = ast.parse(source)

    # Get the return statement's expression
    return_node = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.Return)
    )

    # Create a mapping for allowed names (e.g., math, np, etc.)
    allowed_names = func.__globals__.copy()

    # Create symbolic variable
    u = sp.Symbol(var_name)

    # Evaluate the return expression in symbolic context
    expr = eval(
        compile(ast.Expression(return_node.value), "<ast>", "eval"),
        {**allowed_names, var_name: u}
    )

    return expr

def plot_eval_D_multi(modelWrapper_dics, dataobj, D_sym_true_lst, colors, labels, Dnum=1,
                      device="cpu", num_bins=50, K=1700, name=None, fill=True,
                      legend_pos = (0.5,0.5),
                      linestyles = itertools.cycle(['-', ':', '-.', (0, (3, 1, 1, 1)), (0, (1, 1))]),
                      errs=None):
    if errs:
        [err_up, err_low] = errs
    colors = colors[:len(labels)]
    h_properties = hist_properties(dataobj, num_bins)
    low_u = h_properties["low_count"]
    high_u = h_properties["high_count"]

    sample_model = list(list(modelWrapper_dics.values())[0].values())[0].model
    u_vals_torch = sample_model.u_vals_torch
    u_vals_np = sample_model.u_vals.flatten()
    
    results = []
    stored_results = []
    D_true_lst = []
    for wrapper_dic, color, label in zip(modelWrapper_dics.values(), colors, labels):
        diffusion_errors = []
        D_ensemble = []
        
        for i, wrapper in enumerate(wrapper_dic.values()):
            D_true_check = list(wrapper.model.D_true)
            if i==0 and D_true_check  not in D_true_lst:
                D_true_lst.append(D_true_check)
            model = wrapper.model
            model.eval()
            best_val_epoch  = np.argmin(np.abs(np.array(wrapper.val_loss_list)- wrapper.best_val_loss)) 
            with torch.no_grad():
                diff_pred = model.D_scale * model.diffusion(model.u_vals_torch).flatten()
                diffusion_error = (model.D_true_torch - diff_pred) ** 2
                diffusion_errors.append(diffusion_error.unsqueeze(0))  # shape: [1, N]
                D_ensemble.append(diff_pred.unsqueeze(0))  
        D_ensemble = torch.cat(D_ensemble, dim=0)       # shape: [runs, N]
        D_mean = torch.mean(D_ensemble, dim=0)          # shape: [N]
        D_min = torch.min(D_ensemble, dim=0).values
        D_max = torch.max(D_ensemble, dim=0).values
        diffusion_errors = torch.cat(diffusion_errors, dim=0)  # shape: [runs, N]
        mse = torch.mean(diffusion_errors).item()              # scalar
        results.append((mse, label, color, D_mean.numpy(), D_min.numpy(), D_max.numpy()))
    # Sort by MSE
    #results.sort(key=lambda x: x[0])
    # Plot
    fig, ax = plt.subplots(figsize=(7, 5))
    j=0
    for mse, label, color, D_mean, D_min, D_max in results:
        ls = linestyles[j]
        ax.plot(u_vals_np * K, D_mean, lw=2, color=color, linestyle=ls, label=label)

        if fill:
            ax.fill_between(u_vals_np * K, D_min, D_max, alpha=0.4, color=color)
        j+=1
       # err = np.abs(D_mean - D_true_lst[0])
       # PE = err/D_true_lst[0]
       # print(f"MSE {label} (%)", np.mean(PE))



    for i, (D_true, D_sym) in enumerate(zip(D_true_lst, D_sym_true_lst)):
        if i==0:
            ax.plot(u_vals_np * K, D_true, "--", lw=2, color="k", label=f"$D_{Dnum}(u)$", zorder=3)

            # 1% above/below
            if errs is not None:
                ax.plot(u_vals_np * K, np.array(D_true)* (1 + err_up/100), "--", lw=1, color="r", label=f"{err_up}%", zorder=3)
                ax.plot(u_vals_np * K,np.array(D_true)* (1 - err_low/100), "-.", lw=1, color="r", label=f"{err_low}%", zorder=3)
        else:
            ax.plot(u_vals_np * K, D_true, "--", lw=2, color="k", zorder=3)
    
    
    
    ax.axvline(low_u * K, color="#666666", ls="-.", lw=1, label="5% $u$-perc.")
    ax.axvline(high_u * K, color="#BBBBBB", ls="--", lw=1, label="95% $u$-perc.")
    ax.set_xlabel("cell density [cells mm$^{-2}]$", fontsize=11)
    ax.set_ylabel(r"cell diffusion [mm$^2$ days$^{-1}$]", fontsize=11)
    ax.set_facecolor('white')
    ax.legend(
    loc='center',               # anchor point of the legend box (can also use 'upper left', etc.)
    bbox_to_anchor=legend_pos, # (x, y) coordinates (0–1 in axes fraction space)
    ncols=2,
    fontsize=12
    )

    plt.tight_layout()
    if name:
        plt.savefig(name, dpi=100, bbox_inches='tight', facecolor='None')
        print("saved plot:", name)
    plt.show()

def plot_repeats_times_u(binn_models_dics,
                                base_colors, filename=None):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    import matplotlib.cm as cm
    import matplotlib.ticker as ticker

    hatch_styles = ['', '//', '..']

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
                    mean_95(getattr(m, metric_attr1)) * len(getattr(m, metric_attr2))
                    for m in repeats.values()
                ]
        return out

    data = collect_metric(binn_models_dics,
                          metric_attr1='epoch_times',
                          metric_attr2='train_loss_list')

    bar_width = 0.15
    intra_gap, inter_gap = 0.02, 0.10
    current_x = 0.0

    xtick_positions = []
    xtick_labels = []

    plt.figure(figsize=(7, 5))

    for i_w, (w, base_c) in enumerate(zip(widths, base_colors)):
        group_x_positions = []

        for i_d, d in enumerate(depths):
            vals = data[(w, d)]
            mean_val = np.mean(vals)
            std_val = np.std(vals)

            # Plot mean with error bar
            plt.bar(current_x,
                    mean_val,
                    yerr=std_val,
                    width=bar_width,
                    color=base_c,
                    hatch=hatch_styles[i_d],
                    edgecolor='k',
                    error_kw=dict(ecolor='gray', lw=2),
                    capsize=4)

            group_x_positions.append(current_x)
            current_x += bar_width + (intra_gap if i_d < len(depths) - 1 else inter_gap)

        # Center the xtick at the middle of the group
        center_x = np.mean(group_x_positions)
        xtick_positions.append(center_x)
        #xtick_labels.append(rf"NN$_\mathrm{{u}}$={w}")
        xtick_labels.append(f"{w}")

    # Cosmetics
    #plt.xticks(xtick_positions, xtick_labels)
    plt.xticks([], [])
    plt.ylabel(r"Total training time [s]", fontsize=14)

    # Only depth in legend
    depth_patches = [Patch(facecolor='white', edgecolor='k',
                           hatch=hatch_styles[i],
                           label=rf'NN$_\mathrm{{D}}$ = {d}') for i, d in enumerate(depths)]
    plt.legend(handles=depth_patches, loc='upper left', fontsize=16)

    # Log scale for y-axis
    plt.yscale('log')

    # Show and label minor ticks on log y-axis
    ax = plt.gca()
    ax.yaxis.set_minor_locator(ticker.LogLocator(base=10, subs='auto', numticks=100))
    ax.yaxis.set_minor_formatter(ticker.FormatStrFormatter("%.0f"))
    ax.tick_params(axis='y', which='minor', labelsize=11)
    ax.tick_params(axis='y', which='major', labelsize=12)

    plt.tight_layout()

    if filename:
        plt.savefig(f"{filename}.png", dpi=100, bbox_inches='tight', facecolor='None')
    plt.show()


def plot_repeats_times_u_control(binn_models_dics_control, 
                                binn_models_dics,
                                base_colors, filename=None):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    import matplotlib.cm as cm
    import matplotlib.ticker as ticker

    hatch_styles = ['', '//', '..']

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
                    mean_95(getattr(m_control, metric_attr1)) * len(getattr(m, metric_attr2))
                    for m_control, m in zip(repeats_control.values(), repeats.values())
                ]
        return out

    data = collect_metric(binn_models_dics_control,
                          binn_models_dics,
                          metric_attr1='epoch_times',
                          metric_attr2='train_loss_list')

    bar_width = 0.15
    intra_gap, inter_gap = 0.02, 0.10
    current_x = 0.0

    xtick_positions = []
    xtick_labels = []

    plt.figure(figsize=(7, 5))

    for i_w, (w, base_c) in enumerate(zip(widths, base_colors)):
        group_x_positions = []

        for i_d, d in enumerate(depths):
            vals = data[(w, d)]
            mean_val = np.mean(vals)
            std_val = np.std(vals)

            # Plot mean with error bar
            plt.bar(current_x,
                    mean_val,
                    yerr=std_val,
                    width=bar_width,
                    color=base_c,
                    hatch=hatch_styles[i_d],
                    edgecolor='k',
                    error_kw=dict(ecolor='gray', lw=2),
                    capsize=4)

            group_x_positions.append(current_x)
            current_x += bar_width + (intra_gap if i_d < len(depths) - 1 else inter_gap)

        # Center the xtick at the middle of the group
        center_x = np.mean(group_x_positions)
        xtick_positions.append(center_x)
        #xtick_labels.append(rf"NN$_\mathrm{{u}}$={w}")
        xtick_labels.append(f"{w}")

    # Cosmetics
    #plt.xticks(xtick_positions, xtick_labels)
    plt.xticks([], [])
    plt.ylabel(r"Total training time [s]", fontsize=14)

    # Only depth in legend
    depth_patches = [Patch(facecolor='white', edgecolor='k',
                           hatch=hatch_styles[i],
                           label=rf'NN$_\mathrm{{D}}$ = {d}') for i, d in enumerate(depths)]
    plt.legend(handles=depth_patches, loc='upper left', fontsize=16)

    # Log scale for y-axis
    plt.yscale('log')

    # Show and label minor ticks on log y-axis
    ax = plt.gca()
    ax.yaxis.set_minor_locator(ticker.LogLocator(base=10, subs='auto', numticks=100))
    ax.yaxis.set_minor_formatter(ticker.FormatStrFormatter("%.0f"))
    ax.tick_params(axis='y', which='minor', labelsize=11)
    ax.tick_params(axis='y', which='major', labelsize=12)

    plt.tight_layout()

    if filename:
        plt.savefig(f"{filename}.png", dpi=100, bbox_inches='tight', facecolor='None')
    plt.show()

def prepare_timing_data_control(dn_models_dics_c, dn_models_dics, plot_params=None):
    import numpy as np

    if plot_params is None:
        plot_params = {}

    raw_timing_data = []
    model_keys = list(dn_models_dics.keys())
    dic_values_c = list(dn_models_dics_c.values())
    dic_values = list(dn_models_dics.values())



    for i, (key, model_dic_c, model_dic) in enumerate(zip(model_keys, dic_values_c, dic_values)):
        modelWrappers = list(model_dic.values())
        modelWrappers_c = list(model_dic_c.values())

        sample_model = modelWrappers[0]
        sf = sample_model.print_freq if len(sample_model.train_loss_list)- 3 > len(sample_model.epoch_times) else 1
        # Extract individual plot parameters or use defaults
        color, linestyle, markerstyle, label = plot_params.get(
            key,
            ['C{}'.format(i), '-', '.', f'Run {i+1}']
        )
        
        epoch_times_arr = []
        for m_c in modelWrappers_c:
            arr = np.array(m_c.epoch_times, dtype=np.float64)
            p95 = np.percentile(arr, 95)
            clipped = arr[arr <= p95]
            epoch_times_arr.append(clipped)

        total_num_epochs = [len(m.epoch_times) for m in modelWrappers]

        mean_epoch_times = [np.mean(times) for times in epoch_times_arr]
        avg_epoch_time = np.mean(mean_epoch_times)
        std_epoch_time = np.std(mean_epoch_times)

        total_time_arr = [np.mean(times) * epoch_num * sf for times,epoch_num in zip(epoch_times_arr, total_num_epochs)]
        run_time_mean = np.mean(total_time_arr)
        run_time_std = np.std(total_time_arr)

        raw_timing_data.append({
            'label': label,
            'avg_epoch_time': avg_epoch_time,
            'std_epoch_time': std_epoch_time,
            'run_time_mean': run_time_mean,
            'run_time_std': run_time_std,
            'color': color
        })

    return raw_timing_data

def plot_epoch_and_total_times_control(dn_models_dics_c, dn_models_dics, plot_params=None, name=None):

    raw_timing_data = prepare_timing_data_control(dn_models_dics_c, dn_models_dics, plot_params=plot_params)

    x = np.arange(len(raw_timing_data))
    avg_times = [d['avg_epoch_time'] for d in raw_timing_data]
    std_times = [d['std_epoch_time'] for d in raw_timing_data]
    run_time_means = [d['run_time_mean'] for d in raw_timing_data]
    run_time_stds = [d['run_time_std'] for d in raw_timing_data]
    scatter_colors = [d['color'] for d in raw_timing_data]
    bar_labels = [d['label'] for d in raw_timing_data]

    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax2 = ax1.twinx()

    ax1.errorbar(x, avg_times, yerr=std_times, fmt='o', capsize=4, color='black', label='Avg Epoch Time')
    ax1.set_ylabel("Avg Epoch Time [s]", fontsize=11, color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    #ax1.set_ylim(bottom=0)

    for i in range(len(x)):
        ax2.bar(
            x[i],
            run_time_means[i],
            yerr=run_time_stds[i],
            capsize=5,
            color=scatter_colors[i],
            edgecolor='gray',
            alpha=0.4,
            error_kw=dict(ecolor="gray", alpha=1, linewidth=1.5)
        )

    ax2.set_ylabel("Total Run Time [s]", fontsize=11, color=scatter_colors[0])
    ax2.tick_params(axis='y', labelcolor=scatter_colors[0])
    #ax2.set_ylim(bottom=0)

    ax1.set_xticks(x)
    ax1.set_xticklabels(bar_labels, fontsize=11)

    fig.tight_layout()
    if name:
        plt.savefig(name, dpi=100, bbox_inches='tight', facecolor='None')
        print("saved plot:", name)
    plt.show()

def plot_running_min_val_loss_broken_x_log_lst(dn_models_dics_list,
                                           plot_params=None,
                                           plot_settings=None):
    import matplotlib.pyplot as plt
    import numpy as np

    # Default plot settings
    default_settings = {
        'xaxis': {'min': 1, 'max': 1e5, 'break': 1e3},
        'legend': {'panel': 1, 'loc': (0.05, 0.95), 'fontsize': 10, 'ncols':1},
        'name': 'running_min_val_loss_broken_xaxis_loglog.png',
        'fill': True,
        'line_lengths': {'hlength': 10000, 'vlength_factor': 2.0},
        'line_widths_on_axis': {'hwidth': 1, 'vwidth': 1},
        'line_width': 1.5,
        'fontsizes': {'xaxis':12, 'xtick_labels':10, 'yaxis':12, 'ytick_labels':10},
        'figsize':(7,5)
    }

    # Merge user-provided settings with defaults
    plot_settings = plot_settings or {}
    settings = {**default_settings, **plot_settings}
    xaxis = {**default_settings['xaxis'], **settings.get('xaxis', {})}
    legend = {**default_settings['legend'], **settings.get('legend', {})}
    name = settings.get('name', default_settings['name'])
    fill = settings.get('fill', default_settings['fill'])
    line_width = settings.get('line_width', default_settings['line_width'])
    figsize = settings.get('figsize', default_settings['figsize'])
    line_width_on_axis = {**default_settings['line_widths_on_axis'], **settings.get('line_widths_on_axis', {})}
    line_lengths = {**default_settings['line_lengths'], **settings.get('line_lengths', {})}
    fontsizes = {**default_settings['fontsizes'], **settings.get('fontsizes', {})}

    xfont = fontsizes['xaxis']
    xtick_font = fontsizes['xtick_labels']
    yfont = fontsizes['yaxis']
    ytick_font = fontsizes['ytick_labels']

    hlength = line_lengths['hlength']
    hwidth = line_width_on_axis["hwidth"]
    vwidth = line_width_on_axis["vwidth"]
    vlength_factor = line_lengths['vlength_factor']

    break_x = xaxis['break']
    x_min = xaxis['min']
    x_max = xaxis['max']

    # Style cycles
    marker_styles = ['o', 's', 'D', '^', 'v', 'P', '*', 'X']
    line_styles = ['-', '--']#, '-.', ':',':']

    run_data_all = []
    
    for i, dn_models_dics in enumerate(dn_models_dics_list):
        marker = marker_styles[i % len(marker_styles)]
        linestyle = line_styles[i % len(line_styles)]
        


        run_data, _ = prepare_model_run_data(dn_models_dics, plot_params=plot_params)

        for run in run_data:
            run['markerstyle'] = marker
            run['linestyle'] = linestyle

        run_data_all.extend(run_data)

    run_data_all.sort(key=lambda x: x['hticks'])

    

    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=figsize,
                                   gridspec_kw={'width_ratios': [1, 5], 'wspace': 0.05})

    for en, run in enumerate(run_data_all):

        x = np.where(run['epochs'] == 0, 1e-1, run['epochs'])
        y_min = run['running_min_min']
        color = run['color']
        linestyle = run.get('linestyle', '-')
        markerstyle = run.get('markerstyle', 'o')

        mask1 = x <= break_x
        mask2 = x > break_x


        if linestyle=='-':
            label = run['label']
        else:
            label = None

        ax1.plot(x[mask1], y_min[mask1], color=color, label=label, linestyle=linestyle, lw=line_width)
        ax2.plot(x[mask2], y_min[mask2], color=color, label=label, linestyle=linestyle, lw=line_width)
        
        #if en in index_track:
        #    label = f"ES{index_track}"
        #else:
        #    label =  None
        
        ax2.scatter(x[run['best_epoch_idx']], y_min[run['best_epoch_idx']], color=color, s=20, zorder=5, marker=markerstyle)

        if fill:
            y_max = run['running_min_max']
            ax1.fill_between(x[mask1], y_min[mask1], y_max[mask1], alpha=0.2, color=color)
            ax2.fill_between(x[mask2], y_min[mask2], y_max[mask2], alpha=0.2, color=color)

    for ax in [ax1, ax2]:
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_facecolor('white')

    ax2.set_xlabel("epoch (log)", fontsize=xfont)
    ax1.set_xlim(left=x_min, right=break_x)
    ax2.set_xlim(left=break_x, right=x_max)
    ax1.set_ylabel("running min val loss [a.u]", fontsize=yfont)

    ax1.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax1.tick_params(labelright=False)
    ax2.tick_params(labelleft=False)

    ax1.yaxis.set_tick_params(labelsize=ytick_font)
    ax1.xaxis.set_tick_params(labelsize=xtick_font)
    ax2.xaxis.set_tick_params(labelsize=xtick_font)

    d = .015
    kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
    ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)
    ax1.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
    kwargs.update(transform=ax2.transAxes)
    ax2.plot((-d, +d), (-d, +d), **kwargs)
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)

    if legend['panel'] == 1:
        ax1.legend(fontsize=legend['fontsize'], bbox_to_anchor=legend['loc'], ncols=legend['ncols'])
    else:
        ax2.legend(fontsize=legend['fontsize'], bbox_to_anchor=legend['loc'], ncols=legend['ncols'])

    ax2_right = ax2.twinx()
    y_min_axis, y_max_axis = ax1.get_ylim()
    y_top = y_min_axis * vlength_factor
    ax2_right.tick_params(axis='y', which='both', left=False, right=False, labelleft=False, labelright=False)

    for en, run in enumerate(run_data_all):
        linestyle = run.get('linestyle', '-')
        color = run['color']

        
        if linestyle=='-':
            label = run['label']

            ax2.hlines(
                y=run['hticks'],
                xmin=x_max - hlength,
                xmax=x_max,
                color=color,
                linestyle=linestyle,
                linewidth=hwidth,
                zorder=10
            )

            best_epochs_lst = run['best_epochs_lst']
            to_plot = np.min(best_epochs_lst)

            if to_plot < break_x:
                ax1.vlines(to_plot, y_min_axis, y_top, color=color, linestyle=linestyle, lw=vwidth)
            ax2.vlines(to_plot, y_min_axis, y_top, color=color, linestyle=linestyle, lw=vwidth)

    ax2.set_ylim(y_min_axis)
    fig.tight_layout()

    if name:
        plt.savefig(name, dpi=100, bbox_inches='tight', facecolor='None')
        print("saved plot:", name)
    plt.show()

    return {
        'name': name,
        'legend': legend,
        'fill': fill,
        'xaxis': xaxis,
        'num_runs': len(run_data_all)
    }
def plot_epoch_and_total_times_lst(dn_models_dics_list, label_list=None, plot_params=None, name=None,
                                   hatch_patterns = ['', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*'],
                                   legend_col="#0033A0", legend_pos=(0.05, 0.95), legend_fontsize=16,
                                   legend_ncols=1,
                                   figsize=(7,5)):
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib.cm as cm
    from matplotlib.patches import Patch

    if label_list is None:
        label_list = [f'Group {i+1}' for i in range(len(dn_models_dics_list))]


    # Collect all data
    grouped_data = []
    all_labels_set = set()

    for dn_models_dics in dn_models_dics_list:
        _, raw_data = prepare_model_run_data(dn_models_dics, plot_params=plot_params)
        label_to_data = {d['label']: d for d in raw_data}
        grouped_data.append(label_to_data)
        all_labels_set.update(label_to_data.keys())

    all_labels = sorted(list(all_labels_set), key=float)
    num_groups = len(dn_models_dics_list)
    num_bars = len(all_labels)
    bar_width = 0.8 / num_groups
    
    # User-defined space between bars in the same group (as a fraction of total width)
    intra_group_spacing = 0.05  # try 0.05–0.15 for visible spacing

    num_groups = len(dn_models_dics_list)
    num_bars = len(all_labels)

    total_group_width = 0.8  # how much of x-axis to cover per group (per label)
    total_spacing = intra_group_spacing * (num_groups - 1)
    bar_width = (total_group_width - total_spacing) / num_groups

    # Offsets within group (centered)
    group_offsets = np.linspace(
        -total_group_width / 2 + bar_width / 2,
        total_group_width / 2 - bar_width / 2,
        num_groups
    )

    fig, ax1 = plt.subplots(figsize=figsize)
    ax2 = ax1.twinx()  # ax2 is now for epoch time

    # Plot bars and epoch times
    for i, label in enumerate(all_labels):
        for j, group_dict in enumerate(grouped_data):
            if label not in group_dict:
                continue

            d = group_dict[label]
            x_pos = i + group_offsets[j]

            # Total run time as bar (now on ax1)
            ax1.bar(
                x_pos,
                d['run_time_mean'],
                yerr=d['run_time_std'],
                width=bar_width,
                capsize=5,
                color=d.get('color', 'gray'),
                edgecolor='gray', #k
                hatch=hatch_patterns[j % len(hatch_patterns)],
                alpha=0.5,
                error_kw=dict(ecolor="gray", alpha=1, linewidth=1.5),
                label=None  # legend handled separately
            )

            # Avg epoch time as errorbar (now on ax2)
            ax2.errorbar(
                x_pos,
                d['avg_epoch_time'],
                yerr=d['std_epoch_time'],
                fmt='o',
                capsize=8,
                color='black',
                markersize=8,
                label='Avg Epoch Time' if i == j == 0 else None
            )

    # Custom legend for hatch patterns
    custom_handles = [
        Patch(
            facecolor='lightgray',
            edgecolor='gray',
            hatch=hatch_patterns[i % len(hatch_patterns)],
            label=label_list[i]
        ) for i in range(num_groups)
    ]

    # Combine legends
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
    custom_handles + handles2,
    label_list,# + labels2,
   # loc='upper left',          # anchor point inside the legend
    bbox_to_anchor=legend_pos,  # coordinates (x, y) relative to the axes
    fontsize=legend_fontsize,
    ncols=legend_ncols
)

    #ax1.legend(custom_handles + handles2, label_list + labels2, loc='upper left', fontsize=14)

    # Labels and ticks
    d = group_dict[all_labels[0]]
    ax1.set_ylabel("Total Run Time [s]", fontsize=14, color=legend_col)
    ax2.set_ylabel("Avg Epoch Time [s]", fontsize=14, color='black')

    ax1.set_yscale('log')
    #ax1.grid(True, which='major',  axis='y', linestyle='-', linewidth=0.75, alpha=0.8)
    #ax1.grid(True, which='minor', linestyle='--', linewidth=0.5, alpha=0.4)
    ax1.tick_params(axis='y', which='major', length=10)   # major y ticks
    ax1.tick_params(axis='y', which='minor', length=5)   # major y ticks
    ax1.tick_params(axis='y', labelcolor=legend_col, labelsize=14)
    ax2.tick_params(axis='y', labelcolor='black', labelsize=14)

    ax1.set_xticks(np.arange(num_bars))
    ax1.set_xticklabels(all_labels, rotation=45, ha='right', fontsize=14)

    fig.tight_layout()
    if name:
        plt.savefig(name, dpi=100, bbox_inches='tight', facecolor='None')
        print("saved plot:", name)
    plt.show()

def prepare_diffusion_mse_data(dn_models_dics, plot_params=None):
    import numpy as np

    if plot_params is None:
        plot_params = {}

    mse_data = []
    model_keys = list(dn_models_dics.keys())
    dic_values = list(dn_models_dics.values())



    for i, (key, model_dic) in enumerate(zip(model_keys, dic_values)):
        modelWrappers = list(model_dic.values())

        sample_model = modelWrappers[0]
        sf = sample_model.print_freq if len(sample_model.train_loss_list)- 3 > len(sample_model.epoch_times) else 1
        # Extract individual plot parameters or use defaults
        color, linestyle, markerstyle, label = plot_params.get(
            key,
            ['C{}'.format(i), '-', '.', f'Run {i+1}']
        )


        final_mses = [m.best_diffusion_error for m in modelWrappers]

        mean_mses = [np.mean(final_mse) for final_mse in final_mses]
        avg_mse = np.mean(mean_mses)
        std_mse = np.std(mean_mses)

        mse_data.append({
            'label': label,
            'avg_mse': avg_mse,
            'std_mse': std_mse,
            'color': color
        })

    return mse_data


def plot_diff_mse_lst(dn_models_dics_list, label_list=None, plot_params=None, name=None):
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib.cm as cm
    from matplotlib.patches import Patch

    if label_list is None:
        label_list = [f'Group {i+1}' for i in range(len(dn_models_dics_list))]

    # Hatching patterns for textures
    hatch_patterns = ['', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*']

    # Collect all data
    grouped_data = []
    all_labels_set = set()

    for dn_models_dics in dn_models_dics_list:
        mse_data = prepare_diffusion_mse_data(dn_models_dics, plot_params=plot_params)
        label_to_data = {d['label']: d for d in mse_data}
        grouped_data.append(label_to_data)
        all_labels_set.update(label_to_data.keys())

    all_labels = sorted(list(all_labels_set), key=float)
    num_groups = len(dn_models_dics_list)
    num_bars = len(all_labels)
    bar_width = 0.8 / num_groups
    group_offsets = np.linspace(-0.4 + bar_width / 2, 0.4 - bar_width / 2, num_groups)

    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Plot bars and epoch times
    for i, label in enumerate(all_labels):
        for j, group_dict in enumerate(grouped_data):
            if label not in group_dict:
                continue

            d = group_dict[label]
            x_pos = i + group_offsets[j]


            # Avg epoch time as errorbar (now on ax2)


            ax1.bar(
                x_pos,
                d['avg_mse'],
                yerr=d['std_mse'],
                width=bar_width,
                capsize=5,
                color=d.get('color', 'gray'),
                edgecolor='gray',
                hatch=hatch_patterns[j % len(hatch_patterns)],
                alpha=0.5,
                error_kw=dict(ecolor="gray", alpha=1, linewidth=1.5),
                label=None  # legend handled separately
            )
    # Custom legend for hatch patterns
    custom_handles = [
        Patch(
            facecolor='lightgray',
            edgecolor='gray',
            hatch=hatch_patterns[i % len(hatch_patterns)],
            label=label_list[i]
        ) for i in range(num_groups)
    ]

    # Combine legends
    ax1.legend(custom_handles, label_list,
           loc='center',                   # anchor point within legend box
           bbox_to_anchor=(0.25, 0.85),     # (x, y) in axes fraction coords
           fontsize=16)


    # Labels and ticks
    d = group_dict[all_labels[0]]
    ax1.set_ylabel("Diffusion MSE [day$^{-2}$ mm$^{4}$]", fontsize=14, color='k')


    ax1.tick_params(axis='y', labelcolor='k', labelsize=14)


    ax1.set_xticks(np.arange(num_bars))
    ax1.set_xticklabels(all_labels, rotation=45, ha='right', fontsize=14)

    ax1.set_yscale('log')

    fig.tight_layout()
    if name:
        plt.savefig(name, dpi=100, bbox_inches='tight', facecolor='None')
        print("saved plot:", name)
    plt.show()
    

def plot_running_min_MSE_diff_loss_broken_x_log_lst(dn_models_dics_list, 
                                                plot_params=None,
                                                plot_settings=None):
    import matplotlib.pyplot as plt
    import numpy as np

    default_settings = {
        'xaxis': {'min': 1, 'max': 1e5, 'break': 1e3},
        'legend': {'panel': 1, 'loc': 'lower left', 'fontsize': 10, 'ncols':1},
        'name': 'running_min_diff_loss_broken_xaxis_loglog.png',
        'fill': True,
        'line_lengths': {'hlength': 10000, 'vlength_factor': 2.0},
        'line_width': 1.5,
        'markersize': 4,
        'line_widths_on_axis': {'hwidth': 1, 'vwidth': 1},
        'fontsizes': {'xaxis': 12, 'xtick_labels': 10, 'yaxis': 12, 'ytick_labels': 10}
    }

    plot_settings = plot_settings or {}
    settings = {**default_settings, **plot_settings}
    xaxis = {**default_settings['xaxis'], **settings.get('xaxis', {})}
    legend = {**default_settings['legend'], **settings.get('legend', {})}
    name = settings.get('name', default_settings['name'])
    fill = settings.get('fill', default_settings['fill'])
    line_width = settings.get('line_width', default_settings['line_width'])
    line_lengths = {**default_settings['line_lengths'], **settings.get('line_lengths', {})}
    hlength = line_lengths['hlength']
    vlength_factor = line_lengths['vlength_factor']
    markersize = settings.get('markersize', 4)
    line_width_on_axis = {**default_settings['line_widths_on_axis'], **settings.get('line_widths_on_axis', {})}
    fontsizes = {**default_settings['fontsizes'], **settings.get('fontsizes', {})}

    xfont = fontsizes['xaxis']
    xtick_font = fontsizes['xtick_labels']
    yfont = fontsizes['yaxis']
    ytick_font = fontsizes['ytick_labels']
    hwidth = line_width_on_axis["hwidth"]
    vwidth = line_width_on_axis["vwidth"]

    break_x = xaxis['break']
    x_min = xaxis['min']
    x_max = xaxis['max']

    # Define marker/linestyle cycles
    marker_styles = ['o', 's', 'D', '^', 'v', 'P', '*', 'X']
    line_styles = ['-', '--', '-.', ':']

    # Gather and tag run data
    all_run_data = []
    for i, dn_models_dics in enumerate(dn_models_dics_list):
        group_runs = prepare_diff_run_data(dn_models_dics, plot_params)
        marker = marker_styles[i % len(marker_styles)]
        linestyle = line_styles[i % len(line_styles)]
        for run in group_runs:
            run['markerstyle'] = marker
            run['linestyle'] = linestyle
        all_run_data.extend(group_runs)

    all_run_data.sort(key=lambda x: x['hticks'])

    # Sample model to get print_freq
    sample_model = list(list(dn_models_dics_list[0].values())[0].values())[0]
    sf = sample_model.print_freq if len(sample_model.train_loss_list) > len(sample_model.epoch_times) else 1

    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(8, 5),
                                   gridspec_kw={'width_ratios': [1, 5], 'wspace': 0.05})

    for run in all_run_data:
        x = np.where(run['epochs'] * sf == 0, 1e-1, run['epochs'] * sf)
        y_mean = run['mean_vals']
        y_min = run['min_vals']
        y_max = run['max_vals']
        color = run['color']
        markerstyle = run.get('markerstyle', '-')
        linestyle = run.get('linestyle', '-')

        mask1 = x <= break_x
        mask2 = x > break_x

        if linestyle == '-':
            label=run['label']
        else:
            label=None

        ax1.plot(x[mask1], y_mean[mask1], color=color, label=label,
                 lw=line_width, linestyle=linestyle, marker=markerstyle, markersize=markersize)
        ax2.plot(x[mask2], y_mean[mask2], color=color, label=label,
                 lw=line_width, linestyle=linestyle, marker=markerstyle, markersize=markersize)

        if fill:
            ax1.fill_between(x[mask1], y_min[mask1], y_max[mask1], alpha=0.2, color=color)
            ax2.fill_between(x[mask2], y_min[mask2], y_max[mask2], alpha=0.2, color=color)

    # Axis formatting
    for ax in [ax1, ax2]:
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_facecolor('white')

    ax1.yaxis.set_tick_params(labelsize=ytick_font)
    ax1.xaxis.set_tick_params(labelsize=xtick_font)
    ax2.xaxis.set_tick_params(labelsize=xtick_font)

    ax2.set_xlabel("epoch (log)", fontsize=xfont)
    ax1.set_xlim(left=x_min, right=break_x)
    ax2.set_xlim(left=break_x, right=x_max)
    ax1.set_ylabel(r"Diffusion MSE [mm$^4$ day$^{-2}$]", fontsize=yfont)

    # Axis break visuals
    ax1.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax1.tick_params(labelright=False)
    ax2.tick_params(labelleft=False)

    d = .015
    kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
    ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)
    ax1.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
    kwargs.update(transform=ax2.transAxes)
    ax2.plot((-d, +d), (-d, +d), **kwargs)
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)

    # Legend
    if legend['panel'] == 1:
        ax1.legend(fontsize=legend['fontsize'], loc=legend['loc'], ncols=legend['ncols'])
    else:
        ax2.legend(fontsize=legend['fontsize'], loc=legend['loc'], ncols=legend['ncols'])

    ax2_right = ax2.twinx()
    y_min_plot, y_max_plot = ax1.get_ylim()
    y_top = y_min_plot * vlength_factor
    ax2_right.set_yscale('log')
    ax2_right.set_ylim(ax2.get_ylim())
    ax2_right.tick_params(axis='y', which='both', left=False, right=False, labelleft=False, labelright=False)

    # Highlight vertical bars for best epoch and final value
    for run in all_run_data:
        linestyle = run.get('linestyle', '-')
        color = run['color']

        if linestyle == '-':
            best_epochs_lst = run['best_epochs_lst']
            to_plot = np.min(best_epochs_lst)

            # Vertical line for best epoch
            if to_plot < break_x:
                ax1.vlines(to_plot, y_min_plot, y_top, color=color, linestyle=linestyle, lw=vwidth)
            ax2.vlines(to_plot, y_min_plot, y_top, color=color, linestyle=linestyle, lw=vwidth)

            # Horizontal line at final performance
            ax2.hlines(run['hticks'], x_max - hlength, x_max, color=color, linestyle=linestyle, linewidth=hwidth, zorder=10)
        if linestyle == '-' or linestyle == '--':
            # Vertical line at final x position
            x_final = run['epochs'][-1] * sf
            if x_final <= break_x:
                ax1.vlines(x_final, y_min_plot, y_max_plot, color=color, linestyle=linestyle, lw=1, alpha=0.75)
            else:
                ax2.vlines(x_final, y_min_plot, y_max_plot, color=color, linestyle=linestyle, lw=1, alpha=0.75)

    ax2.set_ylim(y_min_plot)
    fig.tight_layout()

    if name:
        plt.savefig(name, dpi=100, bbox_inches='tight', facecolor='None')
        print("saved plot:", name)
    plt.show()


def plot_final_loss_lst(dn_models_dics_list, label_list=None, plot_params=None, name=None,
                        bbox_to_anchor=(0.5,0.5), legend_size=16,
                        y_label="Validation loss [a.u.]",
                        hatch_patterns = ['', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*'],
                        figsize=(10, 5)):
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib.cm as cm
    from matplotlib.patches import Patch

    if label_list is None:
        label_list = [f'Group {i+1}' for i in range(len(dn_models_dics_list))]

    # Hatching patterns for textures
    

    # Collect all data
    grouped_data = []
    all_labels_set = set()

    for dn_models_dics in dn_models_dics_list:
        run_data, _ = prepare_model_run_data(dn_models_dics, plot_params=plot_params)
        label_to_data = {d['label']: d for d in run_data}
        grouped_data.append(label_to_data)
        all_labels_set.update(label_to_data.keys())

    all_labels = sorted(list(all_labels_set), key=float)
    num_groups = len(dn_models_dics_list)
    num_bars = len(all_labels)
    bar_width = 0.8 / num_groups
    group_offsets = np.linspace(-0.4 + bar_width / 2, 0.4 - bar_width / 2, num_groups)

    fig, ax1 = plt.subplots(figsize=figsize)

    # Plot bars and epoch times
    for i, label in enumerate(all_labels):
        for j, group_dict in enumerate(grouped_data):
            if label not in group_dict:
                continue

            d = group_dict[label]
            x_pos = i + group_offsets[j]

            # Total run time as bar (now on ax1)
            ax1.bar(
                x_pos,
                d['mean_final_loss'],
                yerr=d['std_final_loss'],
                width=bar_width,
                capsize=5,
                color=d.get('color', 'gray'),
                edgecolor='gray',
                hatch=hatch_patterns[j % len(hatch_patterns)],
                alpha=1,
                error_kw=dict(ecolor="gray", alpha=1, linewidth=1.5),
                label=None  # legend handled separately
            )


    # Custom legend for hatch patterns
    custom_handles = [
        Patch(
            facecolor='lightgray',
            edgecolor='gray',
            hatch=hatch_patterns[i % len(hatch_patterns)],
            label=label_list[i]
        ) for i in range(num_groups)
    ]

    # Combine legends
    ax1.legend(custom_handles, label_list, loc='upper left', bbox_to_anchor=bbox_to_anchor, fontsize=legend_size)

    

    # Labels and ticks
    d = group_dict[all_labels[0]]
    ax1.set_ylabel(y_label, fontsize=14)

    ax1.set_yscale('log')
    #ax1.grid(True, which='major',  axis='y', linestyle='-', linewidth=0.75, alpha=0.8)
    #ax1.grid(True, which='minor', linestyle='--', linewidth=0.5, alpha=0.4)
    ax1.tick_params(axis='y', which='major', length=10)   # major y ticks
    ax1.tick_params(axis='y', which='minor', length=5)   # major y ticks
    ax1.tick_params(axis='y',  labelsize=14)

    ax1.set_xticks(np.arange(num_bars))
    ax1.set_xticklabels(all_labels, rotation=45, ha='right', fontsize=14)

    fig.tight_layout()
    if name:
        plt.savefig(name, dpi=100, bbox_inches='tight', facecolor='None')
        print("saved plot:", name)
    plt.show()