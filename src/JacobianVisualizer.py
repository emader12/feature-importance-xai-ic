
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize,TwoSlopeNorm,LinearSegmentedColormap
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, peak_prominences


def heatmap_vs_XAI(jacobian_matrix,
                   wrt,
                   wl,
                   dataset,
                   shap_spectrum,
                   const_dict={'temperature': 800, 'gravity': 4.0, 'c_o_ratio': 0.5, 'metallicity': 1},
                   parameters=np.array(['temperature', 'gravity', 'c_o_ratio', 'metallicity']),
                   __save__ = False,
                   save_path = None,
                  ):
    
    # create list of constant names
    const_params = [p for p in parameters if p != wrt]
    p1,p2,p3 = const_params

    # create parameter grid of wrt variable
    wrt_grid = np.sort(dataset[wrt].unique())[1:-1]

    # ideces of params
    wrt_id = np.where(parameters == wrt)[0][0]
    other_ids = [i for i in range(4) if i != wrt_id]

    # transpose to make wrt axis 0
    new_order = [wrt_id] + other_ids + [4]
    jacob_reordered = np.transpose(jacobian_matrix, new_order)

    # create parameter grids of const variables 
    const_grids = {p: np.sort(dataset[p].unique()) for p in const_params}
    
    id_p1 = np.where(const_grids[p1] == const_dict[p1])[0][0]
    id_p2 = np.where(const_grids[p2] == const_dict[p2])[0][0]
    id_p3 = np.where(const_grids[p3] == const_dict[p3])[0][0]

    # slice
    jacob_slice = jacob_reordered[:, id_p1, id_p2, id_p3, :]  # shape (N_wrt, N_lambda)

    # create figure
    fig, axes = plt.subplots(2,1,figsize=(10,8),gridspec_kw={'height_ratios': [2, 1]})
    ax1, ax2 = axes
    
    # manually skip redundant models 
    if wrt == 'metallicity':
        skip_vals = np.array([-0.7, -0.5, -0.3, 0.5])
        id_skip = np.where(np.isin(wrt_grid, skip_vals))[0]
        jacob_slice = np.delete(jacob_slice, id_skip, axis=0)
        wrt_grid = np.delete(wrt_grid, id_skip)

    # plot heatmap
    X, Y = np.meshgrid(wl, wrt_grid) # create meshgrid 
    
    global_slice_min = jacob_slice.min()
    global_slice_max = jacob_slice.max()
    
    if (global_slice_min<0) & (global_slice_max>0): # neg and pos values
        abs_slice = abs(jacob_slice)
        row_max = np.max(abs_slice, axis=1, keepdims=True)
        norm_slice = jacob_slice / row_max

        norm = TwoSlopeNorm(vmin=-1,vcenter=0,vmax=1)
        colors = [(0.0,"navy"),
                  (0.25,"skyblue"),
                  (0.5,"white"),
                  (0.66,"orange"),
                  (0.83,"red"),
                  (1.0,"darkred")] 
    elif (global_slice_min<0) & (global_slice_max<=0): # all neg values
        abs_slice = np.abs(jacob_slice)
        min_ = np.min(abs_slice, axis=1, keepdims=True)
        max_ = np.max(abs_slice, axis=1, keepdims=True)
        norm_slice = (abs_slice - min_) / (max_ - min_)
        norm = TwoSlopeNorm(vmin=0,
            vcenter=0.5,
            vmax=1)
        colors = [(0.0,"white"),
                  (0.5,"skyblue"),
                  (1.0,"navy")] 
    else: # all pos values
        slice_min = np.min(jacob_slice, axis=1, keepdims=True)  
        slice_max = np.max(jacob_slice, axis=1, keepdims=True)
        norm_slice = (jacob_slice - slice_min) / (slice_max - slice_min)
        norm = TwoSlopeNorm(vmin=0,
            vcenter=0.5,
            vmax=1)
        colors = [(0.0,"white"),
                  (0.33,"orange"),
                  (0.66,"red"),
                  (1.0,"darkred")] 
        
    cmap = LinearSegmentedColormap.from_list("custom",colors)
    im = ax1.pcolormesh(X, Y, norm_slice, cmap=cmap, norm=norm, shading='auto')


    # label dictionary
    labels_dict = {'temperature': ['Temperature (K)','T'],
                  'gravity': ['log$g$ (dex)','g'],
                  'c_o_ratio': ['C/O Ratio','CO'],
                  'metallicity': ['Metallicity [M/H] (dex)','MH']}
    
    # index-based title label
    label_parts = [
        f"{labels_dict[p1][1]} = {const_dict[p1]}",
        f"{labels_dict[p2][1]} = {const_dict[p2]}",
        f"{labels_dict[p3][1]} = {const_dict[p3]}"]

    cbar = fig.colorbar(im,ax=axes)
    if (global_slice_min<0) & (global_slice_max<=0):
        cbar.set_ticks(np.arange(0,1.1,0.2))
        cbar.set_ticklabels(np.round(-1*np.arange(0,1.1,0.2),1))
        cbar.ax.invert_yaxis()
    elif (global_slice_min<0) & (global_slice_max>0):
        cbar.set_ticks(np.arange(-1,1.1,0.2))

    cbar.set_label(f'Normalized ∂F/∂{labels_dict[wrt][1]}', fontsize=13, rotation=270,labelpad=20)

    # plot SHAP 
    mean_abs_shap = np.mean(np.abs(shap_spectrum), axis=0)
    norm_shap = (mean_abs_shap-mean_abs_shap.min()) / (mean_abs_shap.max()-mean_abs_shap.min())
    ax2.plot(wl,norm_shap,color='k',label='Mean Abs. SHAP',lw=1.7,zorder=1000)

    # plot vlines at peaks
    peaks, _ = find_peaks(norm_shap)
    proms = peak_prominences(norm_shap, peaks)[0]
    # get top 3 most prominent peaks
    top3_idx = np.argsort(proms)[-3:]
    top3_peaks = peaks[top3_idx]

    for ax in (ax1,ax2):
        for peak in top3_peaks:
            ax.axvline(wl[peak], ymin=norm_shap.min(), ymax=norm_shap.max(), color='k', lw=1.7,linestyle='--')


    # xticks, yticks
    if wrt == 'temperature':
        yticks = wrt_grid
        ax1.set_yticks(yticks)
        labels = [str(int(y)) if y in np.arange(200, 2401, 200) else '' for y in yticks]
        ax1.set_yticklabels(labels)
    else:
        ax1.set_yticks(wrt_grid)

    # axes labels
    ax1.set_xlabel('Wavelength (μm)',fontsize=13)
    ax2.set_xlabel('Wavelength (μm)',fontsize=13)
    
    ax2.legend()
    ax2.set_ylabel("XAI Feature Importance",fontsize=13)
    ax1.set_ylabel(labels_dict[wrt][0],fontsize=13)
    ax1.set_title(f"Jacobian: ∂F/∂{labels_dict[wrt][1]} ({','.join(label_parts)})",fontsize=13)

    for ax in (ax1,ax2):
        # xlim
        ax.set_xlim(0.9,2.5)
        # tickmarks
        ax.tick_params(axis='both', which='major', direction='in', length=5, width=1,labelsize=12)
        ax.tick_params(axis='x', direction='in', top=True)
        ax.tick_params(axis='y', direction='in', right=True)
        ax.xaxis.minorticks_on()
        ax.tick_params(axis='x', which='minor', direction='in', length=3, width=1)
        ax.tick_params(axis='x', which='minor', direction='in', top=True)
    
    if __save__:
        save_path = f'{wrt}_heatmap_XAI_plot.pdf' if save_path is None else save_path
        plt.savefig(save_path,bbox_inches='tight')
    else:
        plt.show()



def compare_IC_ML(jacobian_matrix,
                  wrt,
                  wavelengths,
                  gp_plot,
                  dataset,
                  shap_spectrum,
                  const_dict = {'temperature':800,
                                'gravity':4.0,
                                'c_o_ratio':0.5,
                                'metallicity':1},
                  parameters = np.array(['temperature','gravity','c_o_ratio','metallicity']),
                  plot_LIME=True,
                  plot_SHAP=True,
                  cmap = plt.cm.plasma,
                  reverse_cmap = False,
                  reverse_zorder = True,
                  alpha = 0.2,
                  plot_jacob_lines = True,
                  line_alpha=0.8,
                  __save__=False,
                  save_path=None,
                 ):
              
    param_grid = dataset[parameters]
    wrt_grid = np.sort(param_grid[wrt].unique())[1:-1]
    const_params = [p for p in parameters if p != wrt]
    p1,p2,p3 = const_params
    const_grids = {p: np.sort(param_grid[p].unique())[1:-1] for p in const_params}

    # slice jacobian
    num_wrt = np.where(parameters==wrt)[0][0]
    num_others = [i for i in range(4) if i != num_wrt]
    new_order = [num_wrt] + num_others + [4] 
    jacob_reordered = np.transpose(jacobian_matrix, new_order)

    # find slices
    const_indices = []
    for p in const_params:
        grid_vals = const_grids[p]
        const_idx = np.where(grid_vals == const_dict[p])[0][0]
        const_indices.append(const_idx)

    # slice
    id_p1, id_p2, id_p3 = const_indices    
    jacob_slice = jacob_reordered[:, id_p1, id_p2, id_p3, :]       

    id_gp_plot = [np.where(wrt_grid == i)[0][0] for i in gp_plot]

    # create figure
    fig, ax = plt.subplots(figsize=(8, 5))
    
    if reverse_cmap:
        cmap = cmap.reversed()
    else:
        cmap = cmap

    for i, ind in enumerate(id_gp_plot):
        color = cmap(i / (len(id_gp_plot) - 1))
        y_data = abs(jacob_slice[ind])

        denom = np.ptp(y_data)
        if denom == 0:
            y_normalized = np.zeros_like(y_data)
        else:
            y_normalized = (y_data - y_data.min()) / denom
    
        # add fill_between from 0 to line
        zorder = len(id_gp_plot) - i if reverse_zorder else 1
        ax.fill_between(wavelengths, 0, y_normalized, 
                        color=color, 
                        alpha=alpha, 
                        zorder=zorder
                        )
        if plot_jacob_lines: 
            ax.plot(wavelengths, y_normalized, 
                    color=color, 
                    alpha=line_alpha, 
                    zorder=100
                    )
    if plot_SHAP:    
        # plot SHAP 
        mean_abs_shap = np.mean(np.abs(shap_spectrum), axis=0)
        norm_shap = (mean_abs_shap-mean_abs_shap.min()) / (mean_abs_shap.max()-mean_abs_shap.min())
        ax.plot(wavelengths,norm_shap,color='k',label='Mean Abs. SHAP',lw=1.7,zorder=1000)

    labels_dict = {'temperature': ['Temperature (K)','T','t'],
              'gravity': ['log$g$ (dex)','g','g'],
              'c_o_ratio': ['C/O Ratio','C/O','co'],
              'metallicity': ['Metallicity [M/H] (dex)','[M/H]','met']}
    label_parts = [
        f"{labels_dict[p1][1]} = {const_dict[p1]}",
        f"{labels_dict[p2][1]} = {const_dict[p2]}",
        f"{labels_dict[p3][1]} = {const_dict[p3]}"]
    
    # create colorbar
    sm = ScalarMappable(cmap=cmap, norm=Normalize(vmin=gp_plot.min(), vmax=gp_plot.max()))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax,pad=0.02)
    cbar.set_label(labels_dict[wrt][0], rotation=270, labelpad=20, fontsize=14)
    cbar.ax.tick_params(axis='both', direction='in',length=4, width=1,labelsize=13)

    ax.set_xlim(0.9,2.5)
    ax.set_ylim(0,1.1)
    
    ax.legend(loc='upper left',fontsize=13,frameon=False)
    
    ax.set_xlabel('Wavelength (μm)', fontsize=14)
    ax.set_ylabel(f'Normalized ∂F/∂{labels_dict[wrt][1]}', fontsize=14)

    ax.text(0.97,0.97,f"[{','.join(label_parts)}]",fontsize=13, ha='right',va='top',transform = ax.transAxes)
    
    ax.tick_params(axis='both', which='major', direction='in', length=6, width=1,labelsize=13)
    ax.tick_params(axis='x', direction='in', top=True)
    ax.tick_params(axis='y', direction='in', right=True)
    ax.xaxis.minorticks_on()
    ax.tick_params(axis='both', which='minor', direction='in', length=3.5, width=1)
    ax.tick_params(axis='x', which='minor', direction='in', top=True)
    ax.tick_params(axis='y', which='minor', direction='in', right=True)

    if __save__:
        gp_string = f"{labels_dict[p1][2]}{const_dict[p1]}{labels_dict[p2][2]}{const_dict[p2]}{labels_dict[p3][2]}{const_dict[p3]}"
        
        save_path = f'IC_ML_{wrt}_{gp_string}.pdf'if save_path is None else save_path
        plt.savefig(save_path,bbox_inches='tight')
    else:
        plt.show()


def plot_normalized_slice(matrices_dict,
                          dataset, 
                          wl,
                          parameters = [
                              'temperature',
                              'gravity',
                              'c_o_ratio',
                              'metallicity'],
                          const_dict = {
                              'temperature':800,
                              'gravity':4.0,
                              'c_o_ratio':0.5,
                              'metallicity':1},
                          abs_value = True,
                          smoothing = True
                          ):
    
    parameters = np.array(parameters)
    param_cols = dataset[parameters]
    params_grids = {p: np.sort(param_cols[p].unique()) for p in parameters}


    # Get index for each parameter from const_dict
    id_vals = {}
    for p in parameters:
        grid = params_grids[p]
        val = const_dict[p]
        id_vals[p] = np.where(grid == val)[0][0]

    fig, axes = plt.subplots(2,1,figsize=(10,8))
    ax1,ax2=axes

    slices = []
    for i, param in enumerate(parameters):

        jacobian = matrices_dict[param]
        idx = (
            id_vals['temperature'],
            id_vals['gravity'],
            id_vals['c_o_ratio'],
            id_vals['metallicity']
        )
        jacobian_slice = jacobian[idx]

         # Normalize
        if abs_value:
            abs_slice = np.abs(jacobian_slice)
        else:
            abs_slice = jacobian_slice
        min_ = np.min(abs_slice)
        max_ = np.max(abs_slice)

        if max_ - min_ != 0:
            normalized = (abs_slice - min_) / (max_ - min_)
        else:
            normalized = abs_slice  # all values are the same

        slices += [normalized]

        ax1.plot(wl, normalized,
                color=['#6a6ff5', '#bb6cd7', '#52d265', '#ff7047'][i],
                label=param,
                alpha=0.9)


    mean_slice = np.mean(np.array(slices),axis=0)

    ax1.plot(wl, mean_slice,color='k',label='mean',ls='--')

    for i, param in enumerate(parameters):
        
        
        diff = slices[i]-mean_slice
        
        if smoothing:
            diff = gaussian_filter1d(diff, sigma=2)

        ax2.plot(wl, diff,
                color=['#6a6ff5', '#bb6cd7', '#52d265', '#ff7047'][i],
                label=param,
                alpha=0.9)

    


    ax2.axhline(y=0,color='k',label='mean',ls='--')

    ax1.grid(True, axis='both',alpha=0.6,ls='--')
    ax2.grid(True, axis='both',alpha=0.6,ls='--')
    ax1.legend(loc='center left',bbox_to_anchor=(1.01, 0.5))
    ax2.legend(loc='center left',bbox_to_anchor=(1.01, 0.5))

    ax1.set_xlabel('Wavelength (μm)',fontsize=13)
    ax2.set_xlabel('Wavelength (μm)',fontsize=13)

    ax1.set_ylabel('Normalized ∂F/∂[T,g,C/O,M/H]',fontsize=13)
    ax2.set_ylabel('Standardized ∂F/∂[T,g,C/O,M/H]',fontsize=13)

    ax1.set_title(
        f'T= {const_dict["temperature"]}, g={const_dict["gravity"]}, C/O = {const_dict["c_o_ratio"]}, M/H = {const_dict["metallicity"]}',
        fontsize=13)

    
    for ax in (ax1,ax2):

        ax.tick_params(axis='both', which='major', direction='in', length=5, width=1,labelsize=12)
        ax.tick_params(axis='x', direction='in', top=True)
        ax.tick_params(axis='y', direction='in', right=True)
        ax.minorticks_on()
        ax.tick_params(axis='both', which='minor', direction='in', length=3, width=1)
        ax.tick_params(axis='x', which='minor', direction='in', top=True)
        ax.tick_params(axis='y', which='minor', direction='in', right=True)

    plt.subplots_adjust(right=0.8)
    plt.show()







def delta_flux_delta_J(jacob_matrix,
                        wrt,
                        dataset, 
                        wl,
                        parameters = ['temperature',
                                    'gravity',
                                    'c_o_ratio',
                                    'metallicity'],
                        const_dict = {'temperature':800,
                                        'gravity':4.0,
                                        'c_o_ratio':0.5,
                                        'metallicity':1},
                        logscale=True,
                        __save__=False,
                        save_path=None,
                        ):

    """
    Plot a heatmap of the jacobian of the model grid at a
    chosen grid point. 

    jacob_matrix: np.ndarray
        Jacobian matrix. 
    dataset: pd.DataFrame
        Dataset with columns corresponding to parameter grid 
        points and synthetic fluxes over the wavelength grid. 
    wrt: string
        Parameter to compute Jacobian matrix with respect to.
    wrt_value: float
        Value to slice jacobian. 
    wl: np.ndarray  
        Wavelength grid of the spectra. 
    parameters: list
        List of the names of parameters in the parameter space. 
    const_dict: dict
        Dictionary of parameters to hold constant. 
    __save__ : bool
        Whether or not to save the plot. 
    save_path : string
        Path to save the plot. 
    """

    parameters = np.array(parameters)
    wrt_value = const_dict[wrt]
    
    param_grid = dataset[parameters]
    spec_grid = dataset.drop(columns=parameters)

    const_params = [p for p in parameters if p != wrt]

    # index-based labels
    p1,p2,p3 = const_params

    labels_dict = {'temperature': ['Temperature (K)','T'],
              'gravity': ['log$g$ (dex)','g'],
              'c_o_ratio': ['C/O Ratio','C/O'],
              'metallicity': ['Metallicity [M/H] (dex)','M/H']}
    label_parts = [
        f"{labels_dict[wrt][1]} = {wrt_value}",
        f"{labels_dict[p1][1]} = {const_dict[p1]}",
        f"{labels_dict[p2][1]} = {const_dict[p2]}",
        f"{labels_dict[p3][1]} = {const_dict[p3]}"]

    var = labels_dict[wrt][1]

    const_grids = {p: np.sort(param_grid[p].unique()) for p in const_params}
    wrt_grid = np.sort(param_grid[wrt].unique())

    # locate grid points
    id_wrt = np.where(wrt_grid == wrt_value)[0][0]

    # Get wrt values for neighbors
    wrt_lo = wrt_grid[id_wrt - 1]
    wrt_hi = wrt_grid[id_wrt + 1]

    # Make masks for 2 gridpoints
    mask1 = param_grid[wrt] == wrt_lo
    mask2 = param_grid[wrt] == wrt_hi
    for p in const_params:
        mask1 &= param_grid[p] == const_dict[p]
        mask2 &= param_grid[p] == const_dict[p]


    id_gp1 = param_grid[mask1].index[0]
    id_gp2 = param_grid[mask2].index[0]

    # Slice jacobian
    num_wrt = np.where(parameters==wrt)[0][0]
    num_others = [i for i in range(4) if i != num_wrt]
    new_order = [num_wrt] + num_others + [4] # transpose to make wrt first dim
    jacob_reordered = np.transpose(jacob_matrix, new_order)
    # find slices
    const_indices = []
    for p in const_params:
        grid_vals = const_grids[p]
        const_idx = np.where(grid_vals == const_dict[p])[0][0]
        const_indices.append(const_idx)
    # slice
    id_p1, id_p2, id_p3 = const_indices    
    jacob_slice = jacob_reordered[id_wrt, id_p1, id_p2, id_p3, :]  # shape (N_wrt, N_lambda)

    # Create figure
    fig, axes = plt.subplots(2,1, figsize=(10,7))
    ax1,ax2 = axes

    ax0 = fig.add_subplot(111)
    ax0.axis("off")

    ax1.plot(wl, jacob_slice,color='b',label='Jacobian')

    ax2.plot(wl, spec_grid.iloc[id_gp2],color='r',alpha=0.7,label=f'{var}={wrt_hi}')
    ax2.plot(wl, spec_grid.iloc[id_gp1],color='orange',alpha=0.7,label=f'{var}={wrt_lo}')


    for ax in (ax1,ax2):
        ax.grid(True, axis='both',alpha=0.6,ls='--')
        ax.set_xlabel('Wavelength (μm)',fontsize=13)
        if logscale:
            ax.set_yscale('log')
        ax.legend()
        
    
    ax0.set_title(f"Jacobian: ∂F/∂{labels_dict[wrt][1]} ({','.join(label_parts)})\n")
    

    ax1.set_ylabel(f'∂F$_{{{wrt_value}}}$/∂{var}'+ r' $\approx$ ' + f'(F$_{{{wrt_hi}}}$-F$_{{{wrt_lo}}}$)/({wrt_hi-wrt_lo})',fontsize=13)
    ax2.set_ylabel(r'TOA F$_{\nu}^{\rm Syn}$  [erg/cm$^2$/s/Hz]',fontsize=13)

    for ax in (ax1,ax2):

        ax.tick_params(axis='both', which='major', direction='in', length=5, width=1,labelsize=12)
        ax.tick_params(axis='x', direction='in', top=True)
        ax.tick_params(axis='y', direction='in', right=True)
        ax.minorticks_on()
        ax.tick_params(axis='both', which='minor', direction='in', length=3, width=1)
        ax.tick_params(axis='x', which='minor', direction='in', top=True)
        ax.tick_params(axis='y', which='minor', direction='in', right=True)

    plt.subplots_adjust(hspace=0.25)

    plt.show()







def partial_heatmap(jacobian_matrix,
                    wrt,
                    wl,
                    dataset,
                    parameters = ['temperature',
                                    'gravity',
                                    'c_o_ratio',
                                    'metallicity'],
                    const_dict = {'temperature':800,
                                'gravity':4.0,
                                'c_o_ratio':0.5,
                                'metallicity':1},
                    __save__=False,
                    save_path=None
                   ):

    """
    Plot a heatmap of the jacobian of the model grid at a
    chosen grid point. 

    jacob_matrix: np.ndarray
        Jacobian matrix. 
    wrt: string
        Parameter that the Jacobian is calculated with respect to.
    wl: np.ndarray
        Wavelengths
    param_grid: pd.DataFrame
        All combinations of grid points. 
    const_dict: dict
        Dictionary of parameters to hold constant. 

    """
    parameters=np.array(parameters)
    # create list of constant names
    const_params = [p for p in parameters if p != wrt]
    p1,p2,p3 = const_params

    # create parameter grid of wrt variable
    wrt_grid = np.sort(dataset[wrt].unique())[1:-1]

    # ideces of params
    wrt_id = np.where(parameters == wrt)[0][0]
    other_ids = [i for i in range(4) if i != wrt_id]

    # transpose to make wrt axis 0
    new_order = [wrt_id] + other_ids + [4]
    jacob_reordered = np.transpose(jacobian_matrix, new_order)

    # create parameter grids of const variables 
    const_grids = {p: np.sort(dataset[p].unique()) for p in const_params}
    
    id_p1 = np.where(const_grids[p1] == const_dict[p1])[0][0]
    id_p2 = np.where(const_grids[p2] == const_dict[p2])[0][0]
    id_p3 = np.where(const_grids[p3] == const_dict[p3])[0][0]

    # slice
    jacob_slice = jacob_reordered[:, id_p1, id_p2, id_p3, :]  # shape (N_wrt, N_lambda)

    # create figure
    fig, ax = plt.subplots(figsize=(10,8))
    
    # manually skip redundant models 
    if wrt == 'metallicity':
        skip_vals = np.array([-0.7, -0.5, -0.3, 0.5])
        id_skip = np.where(np.isin(wrt_grid, skip_vals))[0]
        jacob_slice = np.delete(jacob_slice, id_skip, axis=0)
        wrt_grid = np.delete(wrt_grid, id_skip)

    # plot heatmap
    X, Y = np.meshgrid(wl, wrt_grid) # create meshgrid 
    
    global_slice_min = jacob_slice.min()
    global_slice_max = jacob_slice.max()
    
    if (global_slice_min<0) & (global_slice_max>0): # neg and pos values
        abs_slice = abs(jacob_slice)
        row_max = np.max(abs_slice, axis=1, keepdims=True)
        norm_slice = jacob_slice / row_max

        norm = TwoSlopeNorm(vmin=-1,vcenter=0,vmax=1)
        colors = [(0.0,"navy"),
                  (0.25,"skyblue"),
                  (0.5,"white"),
                  (0.66,"orange"),
                  (0.83,"red"),
                  (1.0,"darkred")] 
    elif (global_slice_min<0) & (global_slice_max<=0): # all neg values
        abs_slice = np.abs(jacob_slice)
        min_ = np.min(abs_slice, axis=1, keepdims=True)
        max_ = np.max(abs_slice, axis=1, keepdims=True)
        norm_slice = (abs_slice - min_) / (max_ - min_)
        norm = TwoSlopeNorm(vmin=0,
            vcenter=0.5,
            vmax=1)
        colors = [(0.0,"white"),
                  (0.5,"skyblue"),
                  (1.0,"navy")] 
    else: # all pos values
        slice_min = np.min(jacob_slice, axis=1, keepdims=True)  
        slice_max = np.max(jacob_slice, axis=1, keepdims=True)
        norm_slice = (jacob_slice - slice_min) / (slice_max - slice_min)
        norm = TwoSlopeNorm(vmin=0,
            vcenter=0.5,
            vmax=1)
        colors = [(0.0,"white"),
                  (0.33,"orange"),
                  (0.66,"red"),
                  (1.0,"darkred")] 
        
    cmap = LinearSegmentedColormap.from_list("custom",colors)
    im = ax.pcolormesh(X, Y, norm_slice, cmap=cmap, norm=norm, shading='auto')


    # label dictionary
    labels_dict = {'temperature': ['Temperature (K)','T'],
                  'gravity': ['log$g$ (dex)','g'],
                  'c_o_ratio': ['C/O Ratio','CO'],
                  'metallicity': ['Metallicity [M/H] (dex)','MH']}
    
    # index-based title label
    label_parts = [
        f"{labels_dict[p1][1]} = {const_dict[p1]}",
        f"{labels_dict[p2][1]} = {const_dict[p2]}",
        f"{labels_dict[p3][1]} = {const_dict[p3]}"]

    cbar = fig.colorbar(im,ax=ax)
    if (global_slice_min<0) & (global_slice_max<=0):
        cbar.set_ticks(np.arange(0,1.1,0.2))
        cbar.set_ticklabels(np.round(-1*np.arange(0,1.1,0.2),1))
        cbar.ax.invert_yaxis()
    elif (global_slice_min<0) & (global_slice_max>0):
        cbar.set_ticks(np.arange(-1,1.1,0.2))

    cbar.set_label(f'Normalized ∂F/∂{labels_dict[wrt][1]}', fontsize=13, rotation=270,labelpad=20)


    # xticks, yticks
    if wrt == 'temperature':
        yticks = wrt_grid
        ax.set_yticks(yticks)
        labels = [str(int(y)) if y in np.arange(200, 2401, 200) else '' for y in yticks]
        ax.set_yticklabels(labels)
    else:
        ax.set_yticks(wrt_grid)

    # axes labels
    ax.set_xlabel('Wavelength (μm)',fontsize=13)
    ax.set_ylabel(labels_dict[wrt][0],fontsize=13)
    ax.set_title(f"Jacobian: ∂F/∂{labels_dict[wrt][1]} ({','.join(label_parts)})",fontsize=13)


    # xlim
    ax.set_xlim(0.9,2.5)
    
    # tickmarks
    ax.tick_params(axis='both', which='major', direction='in', length=5, width=1,labelsize=12)
    ax.tick_params(axis='x', direction='in', top=True)
    ax.tick_params(axis='y', direction='in', right=True)
    ax.xaxis.minorticks_on()
    ax.tick_params(axis='x', which='minor', direction='in', length=3, width=1)
    ax.tick_params(axis='x', which='minor', direction='in', top=True)

    
    if __save__:
        save_path = f'{wrt}_heatmap_plot.pdf' if save_path is None else save_path
        plt.savefig(save_path,bbox_inches='tight')
    else:
        plt.show()
        










