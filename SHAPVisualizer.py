import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pickle as pk
import pandas as pd
import shap
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize,TwoSlopeNorm,LinearSegmentedColormap
import seaborn as sns
from scipy.signal import find_peaks, peak_prominences


def plot_shap_spread(dataset, 
                     wl_synthetic, 
                     feature_to_plot, 
                     filter_bounds, 
                     norm_mean_shap,
                     palette='viridis',
                     output_names = ['gravity', 'temperature', 'c_o_ratio', 'metallicity'],
                     dict_features1 = {'temperature': 'T$_{eff}$ [K]', 
                                      'gravity': 'log$g$', 
                                      'metallicity': '[M/H]', 
                                      'c_o_ratio': 'C/O'},
                     dict_features2 = {'temperature': 'Effective Temperature', 
                                       'gravity': 'Gravity', 
                                       'metallicity': 'Metallicity',
                                       'c_o_ratio': 'Carbon-to-oxygen ratio'}
                    ):

    # Filter dataset to bounds
    filtered_df = dataset.copy()
    for feature, bounds in filter_bounds.items():
        lower_bound, upper_bound = bounds
        filtered_df = filtered_df[(filtered_df[feature] >= lower_bound) & (filtered_df[feature] <= upper_bound)]
    
    # Sort by ascending feature to plot
    filtered_df2 = filtered_df.sort_values(feature_to_plot, ascending=True).drop(columns=output_names)
    # Transpose the DataFrame
    df_transposed = filtered_df2.T  

    # Create figure
    fig, ax1 = plt.subplots(figsize=(12, 4))
    
    # Define a color palette
    num_colors = len(df_transposed.columns)  # Number of colors needed (excluding x-axis)
    colors = sns.color_palette(palette, num_colors)

    # Plot spectra
    for i, col in enumerate(df_transposed.columns):
        if col != 'x':  # Skip the x-axis column
            ax1.semilogy(wl_synthetic, df_transposed[col],
                        color=colors[i], alpha=0.7)
    
    # Add colorbar
    cmap = sns.color_palette(palette, as_cmap=True)
    cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap,
                                              norm=plt.Normalize(vmin=filter_bounds[feature_to_plot][0],
                                                                 vmax=filter_bounds[feature_to_plot][1])), 
                                              ax=ax1,
                                              pad=0.01)
    cbar.set_label(dict_features1[feature_to_plot], fontsize = 12)  

    # Adjust y limits
    ymin, ymax = ax1.get_ylim()
    minexp, maxexp = np.log10(ymin),np.log10(ymax)
    ax1.set_ylim(10**(minexp-2), 10**(maxexp))

    # Plot SHAP
    ax2 = ax1.twinx()
    ax2.plot(wl_synthetic, norm_mean_shap*0.3, color='k', label="Mean Abs SHAP",lw=1.5)
    ax2.set_ylim(0,1)
    ax2.set_yticks([])
    ax2.tick_params(axis='y', labelcolor='k')
    ax2.legend()

    # Plot top 3 SHAP peaks
    peaks, _ = find_peaks(norm_mean_shap)
    proms = peak_prominences(norm_mean_shap, peaks)[0]
    top3_idx = np.argsort(proms)[-3:]
    top3_peaks = peaks[top3_idx]
    for ax in (ax1,ax2):
        for peak in top3_peaks:
            ax.axvline(wl_synthetic[peak], ymin=norm_mean_shap.min(), ymax=norm_mean_shap.max(), color='k', lw=0.8,linestyle='--')

    ax1.set_xlabel('Wavelength [$\mu$m]', fontsize = 12)
    ax1.set_ylabel(r'TOA F$_{\nu}^{\rm Syn}$  [erg/cm$^2$/s/Hz]', fontsize = 12)

    title_parts = [f"{label}={str(filter_bounds[key][0])}" for key, label in dict_features1.items() if key != feature_to_plot]
    title_label = ', '.join(title_parts)
    ax1.set_title(f"{dict_features2[feature_to_plot]} [{title_label}]", fontsize = 14)

    for ax in (ax1,ax2):
        ax.tick_params(axis='both', which='major', direction='in', length=5, width=1,labelsize=12)
        ax.tick_params(axis='x', direction='in', top=True)
        ax.tick_params(axis='y', direction='in', right=True)
        ax.minorticks_on()
        ax.tick_params(axis='both', which='minor', direction='in', length=3, width=1)
        ax.tick_params(axis='x', which='minor', direction='in', top=True)
        ax.tick_params(axis='y', which='minor', direction='in', right=True)
            
    plt.show()