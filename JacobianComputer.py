
import os
import numpy as np
import pandas as pd
from JacobianVisualizer import partial_heatmap
import importlib

def compute_jacobian(dataset, 
                     wl, 
                     wrt, 
                     parameters = ['temperature',
                                    'gravity',
                                    'c_o_ratio',
                                    'metallicity'],
                     __save__=True,
                     save_path=None,
                    ):
    """
    Compute the Jacobian of the synthetic dataset with respect to chosen 
    parameter over the 4 parameters in the grid. 

    dataset: pd.DataFrame
        Dataset with columns corresponding to parameter grid 
        points and synthetic fluxes over the wavelength grid. 
    wl: np.ndarray  
        Wavelength grid of the spectra. 
    wrt: string
        Parameter to compute Jacobian matrix with respect to.
    parameters: list
        List of the names of parameters in the parameter space. 
    __save__ : bool
        Whether or not to save the matrix. 
    save_path : string
        Path to save matrix. 
    """
    
    df = dataset
    parameters = np.array(parameters)

    params_grids = {param: np.sort(df[param].unique()) for param in parameters}

    const_params = [p for p in parameters if p != wrt]
    wrt_grid = params_grids[wrt]

    p1, p2, p3 = const_params
    p1_grid = params_grids[p1]
    p2_grid = params_grids[p2]
    p3_grid = params_grids[p3]

    
    jacob_matrix = np.zeros((len(wrt_grid)-2,
                         len(p1_grid),
                         len(p2_grid),
                         len(p3_grid),
                         len(wl)))


    for id_p1 in range(len(p1_grid)):
        for id_p2 in range(len(p2_grid)):
            for id_p3 in range(len(p3_grid)):
                
                p1i = p1_grid[id_p1]
                p2i = p2_grid[id_p2]
                p3i = p3_grid[id_p3]

                df_sub = df.loc[
                    (df[p1] == p1i) &
                    (df[p2] == p2i) &
                    (df[p3] == p3i)
                ].sort_values(wrt).reset_index(drop=True)

                for ind in range(1, df_sub.shape[0]-1):
                    flux1 = df_sub.iloc[ind+1, 4:]
                    flux2 = df_sub.iloc[ind-1, 4:]

                    P1 = df_sub.iloc[ind+1][wrt]
                    P2 = df_sub.iloc[ind-1][wrt]

                    dfdP = (flux1 - flux2) / (P1 - P2)

                    jacob_matrix[ind-1, id_p1, id_p2, id_p3, :] = dfdP

    # reshape matrix to standard teff x g x co x mh x lam order
    current_order = [wrt, p1, p2, p3]
    axis_map = {k: i for i, k in enumerate(current_order)}
    transpose_order = [axis_map[p] for p in parameters]
    jacob_matrix = np.transpose(jacob_matrix, transpose_order + [4])  # final dim is lam

    if __save__:
        np.save(os.path.join(save_path,f"{wrt}_matrix.npy"),jacob_matrix)

    return jacob_matrix

