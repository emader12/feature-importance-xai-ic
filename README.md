# feature-importance-xai-ic
Looking a spectral feature importance using eXplainable AI (XAI). Train a CNN on spectral grid, compute SHAP values, and compute model sensitivity with IC, and compare the difference with useful visualizations.

## Reference Data
Reference data, including the dataset generated from Sonora Bobcat spectral grid using Picasso, is from TelescopeML and can be downloaded here: [https://zenodo.org/records/11495456]. You will need browndwarf_R100_v4_newWL_v3.csv.bz2 

## **To-Do:**

-  Automatically create CNN input: train/val/test sets and dense layers for each parameter (easy)
-  Create BOHB tuning pipeline for given spectral grid (moderate)
-  Automate SHAP pipeline to save values based on given parameter list (easy)
-  Generalize Jacobian computer to be able to take any number of parameters, right now it can take any kind but needs exactly 4. 
-  Generalize plotting functions (easy): Overall, they need to be able to be flexible to how many parameters in the Jacobian for slicing. It may need be done most easily manually with if else statements 2,3,4 parameters. Additionally needs to take a dictionary for labeling the parameters.
    - TrainCNN.plot_ML_model_loss_bokeh
    - JacobianVisualizer.delta_flux_delta_J (verifies jacobian output)
    - SHAPVisualizer.plot_shap_spread (shap importance over spectra varied by given param)
    - JacobianVisualizer.heatmap_vs_XAI
    - JacobianVisualizer.compare_IC_ML
