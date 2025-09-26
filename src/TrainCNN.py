import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import save_model
import matplotlib.pyplot as plt
from bokeh.plotting import output_notebook
from bokeh.plotting import figure, show
import pickle


def split_dataset(flux_values,
                  output_values,
                  test_size=0.1,
                  val_size = 0.1,
                  random_state = 42,
                  __shuffle__=True):
    
    X_train, X_test, y_train, y_test = train_test_split(flux_values,
                                                        output_values,
                                                        test_size=test_size,
                                                        shuffle=__shuffle__,
                                                        random_state=random_state)
    X_train, X_val, y_train, y_val = train_test_split(X_train,
                                                      y_train,
                                                      test_size=val_size,
                                                      shuffle=__shuffle__,
                                                      random_state=random_state)
    
    return X_train, X_test, X_val, y_train, y_test, y_val



def plot_ML_model_loss_bokeh(trained_ML_model_history=None, title=None):
    """
    Plot the trained model history for all individual target features
    """

    # Define the epochs as a list
    epochs = list(range(len(trained_ML_model_history['loss'])))

    # Define colorblind-friendly colors
    colors = ['#d62728', '#ff7f0e', '#2ca02c', '#9467bd', '#8c564b']

    # Create a new figure
    p = figure(title=title, width=1000, height=300, y_axis_type='log', x_axis_label='Epochs', y_axis_label='Loss')

    # Add the data lines to the figure with colorblind-friendly colors and increased line width
    p.line(epochs, trained_ML_model_history['loss'], line_color=colors[0], line_dash='solid', line_width=2,
           legend_label='Total loss')
    p.line(epochs, trained_ML_model_history['val_loss'], line_color=colors[0], line_dash='dotted', line_width=2)

    p.line(epochs, trained_ML_model_history['output__gravity_loss'], line_color=colors[1], line_dash='solid', line_width=2,
           legend_label='gravity')
    p.line(epochs, trained_ML_model_history['val_output__gravity_loss'], line_color=colors[1], line_dash='dotted', line_width=2)

    p.line(epochs, trained_ML_model_history['output__c_o_ratio_loss'], line_color=colors[2], line_dash='solid', line_width=2,
           legend_label='c_o_ratio')
    p.line(epochs, trained_ML_model_history['val_output__c_o_ratio_loss'], line_color=colors[2], line_dash='dotted', line_width=2)

    p.line(epochs, trained_ML_model_history['output__metallicity_loss'], line_color=colors[3], line_dash='solid', line_width=2,
           legend_label='metallicity')
    p.line(epochs, trained_ML_model_history['val_output__metallicity_loss'], line_color=colors[3], line_dash='dotted', line_width=2)

    p.line(epochs, trained_ML_model_history['output__temperature_loss'], line_color=colors[4], line_dash='solid', line_width=2,
           legend_label='temperature')
    p.line(epochs, trained_ML_model_history['val_output__temperature_loss'], line_color=colors[4], line_dash='dotted', line_width=2)

    #Increase size of x and y ticks
    p.xaxis.major_label_text_font_size = '12pt'
    p.xaxis.axis_label_text_font_size = '12pt'
    p.yaxis.major_label_text_font_size = '12pt'
    p.yaxis.axis_label_text_font_size = '12pt'

    # display legend in top left corner (default is top right corner)
    p.legend.location = "bottom_left"
    p.legend.background_fill_color = 'white'
    p.legend.background_fill_alpha = 0.5

    # Show the plot
    show(p)