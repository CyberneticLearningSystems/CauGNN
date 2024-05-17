#------------------------------------------------------------------------------- SCRIPT INFO


'''
File name:              vis.py
Author:                 Janik Vollenweider, Julia Usher, Oliver Carmignani
Date last modified:     22/04/2024
Python Version:         3.7

FILE HANDLING SYNTAX
    #? QUESTION
    #* SPEEDUP
    #! DEBUG
    # NOT FINISHED
    # TODO
'''


#------------------------------------------------------------------------------- MODULES


## LIBRARIES
import seaborn as sns  
import matplotlib.pyplot as plt  
import pandas as pd
import numpy as np
import pickle
import os


#------------------------------------------------------------------------------- SETTINGS


# APPEARENCE
color_pal = sns.color_palette() # Use seaborn==0.12.2 for best visualizations
plt.style.use('ggplot')


# #------------------------------------------------------------------------------- FUNCTIONS
# TODO: check that valid <--> test change didn't have any effects
# Show metrics of Trained Model
def show_metrics(plot_metrics, run_name, vis=True, save=False):
    """
    # NOT FINISHED
    DESCRIPTION:
        <This function only serves as a template for copy pasting during
        the work-in-progress>

    INPUT:
        - <argument name>       <description of argument>               [<python format of argument>]

    OUTPUT:
        - <prints, plots, or saved files>

    RETURN:
        - <python variable>     <description>                           [<python format of return>]
    """
    fig, ax = plt.subplots(1, len(plot_metrics.keys()), figsize=(15, 5))
    ax = [0] * (1 * len(plot_metrics.keys()))
    lv = 1
    fig.suptitle(run_name)
    for key, value in plot_metrics.items():
        ax = plt.subplot(1, len(plot_metrics.keys()), lv)
        ax.plot(value[0], label=f'Training - {key}')
        ax.plot(value[1], label=f'Test - {key}')
        ax.legend()
        ax.set_xlabel('Epochs')
        ax.set_ylabel(f'{key}')
        lv += 1
    fig.tight_layout()
    if save:
        plt.savefig(os.path.join(save, 'plot_metrics.png'))
    if vis:
        plt.show()
    else:
        plt.close()
    
    return fig

# Show metrics of Model while Training
def show_metrics_continous(plot_metrics):
    """
    # NOT FINISHED
    DESCRIPTION:
        <This function only serves as a template for copy pasting during
        the work-in-progress>

    INPUT:
        - <argument name>       <description of argument>               [<python format of argument>]

    OUTPUT:
        - <prints, plots, or saved files>

    RETURN:
        - <python variable>     <description>                           [<python format of return>]
    """
    fig, ax = plt.subplots(1, len(plot_metrics.keys()), figsize=(5, 5))
    ax = [0] * (1 * len(plot_metrics.keys()))
    lv = 1
    for key, value in plot_metrics.items():
        ax[0] = plt.subplot(1, len(plot_metrics.keys()), lv)
        line1, = ax[0].plot(value[0], label=f'Training - {key}')
        line2, = ax[0].plot(value[1], label=f'Test - {key}')
        ax[0].legend()
        ax[0].set_title(f'{key}')
        lv += 1
    fig.tight_layout()

    return fig, ax, line1, line2




#------------------------------------------------------------------------------- MAIN
if __name__=='__main__':
    # Load evaluation metric
    with open('Model/eval_dat', 'rb') as f:
        plot_metrics = pickle.load(f)
    models = ['model.pt']
    run_name = 'Exchange Rate Prediction'
    show_metrics(models, plot_metrics, run_name, vis=True, save=False)
    

