#------------------------------------------------------------------------------- SCRIPT INFO


'''
File name:              vis.py
Author:                 Janik Vollenweider, Julia Usher, Oliver Carmignani
Date last modified:     15/04/2024
Python Version:         3.7

FILE HANDLING SYNTAX
    # QUESTION
    # SPEEDUP
    # DEBUGG
    # NOT FINISHED
    # !
    # TODO
'''


#------------------------------------------------------------------------------- MODULES


## LIBRARIES
import seaborn as sns  
import matplotlib.pyplot as plt  
import pandas as pd
import numpy as np
import pickle


#------------------------------------------------------------------------------- SETTINGS


# APPEARENCE
color_pal = sns.color_palette() # Use seaborn==0.12.2 for best visualizations
plt.style.use('ggplot')


# #------------------------------------------------------------------------------- FUNCTIONS

# Show metrics of Trained Model
def show_metrics(models, eval_metrics, run_name, vis=True, save=False):
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
    fig, ax = plt.subplots(len(models), len(eval_metrics.keys()), figsize=(15, 5))
    ax = [0] * (len(models) * len(eval_metrics.keys()))
    lv = 1
    for i, model in enumerate(models):
        for key, value in eval_metrics.items():
            ax[i] = plt.subplot(len(models), len(eval_metrics.keys()), lv)
            ax[i].plot(value[0], label=f'Training - {key}')
            ax[i].plot(value[1], label=f'Test - {key}')
            ax[i].legend()
            ax[i].set_title(f'{key}')
            lv += 1
    fig.tight_layout()
    if save:
        plt.savefig('Models/' + run_name + '/plots/' + run_name + '_metrics')
    if vis:
        plt.show()
    else:
        plt.close()





#------------------------------------------------------------------------------- MAIN
if __name__=='__main__':
    # Load evaluation metric
    with open('Model/eval_dat', 'rb') as f:
        eval_metrics = pickle.load(f)
    models = ['model.pt']
    run_name = 'Exchange Rate Prediction'
    show_metrics(models, eval_metrics, run_name, vis=True, save=False)
    

