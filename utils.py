import pandas as pd
import numpy as np
import time
import os

def set_modelID():
    path = 'models/'
    t = str(pd.to_datetime('today').date())
    models = os.listdir(path)
    models = [model for model in models if model.startswith(t)]
    i = max([int(model[-1]) for model in models]) + 1
    return f'{t}_{i}'