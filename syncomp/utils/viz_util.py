import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_heatmap(data, title='', xlabel='', ylabel='', cmap='coolwarm', ax=None, mask:bool=True, **kwargs):
    ax = ax or plt.gca()
    xticklabels = data.columns
    yticklabels = data.index
    # Getting the Upper Triangle of the co-relation matrix
    if mask:
        matrix = np.triu(data)
    else:
        matrix = np.zeros_like(data)
    sns.heatmap(data, cmap=cmap, fmt=".2f", xticklabels=xticklabels, yticklabels=yticklabels, ax=ax, mask=matrix, **kwargs)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    return ax

def plot_hist(real, syn, real_name='real', syn_name='syn', title='', xlabel='', ylabel='', ax=None, **kwargs):
    ax = ax or plt.gca()
    real_df = pd.DataFrame({
        'data': real,
        'label': real_name
    })
    syn_df = pd.DataFrame({
        'data': syn,
        'label': syn_name
    })
    sns.histplot(data=pd.concat([real_df, syn_df]), x='data', hue='label', ax=ax, **kwargs)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    return ax