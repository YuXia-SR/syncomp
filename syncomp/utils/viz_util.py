import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def plot_heatmap(data, title='', xlabel='', ylabel='', cmap='coolwarm', ax=None, **kwargs):
    ax = ax or plt.gca()
    xticklabels = data.columns
    yticklabels = data.index
    sns.heatmap(data, cmap=cmap, fmt=".2f", xticklabels=xticklabels, yticklabels=yticklabels, ax=ax, **kwargs)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    return ax

def plot_hist(real, syn, title='', xlabel='', ylabel='', ax=None, **kwargs):
    ax = ax or plt.gca()
    real_df = pd.DataFrame({
        'data': real,
        'label': 'real'
    })
    syn_df = pd.DataFrame({
        'data': syn,
        'label': 'syn'
    })
    sns.histplot(data=pd.concat([real_df, syn_df]), x='data', hue='label', ax=ax, **kwargs)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    return ax