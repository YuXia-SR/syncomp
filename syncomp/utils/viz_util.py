import seaborn as sns
import matplotlib.pyplot as plt

def plot_heatmap(data, title='', xlabel='', ylabel='', cmap='coolwarm', ax=None, **kwargs):
    ax = ax or plt.gca()
    xticklabels = data.columns
    yticklabels = data.index
    sns.heatmap(data, cmap=cmap, fmt=".2f", xticklabels=xticklabels, yticklabels=yticklabels, ax=ax, **kwargs)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    return ax

