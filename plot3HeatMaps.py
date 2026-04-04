import matplotlib.pyplot as plt
import seaborn as sns

def plot_three_heatmaps(dfs, titles, cmap="coolwarm", vmin=None, vmax=None):
    """
    Plots three heatmaps side by side with standardized color scales.
    
    dfs   : list of 3 DataFrames
    titles: list of 3 titles
    cmap  : colormap
    vmin  : min value for color scale
    vmax  : max value for color scale
    """
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    for i in range(3):
        sns.heatmap(dfs[i].astype(float), annot=True, fmt=".2f", cmap=cmap,
                    vmin=vmin, vmax=vmax, ax=axes[i])
        axes[i].set_title(titles[i])
        axes[i].set_xlabel("Look-ahead")
        axes[i].set_ylabel("Look-Back")
    
    plt.tight_layout()
    plt.show()