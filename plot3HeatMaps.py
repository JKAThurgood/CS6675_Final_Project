import matplotlib.pyplot as plt
import seaborn as sns

def plot_three_heatmaps(dfs, titles, cmap="coolwarm", vmin=None, vmax=None):
    """
    Plots three heatmaps, each on its own row, with standardized color scales.
    
    dfs   : list of 3 DataFrames
    titles: list of 3 titles
    cmap  : colormap
    vmin  : min value for color scale
    vmax  : max value for color scale
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 18))  # 3 rows, 1 column
    
    for i in range(3):
        sns.heatmap(dfs[i].astype(float), annot=True, fmt=".2f", cmap=cmap,
                    vmin=vmin, vmax=vmax, ax=axes[i])
        axes[i].set_title(titles[i], fontsize=14)
        axes[i].set_xlabel("Look-ahead")
        axes[i].set_ylabel("Look-Back")
    
    plt.tight_layout()
    plt.show()