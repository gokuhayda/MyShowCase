import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_correlation_heatmap(df):
    corr = df.corr()

    plt.figure(figsize=(25, 20))

    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, cbar_kws={"shrink": .8},
                annot_kws={"size": 10})

    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)

    plt.title('Correlation Heatmap', fontsize=14)

    plt.show()
