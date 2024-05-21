import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

fig, axs = plt.subplots(1, 4, figsize=(20, 5))

plot_config = [
    ('complete', 'train', 0, 'Training Loss on Complete Graph'),
    ('complete', 'test', 1, 'Testing Loss on Complete Graph'),
    ('random', 'train', 2, 'Training Loss on Random Graph'),
    ('random', 'test', 3, 'Testing Loss on Random Graph')
]

# Colors and labels for the legend
colors1 = ['blue', 'orange', 'green']
labels = ['SGD', 'Adam', 'Ours']

for dataset, split, index, title in plot_config:
    # Read the CSV files
    df1 = pd.read_csv(f'robustness/{dataset}_0.01_seedvalue_2020_{split}.csv')
    df2 = pd.read_csv(f'robustness/{dataset}_0.01_seedvalue_2024_{split}.csv')
    df3 = pd.read_csv(f'robustness/{dataset}_0.01_seedvalue_2333_{split}.csv')
    df4 = pd.read_csv(f'robustness/{dataset}_0.01_seedvalue_1314_{split}.csv')
    df5 = pd.read_csv(f'robustness/{dataset}_0.01_seedvalue_512_{split}.csv')
    df = pd.concat([df1, df2, df3, df4, df5], axis=1)

    sgd, adam, ours = df.iloc[:, [0, 3, 6, 9, 12]].mean(axis=1)[49:], df.iloc[:, [1, 4, 7, 10, 13]].mean(axis=1)[49:], df.iloc[:, [2, 5, 8, 11, 14]].mean(axis=1)[49:]
    sgdstd, adamstd, oursstd = df.iloc[:, [0, 3, 6, 9, 12]].std(axis=1)[49:], df.iloc[:, [1, 4, 7, 10, 13]].std(axis=1)[49:], df.iloc[:, [2, 5, 8, 11, 14]].std(axis=1)[49:]

    # Calculate upper and lower bounds using standard deviation
    sgdupper, adamupper, oursupper = sgd + sgdstd, adam + adamstd, ours + oursstd
    sgdlower, adamlower, ourslower = sgd - sgdstd, adam - adamstd, ours - oursstd

    results = [sgd, adam, ours]
    resultsstd = [sgdstd, adamstd, oursstd]
    resultsupper = [sgdupper, adamupper, oursupper]
    resultslower = [sgdlower, adamlower, ourslower]
    colors = ['lightskyblue', 'lightsalmon', 'lightgreen']

    x = [i for i in range(50, 101)]
    ax = axs[index]
    for i, label in enumerate(labels):
        ax.plot(x, results[i], color=colors1[i], label=label)
        ax.fill_between(x, resultslower[i], resultsupper[i], color=colors[i], alpha=0.5)

    # Adjust y-axis limits based on split and dataset, and add extra space for legend
    if split == 'test' and dataset == 'random':
        ax.set_ylim(0.990, 1.015)
    if split == 'test' and dataset == 'complete':
        ax.set_ylim(0.990, 1.035)
    if split == 'train' and dataset == 'random':
        ax.set_ylim(0.82, 1.03)
    if split == 'train' and dataset == 'complete':
        ax.set_ylim(0.27, 1.12)

    ax.set_xlabel('Epoch', fontweight='bold', fontsize=18)

    # Set font size and weight for tick labels
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        # label.set_fontweight('bold')
        label.set_fontsize('small')  # Reduce the font size for tick labels

    # Add title below each subplot
    ax.set_title(title, fontweight='bold', fontsize=18, pad=10)

    # Add individual legends for each subplot
    # ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize='large', frameon=False)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 0.15), ncol=3, columnspacing=0.4, 
              frameon=False, prop={'weight': 'bold', 'size': 16})
    # Bold plot edges
    for spine in ax.spines.values():
        spine.set_linewidth(2)

# Set the y-axis label for the first subplot
axs[0].set_ylabel('Loss', fontweight='bold', fontsize=18)

plt.tight_layout()
# Save the figure as a PDF
pdf_path = './generalization_combined.pdf'
plt.savefig(pdf_path, format='pdf', bbox_inches='tight')

plt.show()