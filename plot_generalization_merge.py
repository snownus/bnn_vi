import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


starting_epochs = 1

fig, axs = plt.subplots(2, 2, figsize=(15, 13))

plot_config = [
    ('complete', 'train', 0, 'Training Loss on Complete Graph'),
    ('complete', 'test', 1, 'Testing Loss on Complete Graph'),
    ('random', 'train', 2, 'Training Loss on Random Graph'),
    ('random', 'test', 3, 'Testing Loss on Random Graph')
]

# Colors and labels for the legend
colors1 = ['blue', 'orange', 'green']
labels = ['SGD', 'Adam', 'VISPA (Ours)']

for dataset, split, index, title in plot_config:
    if dataset == 'complete':
        # Read the CSV files
        df1 = pd.read_csv(f'robustness/{dataset}_0.1_seedvalue_2020_{split}.csv')
        df2 = pd.read_csv(f'robustness/{dataset}_0.1_seedvalue_2024_{split}.csv')
        df3 = pd.read_csv(f'robustness/{dataset}_0.1_seedvalue_2333_{split}.csv')
        df4 = pd.read_csv(f'robustness/{dataset}_0.1_seedvalue_1314_{split}.csv')
        df5 = pd.read_csv(f'robustness/{dataset}_0.1_seedvalue_512_{split}.csv')
    elif dataset == 'random':
        # Read the CSV files
        df1 = pd.read_csv(f'robustness/{dataset}_0.5_seedvalue_2020_{split}.csv')
        df2 = pd.read_csv(f'robustness/{dataset}_0.5_seedvalue_2024_{split}.csv')
        df3 = pd.read_csv(f'robustness/{dataset}_0.5_seedvalue_2333_{split}.csv')
        df4 = pd.read_csv(f'robustness/{dataset}_0.5_seedvalue_1314_{split}.csv')
        df5 = pd.read_csv(f'robustness/{dataset}_0.5_seedvalue_512_{split}.csv')

    df = pd.concat([df1, df2, df3, df4, df5], axis=1)

    sgd, adam, ours = df.iloc[:, [0, 3, 6, 9, 12]].mean(axis=1)[starting_epochs-1:], df.iloc[:, [1, 4, 7, 10, 13]].mean(axis=1)[starting_epochs-1:], df.iloc[:, [2, 5, 8, 11, 14]].mean(axis=1)[starting_epochs-1:]
    sgdstd, adamstd, oursstd = df.iloc[:, [0, 3, 6, 9, 12]].std(axis=1)[starting_epochs-1:], df.iloc[:, [1, 4, 7, 10, 13]].std(axis=1)[starting_epochs-1:], df.iloc[:, [2, 5, 8, 11, 14]].std(axis=1)[starting_epochs-1:]

    # Calculate upper and lower bounds using standard deviation
    sgdupper, adamupper, oursupper = sgd + sgdstd, adam + adamstd, ours + oursstd
    sgdlower, adamlower, ourslower = sgd - sgdstd, adam - adamstd, ours - oursstd

    results = [sgd, adam, ours]
    resultsstd = [sgdstd, adamstd, oursstd]
    resultsupper = [sgdupper, adamupper, oursupper]
    resultslower = [sgdlower, adamlower, ourslower]
    colors = ['lightskyblue', 'lightsalmon', 'lightgreen']

    x = [i for i in range(starting_epochs, 101)]
    ax = axs[index // 2, index % 2]  # Adjusted for 2x2 layout
    for i, label in enumerate(labels):
        ax.plot(x, results[i], color=colors1[i], label=label)
        ax.fill_between(x, resultslower[i], resultsupper[i], color=colors[i], alpha=0.5)

    # Adjust y-axis limits based on split and dataset, and add extra space for legend
    if split == 'train' and dataset == 'random':
        ax.set_ylim(0, 10)
    if split == 'train' and dataset == 'complete':
        ax.set_ylim(0, 10)
    # if split == 'test' and dataset == 'random':
    #     ax.set_ylim(0.5, 2)
    # if split == 'test' and dataset == 'complete':
    #     ax.set_ylim(0.5, 3)

    ax.set_xlabel('Epoch', fontweight='bold', fontsize=18)
    ax.set_ylabel('Loss', fontweight='bold', fontsize=18)

    # Set font size and weight for tick labels
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize('xx-large')  # Reduce the font size for tick labels

    # Add title below each subplot
    ax.set_title(title, fontweight='bold', fontsize=18, pad=10)

    # Add individual legends for each subplot
    ax.legend(loc='upper center', bbox_to_anchor=(0.65, 0.6), ncol=3, columnspacing=0.4, 
              frameon=False, prop={'weight': 'bold', 'size': 14})

    # Bold plot edges
    for spine in ax.spines.values():
        spine.set_linewidth(2)

fig.subplots_adjust(hspace=2)

plt.tight_layout()
# Save the figure as a PDF
pdf_path = './generalization_combined.pdf'
plt.savefig(pdf_path, format='pdf', bbox_inches='tight')

plt.show()