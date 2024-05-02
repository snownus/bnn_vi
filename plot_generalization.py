import pandas as pd
import matplotlib.pyplot as plt

for dataset in ['complete', 'random']:
    for split in ['train', 'test']:
        # Read the CSV file
        df = pd.read_csv('robustness/' + dataset + '_0.001_' + split + '.csv')
        labels = ['SGD', 'Adam', 'Ours']
        sgd,adam,ours = df['SGD'][49:], df['Adam'][49:], df['Ours'][49:]

        results = [sgd, adam, ours]

        x = [i for i in range(50, 101)]
        fig, ax = plt.subplots(figsize=(5, 4))
        for i, label in enumerate(labels):
            ax.plot(x, results[i], '-', label=label)

        ax.set_xlabel('K', fontweight='bold', fontsize='x-large')
        ax.set_ylabel('Loss', fontweight='bold', fontsize='x-large')

        # Bold tick labels
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight('bold')
            label.set_fontsize('large')

        # Legend with bold text
        if dataset == 'random':
            legend = ax.legend(fontsize='medium', loc='upper right', bbox_to_anchor=(0.95, 0.75), fancybox=True, shadow=True)
        else:
            legend = ax.legend(fontsize='medium', loc='upper right', bbox_to_anchor=(0.95, 0.5), fancybox=True, shadow=True)
        for text in legend.get_texts():
            text.set_fontweight('bold')

        # Bold plot edges
        for spine in ax.spines.values():
            spine.set_linewidth(2)


        plt.tight_layout()
        # Save the figure as a PDF
        pdf_path = './robustness/generalization_' + dataset + '_' + split + '.pdf'
        plt.savefig(pdf_path, format='pdf')

        plt.show()