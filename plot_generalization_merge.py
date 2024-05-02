import pandas as pd
import matplotlib.pyplot as plt

starting_point = 20
for dataset in ['complete', 'random']:
    # Read the CSV file
    
    labels = ['Training loss - SGD', 'Training loss - Adam', 'Training loss - Ours',
              'Testing loss - SGD', 'Testing loss - Adam', 'Testing loss - Ours',
              ]
    df_train = pd.read_csv('robustness/' + dataset + '_0.01_' + 'train' + '.csv')
    df_test = pd.read_csv('robustness/' + dataset + '_0.01_' + 'test' + '.csv')
    sgd_train,adam_train,ours_train = df_train['SGD'][starting_point:], df_train['Adam'][starting_point:], df_train['Ours'][starting_point:]
    sgd_test,adam_test,ours_test = df_test['SGD'][starting_point:], df_test['Adam'][starting_point:], df_test['Ours'][starting_point:]

    results = [sgd_train, adam_train, ours_train, sgd_test,adam_test,ours_test]

    x = [i for i in range(starting_point+1, 101)]
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
    legend = ax.legend(fontsize='medium', loc='upper right', bbox_to_anchor=(0.95, 0.5), fancybox=True, shadow=True)
    for text in legend.get_texts():
        text.set_fontweight('bold')

    # Bold plot edges
    for spine in ax.spines.values():
        spine.set_linewidth(2)


    plt.tight_layout()
    # Save the figure as a PDF
    pdf_path = './robustness/generalization_' + dataset + '.pdf'
    plt.savefig(pdf_path, format='pdf')

    plt.show()