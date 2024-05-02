import matplotlib.pyplot as plt

# Data for plotting
labels_reordered = ['ResNet + CIFAR-10', 'VGGNet + CIFAR-10', 'VGGNet + CIFAR-100', 'ResNet + CIFAR-100']
means_reordered = [
    [87.212, 86.826, 87.028, 85.652, 84.224, 84.264],
    [88.928, 89.074, 88.892, 88.920, 88.914, 88.796],
    [68.774, 68.934, 69.176, 69.36, 69.444, 69.356],
    [64.614, 65.198, 64.848, 64.186, 63.620, 63.176]
]
errors_reordered = [
    [0.302, 0.481, 0.450, 0.425, 0.914, 0.528],
    [0.226, 0.123, 0.238, 0.094, 0.223, 0.177],
    [0.239, 0.142, 0.143, 0.253, 0.168, 0.077],
    [0.512, 0.405, 0.542, 0.572, 0.579, 0.917]
]
K_values = [1, 2, 4, 6, 8, 10]

# Create the plot
fig, ax = plt.subplots(figsize=(5, 4))
for i, label in enumerate(labels_reordered):
    ax.errorbar(K_values, means_reordered[i], yerr=errors_reordered[i], label=label, fmt='-o', capsize=5, elinewidth=2, markeredgewidth=2)

ax.set_xlabel('K', fontweight='bold', fontsize='x-large')
ax.set_ylabel('Accuracy', fontweight='bold', fontsize='x-large')

# Bold tick labels
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontweight('bold')
    label.set_fontsize('large')

# Legend with bold text
legend = ax.legend(fontsize='medium', loc='upper right', bbox_to_anchor=(0.95, 0.75), fancybox=True, shadow=True)
for text in legend.get_texts():
    text.set_fontweight('bold')

# Bold plot edges
for spine in ax.spines.values():
    spine.set_linewidth(2)

plt.grid(True)

plt.tight_layout()
# Save the figure as a PDF
pdf_path = './Error_Bar_Plot_Bold_Edges.pdf'
plt.savefig(pdf_path, format='pdf')

plt.show()
