import matplotlib.pyplot as plt

# Data values and errors
means_new = [59.942, 68.658, 69.210, 68.804, 68.462]
errors_new = [0.459, 0.298, 0.248, 0.099, 0.131]
x_values_new = [1, 10, 100, 1000, 10000]

# Create the plot
fig, ax = plt.subplots(figsize=(5, 4))
ax.errorbar(x_values_new, means_new, yerr=errors_new, fmt='-o', capsize=5, elinewidth=2, markeredgewidth=2)

ax.set_xlabel('scaling factor $s$', fontweight='bold', fontsize='x-large')
ax.set_ylabel('Accuracy', fontweight='bold', fontsize='x-large')

# Bold tick labels
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontweight('bold')
    label.set_fontsize('large')

# Adding specific legend title and positioning it under the line
legend = ax.legend(['VGGNet + CIFAR-100'], fontsize='medium', loc='lower right', bbox_to_anchor=(0.95, 0.75), fancybox=True, shadow=True)
for text in legend.get_texts():
    text.set_fontweight('bold')

# Bold plot edges
for spine in ax.spines.values():
    spine.set_linewidth(2)

plt.xscale('log')  # Setting x-axis to logarithmic scale
plt.grid(True)

plt.tight_layout()
# Save the figure as a PDF
pdf_path = './VGGNet_CIFAR-100_Plot.pdf'
plt.savefig(pdf_path, format='pdf')

plt.show()