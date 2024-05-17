import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

# Sample data with standard deviation values
data = {
    "K=1": ["92.53 ± 0.11", "92.72 ± 0.09", "95.01 ± 0.13", "76.45 ± 0.15", "93.02 ± 0.20", "71.09 ± 0.27"],
    "K=2": ["92.62 ± 0.13", "92.73 ± 0.13", "95.03 ± 0.18", "76.62 ± 0.12", "93.21 ± 0.14", "71.18 ± 0.11"],
    "K=4": ["92.77 ± 0.17", "92.67 ± 0.14", "95.02 ± 0.10", "77.04 ± 0.34", "93.24 ± 0.09", "71.83 ± 0.32"],
    "K=6": ["92.86 ± 0.07", "92.62 ± 0.12", "95.06 ± 0.12", "76.66 ± 0.24", "93.34 ± 0.19", "72.16 ± 0.10"],
    "K=8": ["92.83 ± 0.19", "92.55 ± 0.05", "95.05 ± 0.10", "77.05 ± 0.41", "93.25 ± 0.11", "72.09 ± 0.17"],
    "K=10": ["93.01 ± 0.07", "92.23 ± 0.09", "95.00 ± 0.14", "76.87 ± 0.40", "93.30 ± 0.10", "72.26 ± 0.18"]
}
index = [
    "ResNet18 + CIFAR-10 (1W1A)",
    "VGG-Small + CIFAR-10 (1W1A)",
    "ResNet18 + CIFAR-10 (1W32A)",
    "ResNet18 + CIFAR-100 (1W32A)",
    "VGG16 + CIFAR-10 (1W32A)",
    "VGG16 + CIFAR-100 (1W32A)"
]

# index = {"Settings": [
#     "ResNet18 + CIFAR-10 (1W1A)",
#     "VGG-Small + CIFAR-10 (1W1A)",
#     "ResNet18 + CIFAR-10 (1W32A)",
#     "ResNet18 + CIFAR-100 (1W32A)",
#     "VGG16 + CIFAR-10 (1W32A)",
#     "VGG16 + CIFAR-100 (1W32A)"
# ]}

# Extract numerical values for coloring
num_data = {
    "K=1": [92.53, 92.72, 95.01, 76.45, 93.02, 71.09],
    "K=2": [92.62, 92.73, 95.03, 76.62, 93.21, 71.18],
    "K=4": [92.77, 92.67, 95.02, 77.04, 93.24, 71.83],
    "K=6": [92.86, 92.62, 95.06, 76.66, 93.34, 72.16],
    "K=8": [92.83, 92.55, 95.05, 77.05, 93.25, 72.09],
    "K=10": [93.01, 92.23, 95.00, 76.87, 93.30, 72.26]
}

df = pd.DataFrame(data, index=index)
num_df = pd.DataFrame(num_data, index=index)

# Define a more distinguishable colormap with 6 colors
# colors = ["#ffffcc", "#ffeb99", "#ffcc66", "#ffcc00", "#ff9933", "#ff6600"]
# colors = ["#ffffe0", "#fffacd", "#ffeb99", "#ffdd77", "#ffcc66", "#ffbb33"]

colors = ["#ffffe0", "#fffacd", "#ffeb99", "#ffdd57", "#ffc300", "#ff9900"]

# Create the plot
fig, ax = plt.subplots(figsize=(18, 8))

# Hide axes
ax.axis('tight')
ax.axis('off')

# Create table
table = ax.table(cellText=df.values, rowLabels=df.index, colLabels=df.columns, 
                 cellLoc='center', loc='center')

# Set font size
table.auto_set_font_size(False)
table.set_fontsize(20)

# Apply color to each cell, comparing values only within the same row
for i in range(len(df.index)):
    sorted_indices = num_df.iloc[i].argsort().values
    for color_index, j in enumerate(sorted_indices):
        cell = table[i+1, j]
        cell.set_facecolor(colors[color_index])

# Adjust column and row sizes
for (row, col), cell in table.get_celld().items():
    if col == -1:
        cell.set_width(0.05)
    else:
        cell.set_width(0.1)
    cell.set_height(0.07)
    if row == 0 or col == -1:
        cell.set_text_props(weight='bold')

table.scale(1.5, 1.5)
ax.set_position([0.3, 0.1, 0.6, 0.8])  # [left, bottom, width, height]


# Get the bounding box of the table
bbox = table.get_window_extent(fig.canvas.get_renderer())
table_width = bbox.width / fig.dpi
table_height = bbox.height / fig.dpi

# Adjust the figure size to fit the table
fig.set_size_inches(table_width + 6, table_height + 2)

# Save the table as a PDF
plt.savefig("acc_vs_k.pdf", format='pdf')

# Display the table
plt.show()