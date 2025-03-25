import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path

# Set paths
project_root = Path(__file__).parent.parent
csv_path = project_root / 'data' / 'predictions.csv'
output_path = project_root / 'docs' / 'regression_plot.png'

# Load data
try:
    df = pd.read_csv(csv_path)
except FileNotFoundError:
    print(f"Error: File not found at {csv_path}")
    print("Please run the C program first to generate predictions.csv")
    exit(1)

# Create plot with modern style
plt.style.use('default')  # Reset to default first
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.edgecolor': '.8',
    'axes.labelcolor': '.15',
    'xtick.color': '.15',
    'ytick.color': '.15'
})

fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

# Plot data
ax.scatter(df['x'], df['y_true'], 
           color='#2b8cbe', edgecolor='white',
           s=100, label='True Data', alpha=0.7)
ax.plot(df['x'], df['y_pred'], 
        color='#e41a1c', linewidth=2.5,
        label='Regression Fit')

# Add equation annotation
equation = r'$y = 0.851x + 1.481$' + '\n' + r'$R^2 = {:.3f}$'.format(
    df['y_pred'].corr(df['y_true'])**2)
ax.annotate(equation, xy=(0.05, 0.85), xycoords='axes fraction',
            fontsize=12, bbox=dict(boxstyle="round", alpha=0.8, facecolor='white'))

# Customize plot
ax.set_xlabel('Input Feature (X)', fontweight='bold')
ax.set_ylabel('Target Value (y)', fontweight='bold')
ax.set_title('Linear Regression Results\n(C Implementation)', 
             fontsize=14, pad=20, fontweight='bold')
ax.legend(frameon=True, framealpha=0.9)

# Save and show
plt.tight_layout()
output_path.parent.mkdir(exist_ok=True)
plt.savefig(output_path, bbox_inches='tight')
print(f"Plot saved to {output_path}")
plt.show()