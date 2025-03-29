import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Define paths
project_root = Path(__file__).resolve().parent.parent
csv_path = project_root / 'data' / 'predictions.csv'
output_path = project_root / 'docs' / 'regression_plot.png'

# Ensure the output directory exists
output_path.parent.mkdir(parents=True, exist_ok=True)

# Load data
try:
    df = pd.read_csv(csv_path)
    print(f"Loaded data from {csv_path}")
except FileNotFoundError:
    print(f"Error: File not found at {csv_path}")
    print("Please run the C program first to generate predictions.csv")
    exit(1)

# Ensure data is sorted by x for smooth plotting
df = df.sort_values(by="x")

# Create a modern plot
plt.style.use('ggplot')
fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

# Scatter plot of actual data
ax.scatter(df['x'], df['y_true'], color='blue', label='True Data', edgecolor='white', s=100, alpha=0.7)

# Polynomial regression fit
ax.plot(df['x'], df['y_pred'], color='red', linewidth=2, label='Polynomial Regression Fit')

# Fit polynomial equation dynamically
degree = min(len(df) - 1, 5)  # Limit polynomial degree to avoid overfitting
coeffs = np.polyfit(df['x'], df['y_pred'], deg=degree)
poly_eq = " + ".join([f"{coeff:.3f}x^{i}" for i, coeff in enumerate(coeffs[::-1])])

# Display equation and R^2 value
r_squared = df['y_pred'].corr(df['y_true'])**2
annotation_text = rf'$y = {poly_eq}$' + f'\n$R^2 = {r_squared:.3f}$'
ax.annotate(annotation_text, xy=(0.05, 0.85), xycoords='axes fraction', fontsize=12, 
            bbox=dict(boxstyle="round", facecolor='white', alpha=0.8))

# Labels and title
ax.set_xlabel('Input Feature (X)', fontsize=12, fontweight='bold')
ax.set_ylabel('Target Value (y)', fontsize=12, fontweight='bold')
ax.set_title('Polynomial Regression Results (C Implementation)', fontsize=14, fontweight='bold', pad=20)
ax.legend()

# Save and show plot
plt.tight_layout()
plt.savefig(output_path, bbox_inches='tight')
print(f"Plot saved to {output_path}")
plt.show()
