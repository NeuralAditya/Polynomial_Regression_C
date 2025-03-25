import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set professional style
sns.set(style="whitegrid", palette="muted", font_scale=1.2)
plt.figure(figsize=(10, 6), dpi=300)

# Load predictions
df = pd.read_csv("predictions.csv")

# Create plot
ax = sns.scatterplot(x='x', y='y_true', data=df, 
                    label='True Data', alpha=0.7, 
                    color='#2b8cbe', edgecolor='w', s=80)
sns.lineplot(x='x', y='y_pred', data=df, 
             label='Regression Line', 
             color='#e41a1c', linewidth=2.5)

# Add equation annotation (from your results)
equation = r'$y = 0.851x + 1.481$' + '\n' + r'$R^2 = {:.3f}$'.format(
    df['y_pred'].corr(df['y_true'])**2)
plt.annotate(equation, xy=(0.05, 0.85), xycoords='axes fraction',
             fontsize=12, bbox=dict(boxstyle="round", alpha=0.8, facecolor='white'))

# Customize axes and title
plt.xlabel('Feature (X)', fontweight='bold')
plt.ylabel('Target (y)', fontweight='bold')
plt.title('Linear Regression Results\n(C Implementation)', 
          fontsize=14, pad=20, fontweight='bold')

# Add grid and legend
plt.grid(True, alpha=0.3)
plt.legend(frameon=True, framealpha=0.9)

# Save high-quality image
plt.tight_layout()
plt.savefig("regression_plot.png", bbox_inches='tight', dpi=300)
plt.show()