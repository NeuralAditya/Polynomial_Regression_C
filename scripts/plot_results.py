import pandas as pd
import matplotlib.pyplot as plt

# Load predictions from CSV
df = pd.read_csv("predictions.csv")

# Plot true vs predicted values
plt.scatter(df['x'], df['y_true'], label='True Data', color='blue')
plt.plot(df['x'], df['y_pred'], label='Regression Line', color='red')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.title('Linear Regression Results')
plt.savefig("regression_plot.png")  # Saves the plot as PNG
plt.show()