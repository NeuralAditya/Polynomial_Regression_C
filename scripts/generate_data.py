import numpy as np
import pandas as pd

# Set parameters
np.random.seed(42)
degree = 3  # Change this for different polynomial degrees

# Generate synthetic polynomial data
X = np.linspace(0, 10, 100)
y = 0.5 * X**degree - 2.0 * X**(degree - 1) + 1.5 * X + 2  # Polynomial equation
y += np.random.normal(scale=3.0, size=100)  # Add noise

# Save to CSV
pd.DataFrame({"x": X, "y": y}).to_csv("../data/synthetic.csv", index=False)
print(f"Generated polynomial data (degree {degree}) to ../data/synthetic.csv")
