import numpy as np
import pandas as pd

# Generate synthetic linear data with noise
np.random.seed(42)
X = np.linspace(0, 10, 100)
y = 1.5 * X + 2 + np.random.normal(scale=1.5, size=100)

pd.DataFrame({"x": X, "y": y}).to_csv("../data/synthetic.csv", index=False)
print("Generated data to ../data/synthetic.csv")