# %% [markdown]
# # Bayesian Optimization

# %%
import pandas as pd
import numpy as np

# %%
df = pd.read_csv("/kaggle/input/intel-challenge/function_data.csv")

# %%
df

# %%
X = df[list(df.columns)[:-1]]

# %%
y = df['y']

# %%
from skopt import Optimizer
from skopt.space import Real

# %%
n = 750  # Starting with 750 points
X_initial = X[:n]
y_initial = y[:n]

# %%
space = [Real(X[col].min(), X[col].max(), name=col) for col in X.columns]

# %%
opt = Optimizer(
    dimensions=space, 
    base_estimator="GP",  # Gaussian Process
    acq_func="EI"  # Expected Improvement
)

# %%
opt.tell(X_initial.values.tolist(), y_initial.values.tolist())

# %%
used_indices = set(range(n))  # Initially, the first `n` indices are used

def find_closest_unselected_point(point):
    # Compute distances to all datapoints
    distances = np.linalg.norm(X.values - np.array(point), axis=1)
    # Exclude already used indices
    for idx in sorted(used_indices):
        distances[idx] = np.inf
    # Find the nearest unselected point
    nearest_idx = np.argmin(distances)
    used_indices.add(nearest_idx)  # Mark this point as used
    return X.iloc[nearest_idx].values.tolist(), y.iloc[nearest_idx]

# %%
from tqdm import tqdm

n_calls = 100  # Number of optimization steps

# Wrap the loop with tqdm to show progress
for _ in tqdm(range(n_calls), desc="Optimization Steps", ncols=100):
    # Get the next suggested point
    next_point = opt.ask()

    # Find the closest datapoint in the dataset
    sampled_point, f_val = find_closest_unselected_point(next_point)

    # Update the optimizer with the new point
    opt.tell(sampled_point, f_val)

# %%
best_x = opt.Xi[opt.yi.index(min(opt.yi))]  # Best parameter set
best_f = min(opt.yi)  # Minimum function value

print("Best parameters:", best_x)
print("Best function value:", best_f)

# %%



