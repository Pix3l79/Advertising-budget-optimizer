import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

# Example data
c = np.array([5, 8, 3, 6])        # Cost per unit for 4 media channels
r = np.array([100, 150, 70, 90])   # Reach per unit for 4 media channels
R_target = 10000                  # Target total reach

B = 8000                          # Total budget
s = np.array([3000, 2500, 2000, 1800])  # Max spend per media

m = [50, 30, 40, 15]  # Minimum units for channel 1 and 2

# Media channel names (for plotting)
channels = ['TV', 'Online', 'Radio', 'Newspaper']

# Variables
x = cp.Variable(len(c))  # Number of units to buy for each media

# Objective
objective = cp.Minimize(c @ x)

# Constraints
constraints = [
    r @ x >= R_target,    # Reach constraint
    c @ x <= B,           # Total budget constraint
    c[0] * x[0] <= s[0],  # Individual spend limits
    c[1] * x[1] <= s[1],
    c[2] * x[2] <= s[2],
    c[3] * x[3] <= s[3],
    x[0] >= m[0],         # Minimum units for first two channels
    x[1] >= m[1],
    x[2] >= m[2],
    x[3] >= m[3],
    x >= 0                # Non-negativity
]

# Problem setup
problem = cp.Problem(objective, constraints)

# Solve
problem.solve()

# Results
print("Optimal number of units to purchase:", x.value)
print("Minimum total cost:", problem.value)

# Compute Money Spent
spend = c * x.value

# Plotting: Bar (Money) + Line (Units) in one plot
fig, ax1 = plt.subplots(figsize=(10,6))

# Bar plot for Money Spent
bars = ax1.bar(channels, spend, color='skyblue', edgecolor='black')
ax1.set_ylabel('Money Spent ($)', color='blue', fontsize=12)
ax1.tick_params(axis='y', labelcolor='blue')

# Secondary y-axis for Units Purchased
ax2 = ax1.twinx()
ax2.plot(channels, x.value, color='red', marker='o', linestyle='-', linewidth=2, markersize=8)
ax2.set_ylabel('Units Purchased', color='red', fontsize=12)
ax2.tick_params(axis='y', labelcolor='red')

# Title and grid
plt.title("Money Spent and Units Purchased per Channel")
ax1.grid(axis='y', linestyle='--', alpha=0.7)

plt.show()