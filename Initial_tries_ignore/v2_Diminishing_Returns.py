import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

# Example data
c = np.array([5, 8, 3, 6])  # Cost per unit
a = np.array([100, 150, 70, 90])  # Reach scaling factors
b = np.array([0.01, 0.02, 0.015, 0.012])  # Saturation rates
R_target = 120  # target reach (not used anymore)

B = 120000  # total budget
s = np.array([5000, 4000, 4000, 3000])  # Relaxed per-channel spend limits
m = [5, 3, 4, 10]  # Minimum units for channels
channels = ['TV', 'Online', 'Radio', 'Newspaper']

# Variables
x = cp.Variable(len(c))

# Diminishing returns: log(1 + b_i * x_i)
reach_expr = cp.sum(cp.multiply(a, cp.log(1 + cp.multiply(b, x))))

# Objective
objective = cp.Maximize(reach_expr)

# Constraints
constraints = [
    c @ x <= B,
    c[0] * x[0] <= s[0],
    c[1] * x[1] <= s[1],
    c[2] * x[2] <= s[2],
    c[3] * x[3] <= s[3],
    x[0] >= m[0],
    x[1] >= m[1],
    x[2] >= m[2],
    x[3] >= m[3],
    x >= 0
]

# Problem setup
problem = cp.Problem(objective, constraints)

# Solve
result = problem.solve()

# Check if solvable
if problem.status in ["infeasible", "unbounded"]:
    print(f"Problem status: {problem.status}. No feasible solution found with the given constraints.")
else:
    print("Optimal number of units to purchase:", x.value)
    print("Maximum total reach achieved:", problem.value)

    # Compute Money Spent
    spend = c * x.value

    # ---- FIRST PLOT: Money Spent and Units Purchased ----
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

    # ---- SECOND PLOT: Diminishing Returns Curve ----
    units_range = np.linspace(0, 500, 500)  # Smooth range of units for plotting

    plt.figure(figsize=(12, 8))

    for i in range(len(channels)):
        reach_curve = a[i] * np.log(1 + b[i] * units_range)
        plt.plot(units_range, reach_curve, label=f"{channels[i]}")

    plt.title("Diminishing Returns: Reach vs Units Purchased")
    plt.xlabel("Units Purchased")
    plt.ylabel("Reach Achieved")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()