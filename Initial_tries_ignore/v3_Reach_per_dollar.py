import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# -------------------------
# 📜 Load media parameters and budget dynamically from CSV
try:
    media_df = pd.read_csv('media_parameters.csv', comment='#')
except FileNotFoundError:
    raise FileNotFoundError("The file 'media_parameters.csv' was not found. Please make sure it exists in the working directory.")

# Check if all required columns are present
required_columns = ['Channel', 'Cost_per_Unit', 'Reach_Factor', 'Saturation_Rate', 'Min_Units']
for col in required_columns:
    if col not in media_df.columns:
        raise ValueError(f"Required column '{col}' is missing in 'media_parameters.csv'.")

# Extract user budget
if '_budget_' not in media_df['Channel'].values:
    raise ValueError("The '_budget_' row is missing in your 'media_parameters.csv'. Please add it.")
    
user_budget = media_df.loc[media_df['Channel'] == '_budget_', 'Min_Units'].values[0]

# Remove _budget_ row from the media channels
media_df = media_df[media_df['Channel'] != '_budget_']

# Extract media fields
channels = media_df['Channel'].tolist()
c = media_df['Cost_per_Unit'].values
a = media_df['Reach_Factor'].values
b = media_df['Saturation_Rate'].values
m = media_df['Min_Units'].values

# -------------------------
# 🎯 Dynamic Budget Sweep
low_budget = 0.9 * user_budget
high_budget = 1.1 * user_budget
budget_values = np.linspace(low_budget, high_budget, 10)  # 10 budget points

# Find index closest to user budget
closest_idx = np.argmin(np.abs(budget_values - user_budget))

# -------------------------
# 📊 Solve Optimization for Each Budget
reach_list = []
spend_list = []
efficiency_list = []
x_solutions = []

for B in budget_values:
    x = cp.Variable(len(c))
    reach_expr = cp.sum(cp.multiply(a, cp.log(1 + cp.multiply(b, x))))
    spend_expr = c @ x

    objective = cp.Maximize(reach_expr)

    constraints = [
        spend_expr <= B,
        *(x[i] >= m[i] for i in range(len(c))),
        x >= 0
    ]

    problem = cp.Problem(objective, constraints)

    try:
        problem.solve()

        if problem.status not in ["infeasible", "unbounded"]:
            total_spend = spend_expr.value
            total_reach = reach_expr.value
            efficiency = (total_reach / total_spend) * 100  # Reach per $100

            spend_list.append(total_spend)
            reach_list.append(total_reach)
            efficiency_list.append(efficiency)
            x_solutions.append(x.value)
        else:
            spend_list.append(np.nan)
            reach_list.append(np.nan)
            efficiency_list.append(np.nan)
            x_solutions.append([np.nan]*len(c))

    except Exception as e:
        spend_list.append(np.nan)
        reach_list.append(np.nan)
        efficiency_list.append(np.nan)
        x_solutions.append([np.nan]*len(c))

# -------------------------
# 📈 Combined Plot: Reach and Efficiency vs Budget

fig, ax1 = plt.subplots(figsize=(12, 7))

color = 'tab:blue'
ax1.set_xlabel('Budget ($)', fontsize=12)
ax1.set_ylabel('Total Reach Achieved', color=color, fontsize=12)
ax1.plot(budget_values, reach_list, marker='o', linestyle='-', color=color, label='Reach (left)', markersize=8)
ax1.tick_params(axis='y', labelcolor=color)

ax1.axvline(user_budget, color='red', linestyle='--', linewidth=2, label=f'User Budget: ${user_budget:.0f}')

ax2 = ax1.twinx()
color = 'tab:green'
ax2.set_ylabel('Efficiency (Reach per $100)', color=color, fontsize=12)
ax2.plot(budget_values, efficiency_list, marker='s', linestyle='--', color=color, label='Efficiency (right)', markersize=8)
ax2.tick_params(axis='y', labelcolor=color)

plt.title("Reach and Efficiency vs Budget", fontsize=14)
fig.tight_layout()
fig.legend(loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=2, fontsize=12)
ax1.grid(True, linestyle='--', alpha=0.7)
plt.show()

# -------------------------
# 🎯 Detailed Spending for User Budget (Rounded Units)

final_units = np.ceil(x_solutions[closest_idx])
final_spend = c * final_units

print(f"\nDetailed Spending for Your Budget (${user_budget:.0f}):\n")
for i in range(len(channels)):
    print(f"{channels[i]}: {int(final_units[i])} units, ${final_spend[i]:.2f}")

# 📊 Money Spent and Units Purchased per Channel

fig, ax1 = plt.subplots(figsize=(10,6))

bars = ax1.bar(channels, final_spend, color='lightblue', edgecolor='black')
ax1.set_ylabel('Money Spent ($)', color='blue', fontsize=12)
ax1.tick_params(axis='y', labelcolor='blue')

ax2 = ax1.twinx()
ax2.plot(channels, final_units, color='red', marker='o', linestyle='-', linewidth=2, markersize=8)
ax2.set_ylabel('Units Purchased', color='red', fontsize=12)
ax2.tick_params(axis='y', labelcolor='red')

plt.title(f"Money Spent and Units Purchased per Channel (Budget: ${user_budget:.0f})", fontsize=14)
ax1.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# -------------------------
# 📋 Summary Table: Budget vs Reach vs Efficiency

summary_df = pd.DataFrame({
    'Budget ($)': np.round(budget_values, 2),
    'Total Reach Achieved': np.round(reach_list, 2),
    'Efficiency (Reach per $100)': np.round(efficiency_list, 4)
})

print("\nSummary Table (Budgets vs Reach vs Efficiency):\n")
print(summary_df.to_string(index=False))