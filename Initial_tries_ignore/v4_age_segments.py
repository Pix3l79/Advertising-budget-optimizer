import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# -------------------------
# 📜 Load media parameters and budget dynamically from CSV
try:
    media_df = pd.read_csv('media_parameters_segments.csv', comment='#')
except FileNotFoundError:
    raise FileNotFoundError("The file 'media_parameters_segments.csv' was not found. Please make sure it exists in the working directory.")

# Check if all required columns are present
required_columns = ['Channel', 'Cost_per_Unit', 'Total_Reach', 'Percent_12_20', 'Percent_21_35', 'Percent_36_above', 'Saturation_Rate', 'Min_Units']
for col in required_columns:
    if col not in media_df.columns:
        raise ValueError(f"Required column '{col}' is missing in 'media_parameters_segments.csv'.")

# Extract user budget
if '_budget_' not in media_df['Channel'].values:
    raise ValueError("The '_budget_' row is missing in your 'media_parameters_segments.csv'. Please add it.")

user_budget = media_df.loc[media_df['Channel'] == '_budget_', 'Min_Units'].values[0]

# Remove _budget_ row from media channels
media_df = media_df[media_df['Channel'] != '_budget_']

# Extract fields
channels = media_df['Channel'].tolist()
c = media_df['Cost_per_Unit'].values
a = media_df['Total_Reach'].values
b = media_df['Saturation_Rate'].values
m = media_df['Min_Units'].values

# -------------------------
# 🎯 Dynamic Budget Sweep
low_budget = 0.9 * user_budget
high_budget = 1.1 * user_budget
budget_values = np.linspace(low_budget, high_budget, 10)

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
ax1.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# -------------------------
# 🎯 Detailed Spending for Your Budget (Rounded Units)

final_units = np.ceil(x_solutions[closest_idx])
final_spend = c * final_units

print(f"\nDetailed Spending for Your Budget (${user_budget:.0f}):\n")
for i in range(len(channels)):
    print(f"{channels[i]}: {int(final_units[i])} units, ${final_spend[i]:.2f}")

# -------------------------
# 📋 Summary Table: Budget vs Reach vs Efficiency

summary_df = pd.DataFrame({
    'Budget ($)': np.round(budget_values, 2),
    'Total Reach Achieved': np.round(reach_list, 2),
    'Efficiency (Reach per $100)': np.round(efficiency_list, 4)
})

print("\nSummary Table (Budgets vs Reach vs Efficiency):\n")
print(summary_df.to_string(index=False))

# -------------------------
# 🎯 Segment-wise Reach Breakdown After Optimization

# Extract percentage splits
p_12_20 = media_df['Percent_12_20'].values
p_21_35 = media_df['Percent_21_35'].values
p_36_above = media_df['Percent_36_above'].values

# Calculate total reach per channel after optimization
final_total_reach_per_channel = a * np.log(1 + b * final_units)

# Segment-wise reach per channel
reach_12_20 = final_total_reach_per_channel * (p_12_20 / 100)
reach_21_35 = final_total_reach_per_channel * (p_21_35 / 100)
reach_36_above = final_total_reach_per_channel * (p_36_above / 100)

# Sum across channels
total_reach_12_20 = np.sum(reach_12_20)
total_reach_21_35 = np.sum(reach_21_35)
total_reach_36_above = np.sum(reach_36_above)
total_reach_all_segments = total_reach_12_20 + total_reach_21_35 + total_reach_36_above

# 🎯 Report
print("\nSegment-wise Reach Breakdown (after Optimization):")
print(f"12-20 years : {total_reach_12_20:.2f} ({(total_reach_12_20/total_reach_all_segments)*100:.2f}%)")
print(f"21-35 years : {total_reach_21_35:.2f} ({(total_reach_21_35/total_reach_all_segments)*100:.2f}%)")
print(f"36 and above: {total_reach_36_above:.2f} ({(total_reach_36_above/total_reach_all_segments)*100:.2f}%)")

# 📊 Pie Chart of Segment-Wise Reach
labels = ['12-20 years', '21-35 years', '36 and above']
sizes = [total_reach_12_20, total_reach_21_35, total_reach_36_above]

plt.figure(figsize=(8, 8))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=['#66c2a5','#fc8d62','#8da0cb'])
plt.title('Segment-wise Reach Distribution')
plt.axis('equal')
plt.show()

# -------------------------
# 📊 Stacked Bar + Line Plot: Units Split by Segment + Money Spent

# Compute segment proportions for units
segment_prop_12_20 = (p_12_20 / 100)
segment_prop_21_35 = (p_21_35 / 100)
segment_prop_36_above = (p_36_above / 100)

units_12_20 = final_units * segment_prop_12_20
units_21_35 = final_units * segment_prop_21_35
units_36_above = final_units * segment_prop_36_above

fig, ax1 = plt.subplots(figsize=(12,8))

# --- Stacked Bar for Units Purchased split by segment ---
bars_12_20 = ax1.bar(channels, units_12_20, label='12-20 years', color='#66c2a5')
bars_21_35 = ax1.bar(channels, units_21_35, bottom=units_12_20, label='21-35 years', color='#fc8d62')
bars_36_above = ax1.bar(channels, units_36_above, bottom=units_12_20 + units_21_35, label='36 and above', color='#8da0cb')

ax1.set_xlabel('Channel', fontsize=12)
ax1.set_ylabel('Units Purchased', color='black', fontsize=12)
ax1.tick_params(axis='y', labelcolor='black')

# --- Line plot for Money Spent on second y-axis ---
ax2 = ax1.twinx()
ax2.plot(channels, final_spend, color='red', marker='o', linestyle='-', linewidth=2, markersize=8, label='Money Spent ($)')
ax2.set_ylabel('Money Spent ($)', color='red', fontsize=12)
ax2.tick_params(axis='y', labelcolor='red')

# Titles and combined legends
handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
all_handles = handles1 + handles2
all_labels = labels1 + labels2

fig.legend(all_handles, all_labels, loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=4, fontsize=11)
plt.title('Units Purchased Split by Audience Segments and Money Spent per Channel', fontsize=16)
fig.tight_layout()
ax1.grid(axis='y', linestyle='--', alpha=0.7)

plt.show()