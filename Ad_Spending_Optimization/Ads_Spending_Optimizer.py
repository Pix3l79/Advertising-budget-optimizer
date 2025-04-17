import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def load_and_validate_data():
    """
    Load media parameters and target audience/region prioritization data.
    Validate that the data meets requirements.
    """
    # Load media parameters
    try:
        media_df = pd.read_csv('real_life_media_parameters.csv', comment='#')
    except FileNotFoundError:
        raise FileNotFoundError("The file 'real_life_media_parameters.csv' was not found.")

    required_columns = ['Channel', 'Cost_per_Unit', 'Total_Reach', 'Percent_12_20', 'Percent_21_35', 
                        'Percent_36_above', 'Saturation_Rate', 'Min_Units',
                        'Metro_Share', 'Suburban_Share', 'Rural_Share', 'National_Share']
    for col in required_columns:
        if col not in media_df.columns:
            raise ValueError(f"Required column '{col}' missing in real_life_media_parameters.csv.")

    # Load target audience prioritization
    try:
        target_df = pd.read_csv('target_prioritization.csv', comment='#')
    except FileNotFoundError:
        raise FileNotFoundError("The file 'target_prioritization.csv' was not found.")

    target_df.columns = target_df.columns.str.strip()
    if not np.isclose(target_df['Importance Weight'].sum(), 1.0):
        raise ValueError("The sum of Importance Weights in 'target_prioritization.csv' must be 1.0.")

    # Load region prioritization
    try:
        region_df = pd.read_csv('region_prioritization.csv', comment='#')
    except FileNotFoundError:
        raise FileNotFoundError("The file 'region_prioritization.csv' was not found.")

    region_df.columns = region_df.columns.str.strip()
    region_df['Region Name'] = region_df['Region Name'].str.strip()
    if not np.isclose(region_df['Importance Weight'].sum(), 1.0):
        raise ValueError("The sum of Importance Weights in 'region_prioritization.csv' must be 1.0.")

    return media_df, target_df, region_df


def extract_parameters(media_df, target_df, region_df):
    """
    Extract all necessary parameters from the dataframes.
    """
    # Extract budget
    user_budget = media_df.loc[media_df['Channel'] == '_budget_', 'Min_Units'].values[0]
    media_df = media_df[media_df['Channel'] != '_budget_']

    # Extract media parameters
    channels = media_df['Channel'].tolist()
    c = media_df['Cost_per_Unit'].values
    a = media_df['Total_Reach'].values
    b = media_df['Saturation_Rate'].values
    m = media_df['Min_Units'].values
    p_12_20 = media_df['Percent_12_20'].values
    p_21_35 = media_df['Percent_21_35'].values
    p_36_above = media_df['Percent_36_above'].values

    metro_share = media_df['Metro_Share'].values / 100
    suburban_share = media_df['Suburban_Share'].values / 100
    rural_share = media_df['Rural_Share'].values / 100
    national_share = media_df['National_Share'].values / 100

    # Extract target audience weights
    weight_12_20 = target_df.loc[target_df['Segment Name'] == '12-20 years', 'Importance Weight'].values[0]
    weight_21_35 = target_df.loc[target_df['Segment Name'] == '21-35 years', 'Importance Weight'].values[0]
    weight_36_above = target_df.loc[target_df['Segment Name'] == '36 and above', 'Importance Weight'].values[0]

    # Extract region weights
    region_weights = {
        'Metro': region_df.loc[region_df['Region Name'] == 'Metro', 'Importance Weight'].values[0],
        'Suburban': region_df.loc[region_df['Region Name'] == 'Suburban', 'Importance Weight'].values[0],
        'Rural': region_df.loc[region_df['Region Name'] == 'Rural', 'Importance Weight'].values[0],
        'National': region_df.loc[region_df['Region Name'] == 'National', 'Importance Weight'].values[0]
    }

    params = {
        'user_budget': user_budget,
        'channels': channels,
        'c': c,  # Cost per unit
        'a': a,  # Total reach factor
        'b': b,  # Saturation rate
        'm': m,  # Minimum units
        'p_12_20': p_12_20,
        'p_21_35': p_21_35,
        'p_36_above': p_36_above,
        'metro_share': metro_share,
        'suburban_share': suburban_share,
        'rural_share': rural_share,
        'national_share': national_share,
        'weight_12_20': weight_12_20,
        'weight_21_35': weight_21_35,
        'weight_36_above': weight_36_above,
        'region_weights': region_weights
    }
    
    return params, media_df


def solve_optimization(budget, params):
    """
    Solve the optimization problem for a given budget.
    
    Mathematical Model:
    Objective: Maximize total weighted reach across segments and regions
    
    Variables:
    - x[i]: Number of units to purchase for each channel i
    
    Key Equations:
    - Base reach per channel: a * log(1 + b*x)
      where a = reach factor, b = saturation rate
    - Segment reach: base_reach * segment_percentage
    - Region reach: segment_reach * region_share
    - Total weighted reach: sum(weight_segment * weight_region * reach)
    
    Constraints:
    1. Total spend <= budget
    2. Units >= minimum required units
    3. All units >= 0
    """
    # Extract parameters from dictionary
    c = params['c']  # Cost per unit vector
    a = params['a']  # Total reach factor vector
    b = params['b']  # Saturation rate vector
    m = params['m']  # Minimum units vector
    
    # Segment percentages (per 100)
    p_12_20 = params['p_12_20']     # Age group 12-20 years
    p_21_35 = params['p_21_35']     # Age group 21-35 years
    p_36_above = params['p_36_above']  # Age group 36+ years
    
    # Regional share vectors (as decimals)
    metro_share = params['metro_share']
    suburban_share = params['suburban_share']
    rural_share = params['rural_share']
    national_share = params['national_share']
    
    # Importance weights for segments (must sum to 1)
    weight_12_20 = params['weight_12_20']
    weight_21_35 = params['weight_21_35']
    weight_36_above = params['weight_36_above']
    
    # Region importance weights (must sum to 1)
    region_weights = params['region_weights']

    # Define decision variables
    x = cp.Variable(len(c))
    
    # Base reach calculation using logarithmic saturation model
    # This captures diminishing returns as more units are purchased
    base_reach = cp.multiply(a, cp.log1p(cp.multiply(b, x)))

    # Calculate reach for each age segment
    # Multiply base reach by the percentage of audience in each segment
    reach_expr_12_20 = cp.multiply(base_reach, p_12_20 / 100)
    reach_expr_21_35 = cp.multiply(base_reach, p_21_35 / 100)
    reach_expr_36_above = cp.multiply(base_reach, p_36_above / 100)

    # Apply segment importance weights to get weighted segment reach
    weighted_segment_reach = (weight_12_20 * reach_expr_12_20 +
                            weight_21_35 * reach_expr_21_35 +
                            weight_36_above * reach_expr_36_above)

    # Calculate region-weighted reach by combining:
    # 1. Segment-weighted reach
    # 2. Regional share of each channel
    # 3. Region importance weights
    weighted_region_segment_reach = (
        region_weights['Metro'] * cp.multiply(weighted_segment_reach, metro_share) +
        region_weights['Suburban'] * cp.multiply(weighted_segment_reach, suburban_share) +
        region_weights['Rural'] * cp.multiply(weighted_segment_reach, rural_share) +
        region_weights['National'] * cp.multiply(weighted_segment_reach, national_share)
    )

    # Final objective: sum of all weighted reach across channels
    total_reach_expr = cp.sum(weighted_region_segment_reach)
    
    # Total spend calculation: dot product of cost vector and units
    spend_expr = c @ x

    # Define optimization problem
    objective = cp.Maximize(total_reach_expr)
    constraints = [
        spend_expr <= budget,                    # Budget constraint
        *(x[i] >= m[i] for i in range(len(c))), # Minimum units constraints
        x >= 0                                   # Non-negativity constraint
    ]

    # Solve the problem
    problem = cp.Problem(objective, constraints)
    
    try:
        problem.solve()
        
        if problem.status in ["infeasible", "unbounded"]:
            return {
                'status': problem.status,
                'spend': np.nan,
                'reach': np.nan,
                'efficiency': np.nan,
                'solution': np.array([np.nan] * len(c))
            }
        
        total_spend = spend_expr.value
        total_reach = total_reach_expr.value
        efficiency = (total_reach / total_spend) * 100
        
        return {
            'status': problem.status,
            'spend': total_spend,
            'reach': total_reach,
            'efficiency': efficiency,
            'solution': x.value
        }
    
    except Exception as e:
        print(f"Optimization error: {str(e)}")
        return {
            'status': 'error',
            'spend': np.nan,
            'reach': np.nan,
            'efficiency': np.nan,
            'solution': np.array([np.nan] * len(c))
        }


def budget_sensitivity_analysis(params, budget_range=0.1):
    """
    Perform budget sensitivity analysis by solving the optimization
    problem at different budget levels.
    """
    user_budget = params['user_budget']
    low_budget = (1 - budget_range) * user_budget
    high_budget = (1 + budget_range) * user_budget
    budget_values = np.linspace(low_budget, high_budget, 10)
    closest_idx = np.argmin(np.abs(budget_values - user_budget))
    
    results = []
    for budget in budget_values:
        result = solve_optimization(budget, params)
        results.append(result)
    
    return {
        'budget_values': budget_values,
        'results': results,
        'closest_idx': closest_idx
    }


def plot_budget_sensitivity(sensitivity_results):
    """
    Plot the reach and efficiency vs budget.
    """
    budget_values = sensitivity_results['budget_values']
    results = sensitivity_results['results']
    user_budget = budget_values[sensitivity_results['closest_idx']]
    
    reach_list = [r['reach'] for r in results]
    efficiency_list = [r['efficiency'] for r in results]
    
    fig, ax1 = plt.subplots(figsize=(12, 7))

    # Reach plot (primary y-axis)
    color = 'tab:blue'
    ax1.set_xlabel('Budget ($)', fontsize=12)
    ax1.set_ylabel('Total Reach Achieved', color=color, fontsize=12)
    ax1.plot(budget_values, reach_list, marker='o', linestyle='-', color=color, label='Reach (left)', markersize=8)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.axvline(user_budget, color='red', linestyle='--', linewidth=2, label=f'User Budget: ${user_budget:.0f}')

    # Efficiency plot (secondary y-axis)
    ax2 = ax1.twinx()
    color = 'tab:green'
    ax2.set_ylabel('Efficiency (Reach per $100)', color=color, fontsize=12)
    ax2.plot(budget_values, efficiency_list, marker='s', linestyle='--', color=color, label='Efficiency (right)', markersize=8)
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title("Reach and Efficiency vs Budget", fontsize=14)
    fig.tight_layout()
    fig.legend(loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=2, fontsize=12)
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    return fig


def print_detailed_spending(channels, final_units, c, user_budget):
    """
    Print detailed spending breakdown for the optimal solution.
    """
    final_spend = c * final_units
    
    print(f"\nDetailed Spending for Your Budget (${user_budget:.0f}):\n")
    for i in range(len(channels)):
        units = 0 if np.isnan(final_units[i]) else int(final_units[i])
        spend = 0 if np.isnan(final_spend[i]) else final_spend[i]
        print(f"{channels[i]}: {units} units, ${spend:.2f}")
    
    return final_spend


def print_summary_table(budget_values, results):
    """
    Print summary table of budget vs reach vs efficiency.
    """
    summary_df = pd.DataFrame({
        'Budget ($)': np.round(budget_values, 2),
        'Total Reach Achieved': [np.round(r['reach'], 2) if not np.isnan(r['reach']) else "N/A" for r in results],
        'Efficiency (Reach per $100)': [np.round(r['efficiency'], 4) if not np.isnan(r['efficiency']) else "N/A" for r in results]
    })
    
    print("\nSummary Table (Budgets vs Reach vs Efficiency):\n")
    print(summary_df.to_string(index=False))
    
    return summary_df


def calculate_segment_reach(media_df, final_units):
    """
    Calculate segment-wise reach breakdown after optimization.
    """
    p_12_20 = media_df['Percent_12_20'].values
    p_21_35 = media_df['Percent_21_35'].values
    p_36_above = media_df['Percent_36_above'].values
    a = media_df['Total_Reach'].values
    b = media_df['Saturation_Rate'].values
    
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
    
    segment_reach = {
        '12-20 years': total_reach_12_20,
        '21-35 years': total_reach_21_35,
        '36 and above': total_reach_36_above,
        'total': total_reach_all_segments
    }
    
    print("\nSegment-wise Reach Breakdown (after Optimization):")
    print(f"12-20 years : {total_reach_12_20:.2f} ({(total_reach_12_20/total_reach_all_segments)*100:.2f}%)")
    print(f"21-35 years : {total_reach_21_35:.2f} ({(total_reach_21_35/total_reach_all_segments)*100:.2f}%)")
    print(f"36 and above: {total_reach_36_above:.2f} ({(total_reach_36_above/total_reach_all_segments)*100:.2f}%)")
    
    return segment_reach


def plot_segment_pie_chart(segment_reach):
    """
    Create a pie chart of segment-wise reach distribution.
    """
    labels = ['12-20 years', '21-35 years', '36 and above']
    sizes = [segment_reach['12-20 years'], segment_reach['21-35 years'], segment_reach['36 and above']]
    
    plt.figure(figsize=(8, 8))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=['#66c2a5','#fc8d62','#8da0cb'])
    plt.title('Segment-wise Reach Distribution')
    plt.axis('equal')
    
    return plt.gcf()


def plot_stacked_bar_chart(channels, final_units, final_spend, media_df):
    """
    Create a stacked bar chart of units split by segment and money spent.
    """
    # Compute segment proportions for units
    p_12_20 = media_df['Percent_12_20'].values / 100
    p_21_35 = media_df['Percent_21_35'].values / 100
    p_36_above = media_df['Percent_36_above'].values / 100
    
    units_12_20 = final_units * p_12_20
    units_21_35 = final_units * p_21_35
    units_36_above = final_units * p_36_above
    
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    # Stacked bar for units purchased split by segment
    bars_12_20 = ax1.bar(channels, units_12_20, label='12-20 years', color='#66c2a5')
    bars_21_35 = ax1.bar(channels, units_21_35, bottom=units_12_20, label='21-35 years', color='#fc8d62')
    bars_36_above = ax1.bar(channels, units_36_above, bottom=units_12_20 + units_21_35, label='36 and above', color='#8da0cb')
    ax1.legend(loc='upper left', title='Age Segments')
    
    ax1.set_xlabel('Channel', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    ax1.set_ylabel('Units Purchased (Log Scale)', color='black', fontsize=12)
    ax1.set_yscale('log')
    ax1.tick_params(axis='y', labelcolor='black')
    
    # Line plot for money spent on second y-axis
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
    
    return fig


def main():
    """
    Main function to execute the media optimization process.
    """
    # Load and validate data
    media_df, target_df, region_df = load_and_validate_data()
    
    # Extract parameters
    params, media_df = extract_parameters(media_df, target_df, region_df)
    
    # Perform budget sensitivity analysis
    sensitivity_results = budget_sensitivity_analysis(params)
    
    # Get final solution for the user's budget
    closest_idx = sensitivity_results['closest_idx']
    final_solution = sensitivity_results['results'][closest_idx]['solution']
    final_units = np.ceil(final_solution)  # Round up to whole units
    
    # Plot budget sensitivity
    budget_fig = plot_budget_sensitivity(sensitivity_results)
    plt.show()
    
    # Print detailed spending
    final_spend = print_detailed_spending(params['channels'], final_units, params['c'], params['user_budget'])
    
    # Print summary table
    summary_df = print_summary_table(sensitivity_results['budget_values'], sensitivity_results['results'])
    
    # Calculate segment reach
    segment_reach = calculate_segment_reach(media_df, final_units)
    
    # Plot segment pie chart
    pie_fig = plot_segment_pie_chart(segment_reach)
    plt.show()
    
    # Plot stacked bar chart
    bar_fig = plot_stacked_bar_chart(params['channels'], final_units, final_spend, media_df)
    plt.show()


if __name__ == "__main__":
    main()