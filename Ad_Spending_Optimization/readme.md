# Media Allocation Optimizer

A Python-based optimization tool for media budget allocation across different channels, audience segments, and geographic regions using convex optimization techniques.

## Overview

This project provides a sophisticated approach to optimizing media spending across various channels while considering:
- Multiple target audience segments (age groups)
- Regional targeting priorities
- Budget constraints
- Diminishing returns on investment

The tool uses a logarithmic saturation model to represent diminishing returns in media reach as spending increases, making it suitable for real-world marketing scenarios.

## Features

✅ **Dynamic Budget Adjustment**: Analyze how changes in budget affect overall reach and efficiency  
✅ **Audience Segment Prioritization**: Weight different age demographics according to campaign goals  
✅ **Regional Targeting Prioritization**: Allocate spending based on geographic importance  
✅ **Diminishing Returns Modeling**: Realistic logarithmic saturation curves for media effectiveness  
✅ **Visualization Suite**: Comprehensive charts including budget sensitivity, reach distribution, and channel allocation  
✅ **Error Handling**: Robust validation and graceful handling of edge cases  
✅ **Clear Data Presentation**: Formatted output tables and properly labeled visualizations  

## How It Works

The optimization model:
1. Maximizes weighted reach across audience segments and regions
2. Respects budget constraints and minimum unit requirements
3. Accounts for channel-specific characteristics (reach, saturation, audience composition)
4. Provides sensitivity analysis to understand budget impact

## Mathematical Model

### Objective Function
Maximize total weighted reach across segments and regions

### Key Equations
- Base reach per channel: `a * log(1 + b*x)`
  - Where a = reach factor, b = saturation rate
- Segment reach: `base_reach * segment_percentage`
- Region reach: `segment_reach * region_share`
- Total weighted reach: `sum(weight_segment * weight_region * reach)`

### Constraints
1. Total spend <= budget
2. Units >= minimum required units
3. All units >= 0

## Requirements

- Python 3.6+
- CVXPY
- NumPy
- Matplotlib
- Pandas

## Input Files

The program requires three CSV files:

1. **real_life_media_parameters.csv**
   - Contains parameters for each media channel
   - Must include columns: Channel, Cost_per_Unit, Total_Reach, Percent_12_20, Percent_21_35, Percent_36_above, Saturation_Rate, Min_Units, Metro_Share, Suburban_Share, Rural_Share, National_Share
   - Special row with Channel='_budget_' to specify total budget

2. **target_prioritization.csv**
   - Defines the importance weights for each audience segment
   - Weights must sum to 1.0

3. **region_prioritization.csv**
   - Defines the importance weights for each region type
   - Weights must sum to 1.0

## Usage

```python
python v5_target_audience.py
```

## Output

The program generates:

1. **Budget Sensitivity Analysis**:
   - Plot showing how reach and efficiency change with budget
   - Summary table with numerical values

2. **Detailed Spending Breakdown**:
   - Units purchased and amount spent for each channel

3. **Segment Reach Analysis**:
   - Pie chart showing the distribution of reach across age segments
   - Numerical breakdown with percentages

4. **Channel Allocation Visualization**:
   - Stacked bar chart showing units purchased per channel, split by audience segment
   - Overlay line chart showing money spent per channel

## Example Output

```
Detailed Spending for Your Budget ($10000):
TV Primetime: 5 units, $5000.00
Radio Morning: 10 units, $1500.00
Social Media: 20 units, $2000.00
Search Ads: 15 units, $1500.00

Summary Table (Budgets vs Reach vs Efficiency):
Budget ($)  Total Reach Achieved  Efficiency (Reach per $100)
   9000.0              15230.45                      169.2272
   9500.0              15876.32                      167.1192
  10000.0              16502.18                      165.0218
  10500.0              17108.83                      162.9413
  11000.0              17697.27                      160.8843

Segment-wise Reach Breakdown (after Optimization):
12-20 years : 4950.65 (30.00%)
21-35 years : 8251.09 (50.00%)
36 and above: 3300.44 (20.00%)
```

## Development Process

The project was developed through the following steps:

1. Initial media optimization model with basic budget constraints
2. Addition of audience segment prioritization
3. Implementation of regional targeting prioritization
4. Enhancement of the model with diminishing returns using logarithmic functions
5. Development of visualization suite (bar, line, pie charts)
6. Addition of error handling and data validation
7. Improvements to visualization with clear legends, rotated labels, and log scaling

## License

[MIT License](LICENSE)
