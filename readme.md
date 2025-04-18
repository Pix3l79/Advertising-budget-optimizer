
# 📈 Media Allocation Optimizer

## Project Overview

**Media Allocation Optimizer** is a Python-based convex optimization tool designed to allocate advertising budgets efficiently across multiple media channels, audience segments, and geographic regions.

The model reflects real-world marketing complexities such as diminishing returns on ad spend, varying audience demographics, regional priorities, and strict budgetary constraints — providing data-driven decision support for maximizing campaign reach.

---

## Motivation

The goal of this project was to apply convex optimization techniques to a realistic marketing scenario, progressively incorporating practical challenges faced by media planners.

**Development Milestones:**
1. **Foundational Model:** Implemented a basic optimization model with budget constraints.
2. **Diminishing Returns:** Introduced logarithmic modeling to capture declining marginal returns on media spend.
3. **Reach Efficiency Metric:** Added *Reach per Dollar* to benchmark channel performance.
4. **Audience Segmentation:** Enabled targeting prioritization across different age groups.
5. **Regional Targeting:** Integrated regional allocation across Metro, Suburban, Rural, and National categories.
6. **Channel-Specific Demographics:** Incorporated real-world audience distribution data by channel and age group.
7. **Cost and Saturation Realism:** Sourced average national media costs and saturation rates, with citations embedded in input datasets.
8. **Enhanced Visualization:** Developed comprehensive, presentation-ready plots for storytelling and analysis.

---

## Problem Formulation

The optimizer addresses the following objectives:

- Maximize total weighted reach across selected audience segments and regions.
- Respect overall budget limitations and minimum unit purchase requirements.
- Model realistic audience saturation and media effectiveness.
- Support strategic sensitivity analyses to understand budget impact.

---

## Mathematical Model

### Objective
> Maximize the **total weighted reach** across all audience segments and geographic regions.

### Key Equations
- **Base Channel Reach:**  
  \[
  \text{base\_reach} = a \times \log(1 + b \times x)
  \]
- **Segment-Level Reach:**  
  \[
  \text{segment\_reach} = \text{base\_reach} \times \text{segment\_percentage}
  \]
- **Region-Level Reach:**  
  \[
  \text{region\_reach} = \text{segment\_reach} \times \text{region\_share}
  \]
- **Total Weighted Reach:**  
  \[
  \sum (\text{weight\_segment} \times \text{weight\_region} \times \text{reach})
  \]

### Constraints
- Total expenditure ≤ specified budget
- Units purchased ≥ minimum thresholds

---

## Key Features

- ✅ **Dynamic Budget Adjustment**
- ✅ **Audience Prioritization**
- ✅ **Regional Targeting**
- ✅ **Diminishing Returns Modeling**
- ✅ **Comprehensive Visualization**
- ✅ **Robust Error Handling**
- ✅ **Real-World Data Sources**

---

## Installation Requirements

- Python 3.6+
- CVXPY
- NumPy
- Pandas
- Matplotlib

Install via:
```bash
pip install -r requirements.txt
```

---

## Input Files

- **real_life_media_parameters.csv**
- **target_prioritization.csv**
- **region_prioritization.csv**

---

## Outputs

- Budget Sensitivity Analysis
- Spending Breakdown
- Segment Reach Analysis
- Channel Allocation Visualization

---

## Example Output

```
Detailed Spending Breakdown for $10,000 Budget:
- TV Primetime: 5 units ($5,000)
- Radio Morning: 10 units ($1,500)
- Social Media: 20 units ($2,000)
- Search Ads: 15 units ($1,500)

Budget vs Reach vs Efficiency:
 Budget ($)  | Total Reach  | Efficiency (Reach per $100)
-------------|--------------|------------------------------
    9,000    |    15,230    |           169.23
    9,500    |    15,876    |           167.12
   10,000    |    16,502    |           165.02
   10,500    |    17,109    |           162.94
   11,000    |    17,697    |           160.88

Segment-wise Reach:
- 12–20 years: 4,950.65 (30.00%)
- 21–35 years: 8,251.09 (50.00%)
- 36+ years: 3,300.44 (20.00%)
```

---

## Future Enhancements

- Multi-period optimization
- Dynamic audience modeling
- Channel fatigue adjustment

---

# 🚀 Connect

Connect with me on [LinkedIn](https://www.linkedin.com/in/jaroh23/) or view more projects [here](https://github.com/Rohanjain2312).
