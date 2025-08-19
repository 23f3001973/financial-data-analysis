# Interactive Data Analysis Notebook
# Author: Data Science Research Team
# Contact: 23f3001973@ds.study.iitm.ac.in
# Description: Demonstrates relationship between variables using interactive widgets

import marimo as mo
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Cell 1: Data Generation and Setup
# This cell creates the base dataset that will be used throughout the analysis
# Dependencies: None (root cell)
# Outputs: raw_data (used by cells 2, 3, 4)

def __():
    # Generate synthetic financial dataset for analysis
    np.random.seed(42)  # For reproducible results
    
    n_samples = 1000
    
    # Generate correlated financial variables
    # Base economic indicators
    market_volatility = np.random.gamma(2, 2, n_samples)
    interest_rates = np.random.normal(3.5, 1.2, n_samples)
    
    # Portfolio performance metrics (with dependencies)
    portfolio_returns = (
        0.08 + 
        0.02 * interest_rates + 
        -0.01 * market_volatility + 
        np.random.normal(0, 0.05, n_samples)
    )
    
    # Risk metrics (dependent on volatility and returns)
    risk_score = (
        50 + 
        10 * market_volatility + 
        -5 * portfolio_returns + 
        np.random.normal(0, 8, n_samples)
    )
    
    # Create DataFrame
    raw_data = pd.DataFrame({
        'portfolio_returns': portfolio_returns,
        'market_volatility': market_volatility,
        'interest_rates': interest_rates,
        'risk_score': np.clip(risk_score, 0, 100),  # Risk score 0-100
        'investment_amount': np.random.lognormal(10, 1, n_samples)
    })
    
    return raw_data

raw_data = __()

# Cell 2: Interactive Widget Controls
# This cell defines the slider widgets for interactive analysis
# Dependencies: None (independent widget definitions)
# Outputs: volatility_threshold, return_threshold (used by cells 3, 4, 5)

def __():
    # Interactive slider for volatility threshold
    # This controls which data points are included in the filtered analysis
    volatility_threshold = mo.ui.slider(
        start=0.5, 
        stop=10.0, 
        step=0.1, 
        value=5.0,
        label="Market Volatility Threshold",
        show_value=True
    )
    
    # Interactive slider for minimum return threshold
    # This filters portfolios based on performance criteria
    return_threshold = mo.ui.slider(
        start=-0.1, 
        stop=0.2, 
        step=0.01, 
        value=0.05,
        label="Minimum Portfolio Return Threshold",
        show_value=True
    )
    
    return volatility_threshold, return_threshold

volatility_threshold, return_threshold = __()

# Display the interactive controls
mo.md(f"""
## Interactive Analysis Controls

Use the sliders below to filter the dataset and observe how variable relationships change:

{volatility_threshold}
{return_threshold}

**Current Settings:**
- Volatility Threshold: {volatility_threshold.value}
- Return Threshold: {return_threshold.value:.2%}
""")

# Cell 3: Data Filtering and Processing
# This cell processes the raw data based on widget inputs
# Dependencies: raw_data (Cell 1), volatility_threshold, return_threshold (Cell 2)
# Outputs: filtered_data, summary_stats (used by cells 4, 5)

def __():
    # Filter data based on interactive widget values
    # This demonstrates how widget state affects downstream analysis
    
    filtered_data = raw_data[
        (raw_data['market_volatility'] <= volatility_threshold.value) &
        (raw_data['portfolio_returns'] >= return_threshold.value)
    ]
    
    # Calculate summary statistics for filtered dataset
    summary_stats = {
        'total_samples': len(filtered_data),
        'avg_return': filtered_data['portfolio_returns'].mean(),
        'avg_volatility': filtered_data['market_volatility'].mean(),
        'avg_risk_score': filtered_data['risk_score'].mean(),
        'correlation_return_risk': filtered_data['portfolio_returns'].corr(filtered_data['risk_score']),
        'correlation_volatility_risk': filtered_data['market_volatility'].corr(filtered_data['risk_score'])
    }
    
    return filtered_data, summary_stats

filtered_data, summary_stats = __()

# Cell 4: Dynamic Visualization
# This cell creates plots that update based on filtered data
# Dependencies: filtered_data (Cell 3), volatility_threshold, return_threshold (Cell 2)
# Outputs: correlation_plot (used by Cell 5 for insights)

def __():
    # Create dynamic visualization based on current filter settings
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # Scatter plot: Portfolio Returns vs Risk Score
    ax1.scatter(filtered_data['portfolio_returns'], filtered_data['risk_score'], 
                alpha=0.6, c=filtered_data['market_volatility'], cmap='viridis')
    ax1.set_xlabel('Portfolio Returns')
    ax1.set_ylabel('Risk Score')
    ax1.set_title('Returns vs Risk (colored by volatility)')
    
    # Histogram: Portfolio Returns Distribution
    ax2.hist(filtered_data['portfolio_returns'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax2.axvline(return_threshold.value, color='red', linestyle='--', 
                label=f'Threshold: {return_threshold.value:.2%}')
    ax2.set_xlabel('Portfolio Returns')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Portfolio Returns Distribution')
    ax2.legend()
    
    # Scatter plot: Market Volatility vs Risk Score
    ax3.scatter(filtered_data['market_volatility'], filtered_data['risk_score'], alpha=0.6, color='coral')
    ax3.set_xlabel('Market Volatility')
    ax3.set_ylabel('Risk Score')
    ax3.set_title('Volatility vs Risk Score')
    
    # Box plot: Risk Score by Volatility Quartiles
    filtered_data['volatility_quartile'] = pd.qcut(filtered_data['market_volatility'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    sns.boxplot(data=filtered_data, x='volatility_quartile', y='risk_score', ax=ax4)
    ax4.set_title('Risk Score by Volatility Quartile')
    
    plt.tight_layout()
    return fig

correlation_plot = __()

# Cell 5: Dynamic Markdown Analysis Report
# This cell generates markdown content that changes based on widget state and analysis results
# Dependencies: summary_stats, filtered_data (Cell 3), volatility_threshold, return_threshold (Cell 2)
# Outputs: Dynamic markdown report (terminal cell)

def __():
    # Generate dynamic insights based on current filter settings and correlations
    
    # Determine correlation strength categories
    def correlation_strength(corr_value):
        abs_corr = abs(corr_value)
        if abs_corr >= 0.7:
            return "strong"
        elif abs_corr >= 0.3:
            return "moderate" 
        else:
            return "weak"
    
    # Generate risk assessment based on current thresholds
    risk_level = "Low" if summary_stats['avg_risk_score'] < 40 else "Medium" if summary_stats['avg_risk_score'] < 70 else "High"
    
    # Calculate percentage of data remaining after filtering
    data_retention = (summary_stats['total_samples'] / len(raw_data)) * 100
    
    # Generate dynamic report content
    report_content = f"""
# 📊 Dynamic Analysis Report

**Analysis Parameters:**
- Volatility Threshold: **{volatility_threshold.value}**
- Return Threshold: **{return_threshold.value:.2%}**
- Email Contact: **23f3001973@ds.study.iitm.ac.in**

---

## 📈 Dataset Overview
- **Samples Analyzed:** {summary_stats['total_samples']:,} ({data_retention:.1f}% of original dataset)
- **Average Portfolio Return:** {summary_stats['avg_return']:.2%}
- **Average Market Volatility:** {summary_stats['avg_volatility']:.2f}
- **Average Risk Score:** {summary_stats['avg_risk_score']:.1f}/100

---

## 🔍 Correlation Analysis

### Returns vs Risk Correlation: {summary_stats['correlation_return_risk']:.3f}
**Interpretation:** There is a **{correlation_strength(summary_stats['correlation_return_risk'])}** 
{"negative" if summary_stats['correlation_return_risk'] < 0 else "positive"} correlation between portfolio 
returns and risk scores.

### Volatility vs Risk Correlation: {summary_stats['correlation_volatility_risk']:.3f}
**Interpretation:** Market volatility shows a **{correlation_strength(summary_stats['correlation_volatility_risk'])}** 
{"negative" if summary_stats['correlation_volatility_risk'] < 0 else "positive"} relationship with risk scores.

---

## ⚠️ Risk Assessment

**Current Risk Level:** {risk_level}

{'🟢 **Low Risk Environment**: Current filter settings show portfolios with favorable risk-return profiles.' if risk_level == "Low" else '🟡 **Moderate Risk**: Balanced risk-return characteristics observed in filtered dataset.' if risk_level == "Medium" else '🔴 **High Risk**: Elevated risk scores detected - consider tighter risk controls.'}

---

## 📋 Investment Recommendations

Based on current analysis parameters:

{'✅ **Favorable Conditions**: High-quality investment opportunities identified' if data_retention > 30 and summary_stats['avg_return'] > 0.05 else '⚠️ **Selective Approach**: Limited opportunities meet current criteria' if data_retention > 10 else '❌ **Restrictive Filters**: Consider relaxing thresholds to identify more opportunities'}

**Data Scientist Contact:** 23f3001973@ds.study.iitm.ac.in for detailed methodology questions.
"""
    
    return mo.md(report_content)

dynamic_report = __()

# Cell 6: Statistical Analysis and Insights
# This cell performs advanced statistical analysis on the filtered dataset
# Dependencies: filtered_data (Cell 3)
# Outputs: statistical_insights (terminal analysis cell)

def __():
    if len(filtered_data) < 10:
        return mo.md("⚠️ **Insufficient data for statistical analysis.** Please adjust filter thresholds.")
    
    # Perform statistical tests
    returns_normality = stats.jarque_bera(filtered_data['portfolio_returns'])
    volatility_returns_regression = stats.linregress(
        filtered_data['market_volatility'], 
        filtered_data['portfolio_returns']
    )
    
    # Calculate additional metrics
    sharpe_ratio = (filtered_data['portfolio_returns'].mean() - 0.02) / filtered_data['portfolio_returns'].std()
    information_ratio = filtered_data['portfolio_returns'].mean() / filtered_data['portfolio_returns'].std()
    
    insights_content = f"""
## 🧮 Statistical Analysis Results

### Normality Test (Jarque-Bera)
- **Test Statistic:** {returns_normality.statistic:.3f}
- **P-value:** {returns_normality.pvalue:.4f}
- **Interpretation:** Returns are {'normally distributed' if returns_normality.pvalue > 0.05 else 'not normally distributed'} (α = 0.05)

### Linear Regression: Volatility → Returns
- **Slope:** {volatility_returns_regression.slope:.4f}
- **R-squared:** {volatility_returns_regression.rvalue**2:.3f}
- **P-value:** {volatility_returns_regression.pvalue:.4f}

### Portfolio Performance Metrics
- **Sharpe Ratio:** {sharpe_ratio:.3f}
- **Information Ratio:** {information_ratio:.3f}

---
*Analysis conducted by: 23f3001973@ds.study.iitm.ac.in*
"""
    
    return mo.md(insights_content)

statistical_insights = __()

# Display all outputs
mo.vstack([
    mo.md("# 📊 Interactive Financial Data Analysis"),
    mo.md("**Research Contact:** 23f3001973@ds.study.iitm.ac.in"),
    correlation_plot,
    dynamic_report,
    statistical_insights
])
