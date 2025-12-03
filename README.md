# Quantitative Credit Risk Assessment for Green Energy Bonds 🌍📉

## 📌 Overview
This repository contains the **computational framework** developed to complement the findings of my Master's Thesis: *"Evaluating Credit Risk in Green Bond Financing"*.

While the thesis provided a comprehensive **Systematic Literature Review (SLR)** on the intersection of sustainable finance and credit risk, this project translates those theoretical findings into a **stochastic mathematical model**. It quantifies how technical uncertainties in renewable energy projects (e.g., resource intermittency) propagate into financial credit risks.

## 🎯 Objectives
The primary goal is to bridge the gap between **Energy Engineering** (Technical Risk) and **Sustainable Finance** (Credit Risk) by answering:
> *How does the volatility of wind/solar resources affect the Probability of Default (PD) for a Green Bond issuer?*

## 🛠️ Methodology: Monte Carlo Simulation
Unlike traditional credit scoring models that rely on historical balance sheet data, this project uses a **forward-looking stochastic approach**:

1.  **Stochastic Revenue Modeling:** Using a Gaussian process to model the uncertainty in annual energy production (due to wind/solar intermittency).
2.  **Debt Service Analysis:** Comparing projected revenues against fixed bond obligations (Coupons + Principal).
3.  **Risk Metrics:** Calculating **Probability of Default (PD)**, **Value at Risk (VaR)**, and **Conditional Value at Risk (CVaR)** at a 95% confidence level.

## 💻 Code Structure
The core logic is encapsulated in `green_bond_risk.py`:

```python
# Example Usage
from green_bond_risk import GreenBondRiskAnalyzer

# Initialize a $10M Green Bond for a Wind Farm
model = GreenBondRiskAnalyzer(
    principal=10_000_000, 
    coupon_rate=0.045,        # 4.5% Yield (reflecting 'Greenium')
    projected_revenue=800_000,
    revenue_volatility=250_000 # High volatility due to intermittency
)

# Run 50,000 stochastic scenarios
metrics, data = model.run_monte_carlo(num_simulations=50_000)
