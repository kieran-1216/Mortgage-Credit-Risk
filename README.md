# Mortgage Credit Risk

## Business problem
A lender has to decide which mortgage applications to approve. Approving more grows volume. Approving riskier loans raises delinquencies. Using only what is known at application, it needs to know where the risk sits and where to draw the line.

## Core questions
For conventional conforming, fully amortizing, fixed-rate, single-family (1-4 unit) mortgages purchased by Freddie Mac and Fannie Mae, using only information known at application:

1. **What is each loan's probability of becoming seriously delinquent (90+ days late or worse) within its first 24 payments?**
2. **Which variables drive that risk?**
3. **Where should a lender set approval cutoffs to remove the most delinquent-loan dollars while declining the fewest good-loan dollars?**

## Data and setup
- **Data:** about 25 million loans from Freddie Mac's and Fannie Mae's public single-family loan-level data, originated 2014Q1 to 2023Q4. Loans originated 2020Q2 to 2022Q1 (the pandemic period, when payment relief programs were widespread) are excluded.
- **Training (2014Q1 to 2020Q1):** models learn from these loans.
- **Validation (2022Q2 to 2022Q4):** settings, percentage adjustments and cutoffs are chosen on these loans.
- **Test (2023Q1 to 2023Q4):** results are measured on these loans, which are never used to learn or choose anything.
- **Variables (17, all known at application):** credit score, combined LTV, DTI, rate spread (the loan's rate minus the median rate for similar loans that month), mortgage insurance %, loan amount as a % of that year's conforming loan limit, loan term, loan purpose, number of borrowers, occupancy, first-time homebuyer, channel, property type, number of units, special program, high-balance flag, and an at-the-limit flag.

## How each question is answered

### Q1: Each loan's risk
Two models: **Logistic Regression**, the classic explainable scorecard model, and **LightGBM**, a gradient-boosted tree model that can capture curves and interactions. Comparing a simple, transparent model with a more flexible one shows how much the extra complexity actually buys.

- **Logistic Regression** learns from training loans. It has no settings to tune.
- **LightGBM:** Optuna's tuner tries different setups. Each setup learns from training loans, and the setup whose predictions best match what actually happened on validation loans (by log loss) is kept.
- **Adjusting the percentages:** training loans went bad less often than recent loans, so each model's percentages are adjusted to match validation loans with isotonic calibration (checked on validation loans left out of the fit).
- **Measured on test loans,** all loans and then each agency alone:
  - **Order (AUC):** do the riskiest-rated loans actually go bad most?
  - **Level (calibration):** do the percentages match actual bad rates across ten equal-sized risk groups?

### Q2: What drives the risk
Uses the Q1 models; nothing new is learned or chosen. SHAP values on test loans:
- **Importance:** how much each variable affects predicted risk, ranked, both models side by side.
- **Shape:** how each variable's effect changes across its values.
- **Compounding:** which variables are riskier together than apart (LightGBM).
- **Odds ratios:** Logistic Regression's weights, the standard scorecard view.

### Q3: Where to set the approval cutoff
- **Methods compared:** no cutoff (baseline), Logistic Regression risk, LightGBM risk, each variable alone, every pair of variables, and every trio of variables (in every and/or combination).
- **Score:** % of bad-loan dollars declined minus % of good-loan dollars declined, using each loan's original amount.
- **Process:** cutoffs come from training values, every cutoff is scored on validation loans, and each method's best cutoff is applied once, unchanged, to test loans.
- **Winner:** the method with the highest validation score.
- **Trade-off curve:** on test loans, how much bad-loan money is caught at each amount of good-loan money turned away.

## Limitations
- **Approved loans only.** The data holds only loans that were approved and sold to the agencies, so the models never see declined applications.
- **No cost or profit data.** Q3's cutoff is the point of best separation between bad-loan and good-loan dollars, not a profit-maximizing line. A real lender would set its line from its own costs; the trade-off curve shows the alternatives.
- **Older training loans.** Loans from 2014 to 2020 behaved differently from 2022 and 2023 loans. Percentages are re-adjusted on 2022 loans, but some groups, mainly refinances and larger loans near the conforming limit, are still predicted less accurately in 2023.
- **One model for both agencies.** Freddie Mac loans ran riskier than predicted, and Fannie Mae loans safer. Agency is not a variable because gaps in Fannie Mae's training-era data would teach the model the wrong direction.
- **No race or ethnicity data,** so fair-lending effects can't be measured.
- **Results review.** Models were refined after a review of first test results (calibration and loan-size handling); the 2023 test set was rescored once.

## Tools used
- **Python:** the full pipeline, from data loading to modeling and checks.
- **DuckDB (SQL):** loading and building the 25-million-loan dataset from the raw agency files on a laptop.
- **LightGBM and Optuna:** the gradient-boosted model and its settings search.
- **scikit-learn:** isotonic calibration.
- **SHAP (via LightGBM's built-in contributions):** explaining what drives each prediction.
- **Streamlit:** the live tool.
- **Power BI:** results dashboard.
- **Claude Code:** AI coding assistant, used to help write and test the code.

## Repository
- **Results:** [results/](results/): tables and explanations for Q1, Q2 and Q3
- **Live tool:** coming soon
- **Code:** [code/](code/): the pipeline scripts

## Data notice
Built from Freddie Mac's Single-Family Loan-Level Dataset and Fannie Mae's Single-Family Loan Performance Data, used for noncommercial research. This project is not affiliated with or endorsed by Freddie Mac or Fannie Mae. No loan-level data is included in this repository.
