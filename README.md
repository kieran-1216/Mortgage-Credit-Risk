# Mortgage Credit Risk

## Business Problem
Mortgage lenders balance growth against risk. Approving more loans increases volume, but approving riskier loans increases delinquencies. Lenders must estimate each loan's risk and decide where to set their approval cutoff.

## Core Questions
For conventional conforming, fully amortizing, fixed-rate, single-family (1-4 unit) mortgages purchased by Freddie Mac and Fannie Mae, using only information known at application:

1. **What is each loan's probability of becoming seriously delinquent (90+ days late or worse) within its first 24 payments?**
2. **Which variables drive that risk?**
3. **Where should a lender set approval cutoffs to decline the most delinquent loans while turning away the fewest performing loans, measured by loan balance?**

## Data and Setup
- **Data:** ~25 million loans from Freddie Mac's and Fannie Mae's public single-family loan-level data, originated 2014Q1 to 2023Q4. Loans originated 2020Q2 to 2022Q1 (the pandemic period, when payment relief programs were widespread) are excluded.
- **Training (2014Q1 to 2020Q1):** models learn from these loans.
- **Validation (2022Q2 to 2022Q4):** settings, percentage adjustments and cutoffs are chosen on these loans.
- **Test (2023Q1 to 2023Q4):** results are measured on these loans.
- **Variables (17, all known at application):**
  - **Credit score:** the lowest credit score among the loan's borrowers
  - **Combined LTV:** all loans on the property at closing as a % of its value
  - **DTI:** monthly debt payments, including the new mortgage, as a % of monthly income
  - **Rate spread:** the loan's interest rate minus the median rate for same-term loans that month
  - **Mortgage insurance %:** the share of the loan covered by mortgage insurance (0 if none)
  - **Loan amount % of limit:** the loan amount as a % of that year's conforming loan limit for its number of units
  - **Loan term:** the loan's length in months
  - **Loan purpose:** purchase, cash-out refinance, or no-cash-out refinance
  - **Number of borrowers:** one, or two or more
  - **Occupancy:** primary home, second home, or investment property
  - **First-time homebuyer:** whether the borrower is buying their first home
  - **Channel:** how the loan was originated (retail, broker, or correspondent lender)
  - **Property type:** single-family, planned development, condo, co-op, or manufactured home
  - **Number of units:** 1 to 4
  - **Special program:** whether the loan is in an affordable-lending, housing finance agency, or refinance program
  - **High-balance flag:** whether the loan is above the standard limit, allowed only in high-cost areas
  - **At-the-limit flag:** whether the loan amount equals the conforming limit exactly

## Answering Core Questions

### Q1: How likely is each loan to become seriously delinquent?
Two models are built: Logistic Regression and LightGBM.

**Logistic Regression**
- **Train (2014Q1 to 2020Q1):** learns from training loans, using only variables known at application. It has no settings to tune.
- **Validate (2022Q2 to 2022Q4):** predicts the validation loans, and its percentages are adjusted to match them (isotonic calibration).
- **Test (2023Q1 to 2023Q4):** predicts the test loans with that adjustment and is measured on all test loans, then on each agency's alone:
  - **Order (AUC):** do the riskiest-rated loans actually become delinquent most often?
  - **Level (calibration):** do its percentages match actual delinquency rates across ten equal-sized risk groups?

**LightGBM**
- **Train (2014Q1 to 2020Q1):** Optuna tries different setups within ranges fixed before modeling. Each setup learns from training loans, using only variables known at application.
- **Validate (2022Q2 to 2022Q4):** each setup predicts the validation loans, and its percentages are adjusted to match them (isotonic calibration). The setup whose adjusted percentages come closest to what actually happened is kept, along with its adjustment.
- **Test (2023Q1 to 2023Q4):** predicts the test loans with that adjustment and is measured on all test loans, then on each agency's alone:
  - **Order (AUC):** do the riskiest-rated loans actually become delinquent most often?
  - **Level (calibration):** do its percentages match actual delinquency rates across ten equal-sized risk groups?

### Q2: Which variables drive that risk?
Uses the Q1 models. Nothing new is learned or chosen. SHAP values on test loans (2023Q1 to 2023Q4), on the log-odds scale:
- **Importance:** how much each variable affects predicted risk, ranked. Both models side by side.
- **Shape:** how each variable's effect changes across its values. Both models: smooth for Logistic Regression by design; LightGBM can also show direction changes and sharp jumps.
- **Compounding:** which variables are riskier together than apart. LightGBM only.
- **Odds ratios:** Logistic Regression's weights, the standard scorecard view.

### Q3: Which cutoff best separates delinquent from performing loans?
- **Methods:** no cutoff (baseline); Logistic Regression risk; LightGBM risk; each variable alone; every pair of variables; every trio of variables (pairs and trios in every and/or combination).
- **Riskier end:** for the variable, pair and trio rules, each variable's riskier end is learned from training loans. For numbers, it's the end where loans became delinquent more often (for example, higher DTI). For categories, they're ordered from most to least often delinquent.
- **Missing values:** loans missing a value are kept together and placed in that order by how often they became delinquent in training.
- **Score:** % of delinquent balance declined minus % of performing balance declined, using each loan's original amount.
- **Process (for each method):**
  - **Cutoffs:** set from training values. For a single variable or a model's risk, every distinct value (for a category, each boundary in riskiest-first order). For pairs and trios, each variable at 1% steps, in every combination.
  - **Scoring the cutoffs:** every cutoff is scored on validation loans, and the highest score is kept; ties go to the cutoff declining fewer loan dollars.
  - **Testing the cutoff:** only the chosen cutoff is applied to test loans, as a fixed rule (for example, decline if risk is above 4%). Results include the score and the delinquent and performing loans declined and approved, in counts and dollars, on all test loans and then on each agency's alone. No winner is picked here; test results are just reported.
- **Trade-off curve:** on test loans, how much delinquent balance is caught at each amount of performing balance turned away; a line for each model and single-variable method, plus a dot at each method's chosen cutoff, all on one chart.
- **Winning method:** the method with the highest validation score; ties go to the method declining fewer loan dollars, and if still tied, the method using fewer variables wins.

## Tools
- **Python:** the full pipeline, from data loading to modeling and checks.
- **DuckDB (SQL):** loading and building the 25-million-loan dataset from the raw agency files.
- **LightGBM and Optuna:** the gradient-boosted model and its settings search.
- **scikit-learn:** isotonic calibration.
- **SHAP (via LightGBM's built-in contributions):** explaining what drives each prediction.
- **Streamlit:** the live tool.
- **Power BI:** results dashboard.
- **Claude Code:** AI coding assistant, used to help write and test the code.

## Limitations
- **Approved loans only.** The data holds only loans that were approved and sold to the agencies, so the models never see declined applications.
- **No cost or profit data.** Q3's cutoff is the point of best separation between delinquent and performing balances, not a profit-maximizing line. A real lender would set its line from its own costs; the trade-off curve shows the alternatives.
- **Older training loans.** Loans from 2014 to 2020 behaved differently from 2022 and 2023 loans. Percentages are re-adjusted on 2022 loans, but some groups, mainly refinances and larger loans near the conforming limit, are still predicted less accurately in 2023.
- **One model for both agencies.** Freddie Mac loans ran riskier than predicted, and Fannie Mae loans safer. Agency is not a variable because gaps in Fannie Mae's training-era data would teach the model the wrong direction.
- **Results review.** Models were refined after a review of first test results (calibration and loan-size handling); the 2023 test set was rescored once.

## Repository
- **[Results](results.md):** detailed results of the core questions
- **[Live Tool](live-tool/):** interactive tool that estimates a loan's risk from user inputs
- **[Code](code/):** the pipeline scripts

## Data Disclaimers
Built from Freddie Mac's Single-Family Loan-Level Dataset and Fannie Mae's Single-Family Loan Performance Data, used for noncommercial research. This project is not affiliated with or endorsed by Freddie Mac or Fannie Mae. No loan-level data is included in this repository.
