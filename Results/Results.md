# Results

All results are on the test loans: 1,879,687 loans originated in 2023, of which 19,524 (1.04%) became seriously delinquent (90+ days late or worse) within their first 24 payments. Full tables are in the Q1, Q2 and Q3 folders.

## Q1: What is each loan's probability of becoming seriously delinquent (90+ days late or worse) within its first 24 payments?

### AUC (how well loans are ranked by risk)

| Loans | Logistic Regression | LightGBM |
|---|---|---|
| All Loans | 0.813 | 0.821 |
| Freddie Mac | 0.815 | 0.821 |
| Fannie Mae | 0.805 | 0.817 |

AUC is the chance the model rates a loan that went bad as riskier than one that didn't (0.5 = coin flip, 1.0 = perfect). Both models score above 0.80 everywhere, with LightGBM slightly ahead.

### Calibration (predicted vs actual bad rates)

| Risk Group (1 = Lowest) | Logistic Regression: Predicted | Logistic Regression: Actual | LightGBM: Predicted | LightGBM: Actual |
|---|---|---|---|---|
| 1 | 0.05% | 0.04% | 0.04% | 0.03% |
| 2 | 0.11% | 0.09% | 0.09% | 0.08% |
| 3 | 0.19% | 0.16% | 0.17% | 0.13% |
| 4 | 0.26% | 0.25% | 0.25% | 0.23% |
| 5 | 0.40% | 0.37% | 0.39% | 0.33% |
| 6 | 0.61% | 0.54% | 0.57% | 0.54% |
| 7 | 0.88% | 0.86% | 0.85% | 0.81% |
| 8 | 1.37% | 1.35% | 1.37% | 1.37% |
| 9 | 2.18% | 2.24% | 2.18% | 2.23% |
| 10 | 4.20% | 4.49% | 4.39% | 4.64% |
| All Loans | 1.03% | 1.04% | 1.03% | 1.04% |

Loans are sorted by predicted risk into ten equal groups, and each group's predicted rate should match its actual rate. They track closely, 1.03% vs 1.04% overall. LightGBM's top group went bad about 145 times as often as its bottom group.

Each loan's probability can be estimated well from application information alone. LightGBM is the stronger model: it ranks loans better, and its percentages closely match what happened. The misses are small: Freddie Mac loans went bad about 10% more often than predicted, and Fannie Mae loans about 10% less often.

## Q2: Which variables drive that risk?

### Importance (how much each variable moves predicted risk)

| Variable | LightGBM: Mean Absolute SHAP (Rank) | Logistic Regression: Mean Absolute SHAP (Rank) |
|---|---|---|
| Credit Score | 0.576 (1) | 0.662 (1) |
| Number of Borrowers | 0.506 (2) | 0.461 (2) |
| DTI | 0.288 (3) | 0.231 (3) |
| Combined LTV | 0.238 (4) | 0.149 (6) |
| Rate Spread | 0.202 (5) | 0.212 (4) |
| Loan Purpose | 0.153 (6) | 0.066 (8) |
| Loan Amount % of Limit | 0.139 (7) | 0.052 (9) |
| Loan Term | 0.108 (8) | 0.092 (7) |
| Mortgage Insurance % | 0.078 (9) | 0.169 (5) |
| Occupancy | 0.052 (10) | 0.051 (10) |
| Property Type | 0.033 (11) | 0.036 (12) |
| Channel | 0.032 (12) | 0.031 (13) |
| First-Time Homebuyer | 0.025 (13) | 0.042 (11) |
| High-Balance Flag | 0.008 (14) | 0.008 (16) |
| Special Program | 0.006 (15) | 0.013 (14) |
| At-the-Limit Flag | 0.005 (16) | 0.012 (15) |
| Number of Units | 0.002 (17) | 0.002 (17) |

Mean absolute SHAP is how much a variable moves a loan's predicted risk, on average; higher means more influence. Both models agree on the top three. Logistic Regression leans more on mortgage insurance %.

### Top Interactions (variables whose effects depend on each other, LightGBM)

| Rank | Variable Pair | Mean Absolute SHAP Interaction |
|---|---|---|
| 1 | Loan Amount % of Limit and Loan Purpose | 0.062 |
| 2 | Credit Score and Loan Amount % of Limit | 0.050 |
| 3 | DTI and Number of Borrowers | 0.046 |
| 4 | Credit Score and Number of Borrowers | 0.045 |
| 5 | Rate Spread and Loan Amount % of Limit | 0.039 |
| 6 | Rate Spread and Loan Purpose | 0.037 |
| 7 | Loan Amount % of Limit and Occupancy | 0.033 |
| 8 | DTI and Rate Spread | 0.032 |
| 9 | Credit Score and Loan Purpose | 0.030 |
| 10 | Loan Amount % of Limit and Number of Borrowers | 0.029 |

This measures how much a pair moves risk beyond what each variable does alone. All are small next to the main effects in the importance table. Loan size appears in five of the top ten pairs.

### Odds Ratios (how each variable changes the odds of going bad, Logistic Regression)

| Variable | Change | Odds Ratio | 95% Range |
|---|---|---|---|
| Credit Score | 46 points higher | 0.43 | 0.43 to 0.44 |
| Combined LTV | 17 points higher | 1.18 | 1.16 to 1.19 |
| DTI | 9 points higher | 1.30 | 1.29 to 1.31 |
| Rate Spread | 0.36 points higher | 1.19 | 1.18 to 1.20 |
| Mortgage Insurance % | 12 points higher | 1.19 | 1.18 to 1.20 |
| Loan Amount % of Limit | 27.5 points higher | 0.94 | 0.93 to 0.95 |
| Loan Term | 71 months longer | 1.17 | 1.16 to 1.18 |
| Number of Borrowers | 2 or more vs 1 | 0.40 | 0.39 to 0.40 |
| Loan Purpose | Cash-out refinance vs purchase | 1.25 | 1.23 to 1.28 |
| Occupancy | Investor vs primary home | 0.72 | 0.70 to 0.75 |
| At-the-Limit Flag | Yes vs no | 1.43 | 1.37 to 1.50 |
| High-Balance Flag | Yes vs no | 1.20 | 1.14 to 1.26 |

Above 1 raises the odds of going bad and below 1 lowers them, with other variables held equal (1.30 = 30% higher odds). Number variables are shown per one standard deviation. Mortgage insurance overlaps with combined LTV, and the two flags with loan size, so read those together.

Credit score and number of borrowers drive risk the most, followed by DTI, combined LTV and rate spread. Risk rises with lower credit scores, a single borrower, higher DTI and LTV, a higher rate spread, longer terms, cash-out refinancing and loans at the conforming limit. Loan size also matters through how it combines with other variables.

## Q3: Where should a lender set approval cutoffs to remove the most delinquent-loan dollars while declining the fewest good-loan dollars?

### Cutoff Rules (each method's best line, set on validation loans)

| Method | Cutoff Rule |
|---|---|
| No Cutoff | Approve every loan |
| Logistic Regression Risk | Decline if predicted risk is 1.115% or higher |
| LightGBM Risk | Decline if predicted risk is 1.085% or higher |
| Single Variable | Decline if credit score is 730 or lower |
| Pair of Variables | Decline if credit score is 730 or lower, or the loan is in a housing finance agency program |
| Trio of Variables | Decline if credit score is 732 or lower and (DTI is 29% or higher, or there is one borrower) |

Each method's best-scoring cutoff on the 2022 validation loans. The simple rules all center on a credit score of about 730.

### Scores (% of bad-loan dollars declined minus % of good-loan dollars declined)

| Method | Validation Score | Test Score | Test: Bad-Loan $ Declined | Test: Good-Loan $ Declined | Test: Loans Declined |
|---|---|---|---|---|---|
| No Cutoff | 0.0 | 0.0 | 0% | 0% | 0% |
| Logistic Regression Risk | 47.2 | 49.5 | 76.6% | 27.1% | 30.6% |
| **LightGBM Risk (Winner)** | **48.1** | **50.8** | **76.7%** | **25.9%** | **30.4%** |
| Single Variable | 40.8 | 41.2 | 67.1% | 25.8% | 29.1% |
| Pair of Variables | 40.9 | 41.5 | 68.5% | 27.0% | 30.6% |
| Trio of Variables | 41.4 | 41.8 | 67.4% | 25.6% | 28.6% |

0 is no better than approving everyone; 100 would mean every bad-loan dollar is declined and no good ones. LightGBM scored highest on both validation (2022) and test (2023). It beat the best simple rule by about 9 points and Logistic Regression by about 1.

A lender should set the cutoff on LightGBM's predicted risk, declining loans rated 1.085% or higher. On 2023 loans, that caught 76.7% of bad-loan dollars while turning away 25.9% of good-loan dollars. This is the point of best separation, not a profit-maximizing line. The trade-off curve data in the Q3 folder shows the other options.
