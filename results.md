# Results

All results are on the test loans: 1,879,687 loans originated in 2023, of which 19,524 (1.04%) became seriously delinquent (90+ days late or worse) within their first 24 payments. These are called delinquent loans below; all others are performing loans.

## Q1: What is each loan's probability of becoming seriously delinquent (90+ days late or worse) within its first 24 payments?

### AUC (model ability to rank loans by risk)

| Loans | Logistic Regression | LightGBM |
|---|---|---|
| All Loans | 0.813 | 0.821 |
| Freddie Mac | 0.815 | 0.821 |
| Fannie Mae | 0.805 | 0.817 |

AUC measures how well a model ranks loans by risk, from 0.5 (a coin flip) to 1.0 (perfect). Both models score above 0.80 overall and for each agency, with LightGBM scoring slightly higher.

### Calibration (predicted vs actual delinquency rates by risk group)

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
| Freddie Mac | 1.11% | 1.23% | 1.11% | 1.23% |
| Fannie Mae | 0.94% | 0.86% | 0.96% | 0.86% |

Loans are sorted by predicted risk into ten equal groups. Each group's predicted rate should match its actual rate, and they track closely in every group.

<details>
<summary>Calibration by agency (Freddie Mac and Fannie Mae)</summary>

**Freddie Mac**

| Risk Group (1 = Lowest) | Logistic Regression: Predicted | Logistic Regression: Actual | LightGBM: Predicted | LightGBM: Actual |
|---|---|---|---|---|
| 1 | 0.06% | 0.06% | 0.04% | 0.04% |
| 2 | 0.12% | 0.09% | 0.10% | 0.08% |
| 3 | 0.20% | 0.17% | 0.17% | 0.16% |
| 4 | 0.28% | 0.29% | 0.27% | 0.25% |
| 5 | 0.45% | 0.39% | 0.42% | 0.39% |
| 6 | 0.65% | 0.66% | 0.62% | 0.64% |
| 7 | 0.99% | 1.02% | 0.93% | 0.97% |
| 8 | 1.52% | 1.59% | 1.49% | 1.64% |
| 9 | 2.40% | 2.75% | 2.42% | 2.72% |
| 10 | 4.44% | 5.24% | 4.64% | 5.36% |

**Fannie Mae**

| Risk Group (1 = Lowest) | Logistic Regression: Predicted | Logistic Regression: Actual | LightGBM: Predicted | LightGBM: Actual |
|---|---|---|---|---|
| 1 | 0.05% | 0.03% | 0.04% | 0.03% |
| 2 | 0.10% | 0.08% | 0.09% | 0.07% |
| 3 | 0.18% | 0.15% | 0.16% | 0.11% |
| 4 | 0.25% | 0.21% | 0.24% | 0.22% |
| 5 | 0.36% | 0.35% | 0.37% | 0.28% |
| 6 | 0.56% | 0.47% | 0.53% | 0.45% |
| 7 | 0.80% | 0.76% | 0.80% | 0.70% |
| 8 | 1.19% | 1.11% | 1.24% | 1.13% |
| 9 | 1.99% | 1.79% | 2.00% | 1.83% |
| 10 | 3.94% | 3.61% | 4.11% | 3.76% |

</details>

By agency, Fannie Mae loans became delinquent less often than predicted in every group, and Freddie Mac loans more often in the riskier half.

Each loan gets its own probability, and both models estimate it well. With LightGBM, the stronger model, probabilities average 0.04% in the safest tenth of loans and 4.39% in the riskiest. It ranks loans well (AUC 0.82) and matches actual rates overall (1.03% predicted vs 1.04% actual). Accuracy is weakest by agency. Freddie Mac loans became delinquent about 10% more often than predicted, and Fannie Mae loans about 10% less often. This likely reflects differences between the agencies, such as their lender mix, that the model's variables don't capture.

## Q2: Which variables drive that risk?

### Importance (each variable's impact on predicted risk)

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

Mean absolute SHAP (SHapley Additive exPlanations) is how much a variable moves a loan's predicted risk, on average. Both models agree on the top three variables.

### Shape (how each variable's effect changes across its values)

**Numeric variables**

<details>
<summary>Credit Score</summary>

| Credit Score | 620 | 660 | 700 | 740 | 780 | 820 | Missing |
|---|---|---|---|---|---|---|---|
| LightGBM | 2.31 | 1.63 | 0.79 | 0.00 | -0.48 | -0.91 | 0.85 |
| Logistic Regression | 2.34 | 1.61 | 0.88 | 0.15 | -0.58 | -1.31 | -0.14 |

</details>

<details>
<summary>Combined LTV</summary>

| Combined LTV | 60 | 70 | 80 | 90 | 95 | 97 | 100 | 105 |
|---|---|---|---|---|---|---|---|---|
| LightGBM | -0.20 | -0.11 | -0.01 | 0.17 | 0.41 | 0.55 | 0.69 | 0.78 |
| Logistic Regression | -0.15 | -0.05 | 0.05 | 0.14 | 0.19 | 0.21 | 0.24 | 0.29 |

</details>

<details>
<summary>DTI</summary>

| DTI | 10 | 20 | 30 | 40 | 45 | 46 | 50 |
|---|---|---|---|---|---|---|---|
| LightGBM | -0.50 | -0.47 | -0.21 | 0.20 | 0.44 | 0.35 | 0.53 |
| Logistic Regression | -0.68 | -0.40 | -0.13 | 0.15 | 0.29 | 0.32 | 0.43 |

</details>

<details>
<summary>Rate Spread</summary>

| Rate Spread | -1.00 | -0.50 | -0.25 | 0.00 | 0.25 | 0.50 | 1.00 |
|---|---|---|---|---|---|---|---|
| LightGBM | -0.07 | -0.25 | -0.21 | -0.05 | 0.19 | 0.32 | 0.50 |
| Logistic Regression | -0.51 | -0.27 | -0.15 | -0.03 | 0.09 | 0.21 | 0.44 |

</details>

<details>
<summary>Mortgage Insurance %</summary>

| Mortgage Insurance % | 0 | 12 | 25 | 30 | 35 |
|---|---|---|---|---|---|
| LightGBM | -0.05 | 0.07 | 0.10 | 0.13 | 0.23 |
| Logistic Regression | -0.11 | 0.07 | 0.25 | 0.32 | 0.40 |

</details>

<details>
<summary>Loan Amount % of Limit</summary>

| Loan Amount % of Limit | Under 25 | 25 to 50 | 50 to 75 | 75 to 100 | Over 100 |
|---|---|---|---|---|---|
| LightGBM | 0.33 | 0.01 | -0.08 | -0.02 | -0.20 |
| Logistic Regression | 0.08 | 0.04 | -0.02 | -0.08 | -0.16 |

</details>

<details>
<summary>Loan Term</summary>

| Loan Term (Months) | 120 | 180 | 240 | 300 | 360 |
|---|---|---|---|---|---|
| LightGBM | -0.51 | -0.36 | -0.21 | 0.00 | 0.09 |
| Logistic Regression | -0.45 | -0.32 | -0.18 | -0.05 | 0.08 |

</details>

**Categorical variables**

<details>
<summary>Loan Purpose</summary>

| Value | LightGBM | Logistic Regression |
|---|---|---|
| Purchase | -0.14 | -0.05 |
| Cash-out refinance | 0.29 | 0.18 |
| No-cash-out refinance | 0.03 | -0.05 |

</details>

<details>
<summary>Number of Borrowers</summary>

| Value | LightGBM | Logistic Regression |
|---|---|---|
| 1 | 0.49 | 0.45 |
| 2 or more | -0.53 | -0.48 |

</details>

<details>
<summary>Occupancy</summary>

| Value | LightGBM | Logistic Regression |
|---|---|---|
| Primary home | 0.03 | 0.03 |
| Second home | -0.06 | -0.10 |
| Investment property | -0.29 | -0.29 |

</details>

<details>
<summary>First-Time Homebuyer</summary>

| Value | LightGBM | Logistic Regression |
|---|---|---|
| Yes | 0.05 | 0.07 |
| No | 0.00 | -0.02 |

</details>

<details>
<summary>Channel</summary>

| Value | LightGBM | Logistic Regression |
|---|---|---|
| Retail | 0.00 | -0.02 |
| Broker | 0.11 | 0.11 |
| Correspondent | -0.03 | 0.01 |

</details>

<details>
<summary>Property Type</summary>

| Value | LightGBM | Logistic Regression |
|---|---|---|
| Single-family | 0.01 | 0.01 |
| Planned development | 0.01 | 0.04 |
| Condo | -0.13 | -0.16 |
| Co-op | -0.15 | -0.50 |
| Manufactured home | 0.25 | 0.20 |

</details>

<details>
<summary>Number of Units</summary>

| Value | LightGBM | Logistic Regression |
|---|---|---|
| 1 | 0.00 | 0.00 |
| 2 | -0.07 | -0.07 |
| 3 | -0.01 | -0.04 |
| 4 | 0.00 | 0.13 |

</details>

<details>
<summary>Special Program</summary>

| Value | LightGBM | Logistic Regression |
|---|---|---|
| None | 0.00 | 0.00 |
| Housing finance agency | 0.00 | 0.28 |
| Affordable lending | 0.00 | 0.02 |

</details>

<details>
<summary>High-Balance Flag</summary>

| Value | LightGBM | Logistic Regression |
|---|---|---|
| Yes | 0.16 | 0.18 |
| No | -0.01 | -0.01 |

</details>

<details>
<summary>At-the-Limit Flag</summary>

| Value | LightGBM | Logistic Regression |
|---|---|---|
| Yes | 0.15 | 0.35 |
| No | 0.00 | -0.01 |

</details>

Positive values push predicted risk up and negative values push it down. Credit score has by far the largest effect, from about +2.3 at 620 to -0.9 or lower at 820, followed by number of borrowers, DTI and combined LTV. Logistic Regression's effects change in a straight line by design, while LightGBM captures nonlinear patterns: risk climbs sharply above about 93% combined LTV and rises again for very small loans and for rate spreads far below the market. LightGBM also rates loans with no credit score as risky, while Logistic Regression rates them near average.

### Compounding (variables whose effects depend on each other; LightGBM only)

| Rank | Variable Pair | Mean Absolute SHAP Interaction |
|---|---|---|
| 1 | Loan Amount % of Limit × Loan Purpose | 0.062 |
| 2 | Credit Score × Loan Amount % of Limit | 0.050 |
| 3 | DTI × Number of Borrowers | 0.046 |
| 4 | Credit Score × Number of Borrowers | 0.045 |
| 5 | Rate Spread × Loan Amount % of Limit | 0.039 |
| 6 | Rate Spread × Loan Purpose | 0.037 |
| 7 | Loan Amount % of Limit × Occupancy | 0.033 |
| 8 | DTI × Rate Spread | 0.032 |
| 9 | Credit Score × Loan Purpose | 0.030 |
| 10 | Loan Amount % of Limit × Number of Borrowers | 0.029 |

Mean absolute SHAP interaction is how much a pair of variables moves a loan's predicted risk beyond what each does alone, on average. All pairs are small next to the main effects in the importance table. Loan amount % of limit appears in 5 of the top 10 pairs.

### Odds Ratios (how each variable changes the odds of delinquency; Logistic Regression only)

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

Above 1 raises the odds of delinquency and below 1 lowers them, with other variables held equal (1.30 = 30% higher odds). Number variables are shown per one standard deviation. Some variables overlap, so read them together: mortgage insurance % with combined LTV, and the high-balance and at-the-limit flags with loan amount % of limit.

Credit score and number of borrowers drive risk the most, followed by DTI, combined LTV and rate spread. Risk rises with lower credit scores, a single borrower, higher DTI and combined LTV, a higher rate spread, longer terms, cash-out refinancing and loans at the conforming limit. Loan amount % of limit also matters through how it combines with other variables, especially loan purpose.

## Q3: Where should a lender set approval cutoffs to decline the most delinquent loans while turning away the fewest performing loans, measured by loan balance?

### Cutoff Rules (each method's best line, set on validation loans)

| Method | Cutoff Rule |
|---|---|
| No Cutoff | Approve every loan |
| Logistic Regression Risk | Decline if predicted risk is 1.115% or higher |
| LightGBM Risk | Decline if predicted risk is 1.085% or higher |
| Single Variable | Decline if credit score is 730 or lower |
| Pair of Variables | Decline if credit score is 730 or lower, or the loan is in a housing finance agency program |
| Trio of Variables | Decline if credit score is 732 or lower and (DTI is 29% or higher, or there is one borrower) |

Each method's best-scoring cutoff on the 2022 validation loans. The variable rules all center on a credit score of about 730.

### Scores (% of delinquent balance declined minus % of performing balance declined)

| Method | Validation Score | Test Score | Test: Delinquent Balance Declined | Test: Performing Balance Declined | Test: Loans Declined |
|---|---|---|---|---|---|
| No Cutoff | 0.0 | 0.0 | 0% | 0% | 0% |
| Logistic Regression Risk | 47.2 | 49.5 | 76.6% | 27.1% | 30.6% |
| **LightGBM Risk (Winner)** | **48.1** | **50.8** | **76.7%** | **25.9%** | **30.4%** |
| Single Variable | 40.8 | 41.2 | 67.1% | 25.8% | 29.1% |
| Pair of Variables | 40.9 | 41.5 | 68.5% | 27.0% | 30.6% |
| Trio of Variables | 41.4 | 41.8 | 67.4% | 25.6% | 28.6% |

0 is no better than approving everyone; 100 would mean all delinquent balance is declined and no performing balance. LightGBM scored highest on both validation (2022) and test (2023). It beat the best variable rule by about 9 points and Logistic Regression by about 1.

### Test Results (counts and dollars, all loans then each agency)

**All Loans**

| Method | Score | Delinquent Loans Declined | Performing Loans Declined | Delinquent Loans Approved | Performing Loans Approved |
|---|---|---|---|---|---|
| No Cutoff | 0.0 | 0 ($0.00B) | 0 ($0.00B) | 19,524 ($5.67B) | 1,860,163 ($590.30B) |
| Logistic Regression Risk | 49.5 | 15,282 ($4.35B) | 559,347 ($159.86B) | 4,242 ($1.33B) | 1,300,816 ($430.44B) |
| **LightGBM Risk (Winner)** | **50.8** | **15,563 ($4.35B)** | **555,931 ($153.07B)** | **3,961 ($1.32B)** | **1,304,232 ($437.23B)** |
| Single Variable | 41.2 | 13,487 ($3.80B) | 534,041 ($152.51B) | 6,037 ($1.87B) | 1,326,122 ($437.79B) |
| Pair of Variables | 41.5 | 13,829 ($3.89B) | 560,872 ($159.66B) | 5,695 ($1.79B) | 1,299,291 ($430.64B) |
| Trio of Variables | 41.8 | 13,494 ($3.82B) | 523,374 ($150.90B) | 6,030 ($1.85B) | 1,336,789 ($439.41B) |

**Freddie Mac**

| Method | Score | Delinquent Loans Declined | Performing Loans Declined | Delinquent Loans Approved | Performing Loans Approved |
|---|---|---|---|---|---|
| No Cutoff | 0.0 | 0 ($0.00B) | 0 ($0.00B) | 11,362 ($3.23B) | 915,755 ($286.68B) |
| Logistic Regression Risk | 50.2 | 9,237 ($2.56B) | 298,060 ($83.37B) | 2,125 ($0.67B) | 617,695 ($203.31B) |
| **LightGBM Risk (Winner)** | **51.3** | **9,302 ($2.55B)** | **291,772 ($78.90B)** | **2,060 ($0.69B)** | **623,983 ($207.78B)** |
| Single Variable | 41.2 | 8,112 ($2.21B) | 279,439 ($77.89B) | 3,250 ($1.02B) | 636,316 ($208.79B) |
| Pair of Variables | 41.6 | 8,290 ($2.26B) | 290,370 ($80.84B) | 3,072 ($0.98B) | 625,385 ($205.84B) |
| Trio of Variables | 41.9 | 8,135 ($2.23B) | 275,298 ($77.37B) | 3,227 ($1.01B) | 640,457 ($209.31B) |

**Fannie Mae**

| Method | Score | Delinquent Loans Declined | Performing Loans Declined | Delinquent Loans Approved | Performing Loans Approved |
|---|---|---|---|---|---|
| No Cutoff | 0.0 | 0 ($0.00B) | 0 ($0.00B) | 8,162 ($2.44B) | 944,408 ($303.62B) |
| Logistic Regression Risk | 47.8 | 6,045 ($1.78B) | 261,287 ($76.49B) | 2,117 ($0.66B) | 683,121 ($227.13B) |
| **LightGBM Risk (Winner)** | **49.5** | **6,261 ($1.80B)** | **264,159 ($74.17B)** | **1,901 ($0.64B)** | **680,249 ($229.45B)** |
| Single Variable | 40.8 | 5,375 ($1.59B) | 254,602 ($74.62B) | 2,787 ($0.85B) | 689,806 ($229.00B) |
| Pair of Variables | 40.9 | 5,539 ($1.63B) | 270,502 ($78.83B) | 2,623 ($0.81B) | 673,906 ($224.80B) |
| Trio of Variables | 41.1 | 5,359 ($1.59B) | 248,076 ($73.52B) | 2,803 ($0.85B) | 696,332 ($230.10B) |

Each cell shows the number of loans and their total original balance; declined and approved add up to each group's totals. LightGBM scored highest on all loans and within each agency.

### Trade-off Curve (delinquent balance caught at each level of performing balance declined)

| Performing Balance Declined | Logistic Regression Risk | LightGBM Risk | Credit Score Alone |
|---|---|---|---|
| 5% | 29.3% | 30.6% | 23.6% |
| 10% | 46.1% | 47.4% | 38.6% |
| 20% | 67.4% | 68.7% | 58.7% |
| 30% | 79.5% | 80.6% | 72.0% |
| 40% | 87.2% | 88.0% | 80.7% |
| 50% | 92.2% | 92.6% | 87.1% |

Each cell is the % of delinquent balance caught when a method's cutoff is moved to decline that % of performing balance, read from each method's full curve on the test loans (between curve points, along a straight line). Each method's chosen cutoff, including the pair and trio rules, is one point on this trade-off, shown in the Scores table. LightGBM catches the most at every level, and both models stay well ahead of credit score alone.

A lender should set the cutoff on LightGBM's predicted risk, declining loans rated 1.085% or higher. On 2023 loans, that caught 76.7% of delinquent balance while turning away 25.9% of performing balance. This is the point of best separation, not a profit-maximizing line; a lender with its own cost data could choose a different point.
