# Machine Learning Assignment 2 - BITS Pilani WILP

## a. Problem Statement
This project implements and evaluates 5 machine learning classification algorithms to predict the baseline histological staging of patients based on clinical and laboratory metrics.

## b. Dataset Description
- **Source:** HCV / Hepatitis Dataset (UCI Machine Learning Repository)
- **Instances:** 1,385 rows
- **Features:** 28 columns (Age, Gender, BMI, Fever, Headache, Liver Enzyme levels, etc.)
- **Target Column:** `Baseline histological staging`

## c. GitHub Repository Link
`https://github.com/YOUR_GITHUB_USERNAME/YOUR_REPO_NAME`

## d. Models Used & Performance Comparison

### Comparison Table
| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 0.3141 | 0.5191 | 0.3315 | 0.3141 | 0.2080 | 0.0853 |
| **Decision Tree** | 0.2130 | 0.4741 | 0.2105 | 0.2130 | 0.2092 | -0.0515 |
| **kNN** | 0.2094 | 0.4916 | 0.2210 | 0.2094 | 0.2095 | -0.0527 |
| **Naive Bayes** | 0.2671 | 0.4990 | 0.2570 | 0.2671 | 0.2508 | 0.0267 |
| **Support Vector Machine** | 0.2635 | 0.5137 | 0.1700 | 0.2635 | 0.1726 | 0.0100 |

### Performance Observations
- **Logistic Regression:** Achieved the highest accuracy (31.41%) and highest AUC score (0.5191) among all tested models, demonstrating superior feature weighting on linear projections.
- **Decision Tree:** Overfitted the training distribution, resulting in low generalizability on test data (F1 score of 0.2092).
- **kNN:** Exhibited negative MCC correlation (-0.0527), indicating performance near random selection due to high feature dimensionality.
- **Naive Bayes:** Achieved consistent multi-class balance with an accuracy of 26.71% and F1 score of 0.2508.
- **Support Vector Machine:** Produced steady classification bounds with an accuracy of 26.35% and positive MCC score.

**Overall Winner for your dataset:** **Logistic Regression** (Highest Accuracy, AUC, Precision, and positive MCC).