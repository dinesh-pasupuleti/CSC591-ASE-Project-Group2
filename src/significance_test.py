
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
# Load the MSE data from both files
predicted_mse = pd.read_csv('/Users/dineshpasupuleti/CSC591-ASE-Project-Group2/data/smo/Wine_quality_results.csv')
actual_mse = pd.read_csv('/Users/dineshpasupuleti/CSC591-ASE-Project-Group2/data/gradientsearch/Wine_quality_mse.csv')

# Assuming 'mse' is the column name for MSE values in both dataframes
mse_pred = predicted_mse['mse']
mse_actual = actual_mse['mse']

# Shapiro-Wilk Test for normality
print("Normality Test (SMO):", stats.shapiro(mse_pred))
print("Normality Test (Grid Search):", stats.shapiro(mse_actual))

# Mann-Whitney U Test for non-parametric comparison
u_stat, p_value = stats.mannwhitneyu(mse_pred, mse_actual, alternative='two-sided')
print("U-statistic:", u_stat, "P-value:", p_value)

# Calculate effect size r
r_effect_size = 1 - (2 * u_stat) / (len(mse_pred) * len(mse_actual))
print("Effect Size r:", r_effect_size)



