import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(42)

# Generate basic data
n = 4000
treatment = np.random.binomial(1, 0.5, n)

# Constant treatment effect
constant_effect = 5

# Generate normally distributed base outcome
base_outcome = np.random.normal(loc=100, scale=15, size=n)

# Generate outcome with constant treatment effect
outcome = base_outcome + constant_effect * treatment

# Create DataFrame
df = pd.DataFrame({
    'treatment': treatment,
    'outcome': outcome
})

# Print summary statistics
print(df.groupby('treatment')['outcome'].describe())

# Calculate and print skewness and kurtosis
print("\nSkewness:", stats.skew(df['outcome']))
print("Kurtosis:", stats.kurtosis(df['outcome']))

# Calculate and print quantile treatment effects
quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
control_quantiles = np.quantile(df[df['treatment'] == 0]['outcome'], quantiles)
treated_quantiles = np.quantile(df[df['treatment'] == 1]['outcome'], quantiles)
qte = treated_quantiles - control_quantiles

print("\nQuantile Treatment Effects:")
for q, effect in zip(quantiles, qte):
    print(f"Quantile {q}: {effect:.2f}")

# Visualize the distribution
plt.figure(figsize=(10, 6))
plt.hist(df[df['treatment'] == 0]['outcome'], bins=50, alpha=0.5, label='Control')
plt.hist(df[df['treatment'] == 1]['outcome'], bins=50, alpha=0.5, label='Treated')
plt.legend()
plt.title('Distribution of Outcomes')
plt.xlabel('Outcome')
plt.ylabel('Frequency')
plt.show()