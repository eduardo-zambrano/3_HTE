import numpy as np
import pandas as pd
from scipy import stats

np.random.seed(42)

n = 4000
treatment = np.random.binomial(1, 0.5, n)

# Generate individual characteristic (e.g., farmer skill)
farmer_skill = np.random.normal(0, 1, n)

# Generate base outcome (same for both groups)
base_outcome = np.random.gamma(2, 10, n)

# Constant treatment effect for quantiles
constant_effect = 10

# Create heterogeneity that preserves quantile differences
het_effect = 5 * farmer_skill

# Final outcome
outcome = base_outcome + constant_effect * treatment + het_effect * np.abs(base_outcome - np.mean(base_outcome)) * treatment / 10

# Create DataFrame
df = pd.DataFrame({
    'treatment': treatment,
    'farmer_skill': farmer_skill,
    'outcome': outcome
})

# Print summary statistics
print(df.groupby('treatment')['outcome'].describe())

# Calculate and print quantile treatment effects
quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
control_quantiles = np.quantile(df[df['treatment'] == 0]['outcome'], quantiles)
treated_quantiles = np.quantile(df[df['treatment'] == 1]['outcome'], quantiles)
qte = treated_quantiles - control_quantiles

print("\nQuantile Treatment Effects:")
for q, effect in zip(quantiles, qte):
    print(f"Quantile {q}: {effect:.2f}")

# Calculate and print average treatment effect by farmer skill tertile
df['skill_tertile'] = pd.qcut(df['farmer_skill'], q=3, labels=['Low', 'Medium', 'High'])
print("\nAverage Treatment Effect by Farmer Skill Tertile:")
print(df.groupby('skill_tertile').apply(lambda x: x[x['treatment'] == 1]['outcome'].mean() - x[x['treatment'] == 0]['outcome'].mean()))