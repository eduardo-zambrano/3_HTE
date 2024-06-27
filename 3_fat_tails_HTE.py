import numpy as np
import pandas as pd
from scipy import stats

np.random.seed(42)

# Generate basic data
n = 4000
treatment = np.random.binomial(1, 0.5, n)

# Generate farmer characteristic (e.g., skill level)
farmer_skill = np.random.normal(0, 1, n)

# Generate heterogeneous treatment effect
base_effect = 5
het_effect = 2 * farmer_skill

# Generate error term with fat tails (t-distribution)
error = stats.t.rvs(df=3, size=n)

# Generate outcome
outcome = 100 + base_effect * treatment + het_effect * treatment + error

# Create DataFrame
df = pd.DataFrame({
    'treatment': treatment,
    'farmer_skill': farmer_skill,
    'outcome': outcome
})

# Print summary statistics
print(df.groupby('treatment')['outcome'].describe())