import numpy as np
import pandas as pd
from scipy import stats

np.random.seed(42)

n = 4000
treatment = np.random.binomial(1, 0.5, n)

# Generate farmer responsiveness (combines skill, land quality, microclimate)
farmer_responsiveness = np.random.gamma(2, 2, n)

# Base yield without treatment
base_yield = 100

# Treatment effect depends on farmer responsiveness
treatment_effect = 10 * farmer_responsiveness

# Generate weather impact (can be positive or negative)
weather_impact = np.random.normal(0, 10, n)

# Generate normally distributed noise
noise = np.random.normal(0, 5, n)

# Outcome: base yield + treatment effect + weather impact * responsiveness + noise
outcome = (base_yield + 
           treatment_effect * treatment + 
           weather_impact * farmer_responsiveness +
           noise)

df = pd.DataFrame({
    'treatment': treatment,
    'farmer_responsiveness': farmer_responsiveness,
    'outcome': outcome
})

print(df.groupby('treatment')['outcome'].describe())

# Calculate and print skewness and kurtosis
print("\nSkewness:", stats.skew(df['outcome']))
print("Kurtosis:", stats.kurtosis(df['outcome']))