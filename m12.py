import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline  # Import Pipeline class (not module)

# Create numeric preprocessing pipeline
numeric_processor = Pipeline(
    steps=[
        ('imputation_mean', SimpleImputer(missing_values=np.nan, strategy='mean')),
        ('scaler', StandardScaler())
    ]     
)

import pandas as pd

# Example dataset
data = pd.DataFrame({
    "Age": [25, 30, np.nan, 40, 50],
    "Salary": [40000, np.nan, 60000, 80000, 100000]
})

# Apply pipeline
processed_data = numeric_processor.fit_transform(data)

# Convert back to DataFrame
df_processed = pd.DataFrame(processed_data, columns=data.columns)

print(df_processed)

# Example usage:
# numeric_processor.fit_transform(your_numeric_data)