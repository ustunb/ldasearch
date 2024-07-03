import sys
import os
sys.path.append(os.getcwd())

from unittest.mock import Mock
# Mocking matplotlib
sys.modules['matplotlib'] = Mock()
sys.modules['matplotlib.patches'] = Mock()
sys.modules['matplotlib.pyplot'] = Mock()
sys.modules['matplotlib.ticker'] = Mock()
sys.modules['matplotlib.axes'] = Mock()
sys.modules['matplotlib.gridspec'] = Mock()
sys.modules['matplotlib.transforms'] = Mock()
sys.modules['matplotlib.artist'] = Mock()
sys.modules['matplotlib.axis'] = Mock()
import pandas as pd
from optbinning import BinningProcess

df_raw = pd.read_csv("data/fico/fico_raw.csv")


# Display the first few rows of the dataframe
print(df_raw.head())

# Summary statistics and data types
print(df_raw.describe())
print(df_raw.info())


# Define special values
special_values = [-9, -8, -7]

# Separate features and target
X = df_raw.drop("RiskPerformance", axis=1)
y = df_raw["RiskPerformance"].apply(lambda x: 1 if x == 0 else 0)

# List of all feature names
feature_names = X.columns.tolist()

# Create the BinningProcess
binning_process = BinningProcess(feature_names, special_codes=special_values)

# Fit the binning process
binning_process.fit(X, y)

X_binned = binning_process.transform(X, metric="bins")

# Combine the binned features with the target variable
df_binned = pd.concat([y, X_binned], axis=1)

# Save the DataFrame to a CSV file
df_binned.to_csv("data/fico/fico_binned.csv", index=False)

