import sys
import os
sys.path.append(os.getcwd())

import pandas as pd
import os
from data import utils

abs_path = os.path.abspath(os.getcwd())
df = pd.read_csv(abs_path + "/data/fico/fico_encoded.csv")
df['RiskPerformance'] = df['RiskPerformance'].replace('Good', 1).values
df['RiskPerformance'] = df['RiskPerformance'].replace('Bad', 0).values

# # drop special characters
# df = df[~df.eq(-7).any(1)]
# df = df[~df.eq(-8).any(1)]
# df = df[~df.eq(-9).any(1)]

raw_file = abs_path + "/data/fico/fico_data.csv"
df.to_csv(raw_file, header = True, index = False)
print("results saved!")
