import numpy as np
import pandas as pd
data = {'A': [1,2, np.nan, 4],
        'B':[np.nan, np.nan, np.nan,8],
        'C':[np.nan, 10, 11, 12]}
df = pd.DataFrame(data)
df_dropped_rows = df.dropna()
print(df_dropped_rows)
print("***************************")
print("***************************")
df_dropped_columns = df.dropna(axis=1)
print(df_dropped_columns)
