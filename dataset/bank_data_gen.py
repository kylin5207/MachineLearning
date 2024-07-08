from ucimlrepo import fetch_ucirepo
import pandas as pd

# fetch dataset
bank_marketing = fetch_ucirepo(id=222)

# data (as pandas dataframes)
X = bank_marketing.data.features
y = bank_marketing.data.targets

# concat features and label
data = pd.concat([X,y], axis=1)
print(data.head())
print(data.shape)
data.to_csv("bank_market.csv", index=False)
