import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit

np.random.seed(42)

df_housing = pd.read_csv("C:/Users/gabriel.bonpam/Downloads/CURSO_IA_ML-main/Aula 2/housing.csv")

# df_housing.head(5)

# df_housing.shape

# df_housing.corr()

# df_housing.info()

df_housing.describe()

#df_housing.hist(bins=100, figsize=(20,15))

df_train, df_test = train_test_split(df_housing, test_size=0.2, random_state=7)

print(len(df_train))
print(len(df_test))

#df_housing['median_income'].hist()

df_housing['income_cat'] = np.ceil(df_housing['median_income']/1.5)

df_housing['income_cat'].where(df_housing['income_cat'] < 5, 5.0, inplace=True)

df_housing['income_cat'] = pd.cut(df_housing['median_income'],
                                  bins=[0.,1.5,3.0,4.5,6.,np.inf],
                                  labels=[1,2,3,4,5])

df_housing['income_cat'].value_counts()

#df_housing['income_cat'].hist()

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(df_housing, df_housing['income_cat']):
    strat_train_set = df_housing.loc[train_index]
    strat_test_set = df_housing.loc[test_index]


#strat_test_set['income_cat'].value_counts() / len(strat_test_set)

#strat_train_set['income_cat'].value_counts() / len(strat_train_set)

#df_housing['income_cat'].value_counts() / len(df_housing)

# for set_ in(strat_train_set, strat_test_set):
#     set_.drop('income_cat', axis = 1, inplace = True)

# housing_train = strat_train_set.copy()

# housing_train.plot(kind = "scatter", x = "longitude", y = "latitude")

# housing_train.plot(kind="scatter",x ="longitude",y="latitude",alpha=0.1)

# housing_train.plot(kind="scatter",x="longitude",y="latitude", alpha=0.4,
#                 s=housing_train['population']/100, label="population", figsize=(10,7),
#                 c="median_house_value", cmap = plt.get_cmap('jet'), colorbar=True,
#                 sharex=False)
# plt.legend()

corr_matrix = df_housing.corr()

corr_matrix["median_house_value"].sort_values(ascending=False)

from pandas.plotting import scatter_matrix

attributes = ["median_house_value", "median_income", "total_rooms","housing_median_age"]

scatter_matrix(df_housing[attributes], figsize=(12,8))

df_housing.plot(kind="scatter", x="median_income", y="median_house_value",alpha = 0.1)
plt.axis((0,16,0,550000))



