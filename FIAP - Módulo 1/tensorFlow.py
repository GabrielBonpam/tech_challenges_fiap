# import tensorflow_decision_forests as tfdf

# import pandas as pd

# dataset = pd.read_csv("C:/Users/gabriel.bonpam/Downloads/dataset.csv")
# tf_dataset = tfdf.keras.pd_dataframe_to_tf_dataset(tf_dataset, label = "my_label")

# model = tfdf.keras.RandamForestModel()
# model.fit(tf_dataset)

# print(model.summary())


import tensorflow as tf
import tensorflow_decision_forests as tfdf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

print("TensorFlow v" + tf.__version__)
print("TensorFlow Decision Forests v" + tf.__version__)

train_file_path = "C:/Users/gabriel.bonpam/Downloads/train.csv"
dataset_df = pd.read_csv(train_file_path)
print("Full train dataset shape is {}".format(dataset_df.shape))

dataset_df.head(3)


dataset_df = dataset_df.drop('Id', axis = 1)
dataset_df.head(3)

dataset_df.info()

print(dataset_df['SalePrice'].describe())
plt.figure(figsize=(9,8))
sns.distplot(dataset_df['SalePrice'], color = 'g', bins=200,hist_kws={'alpha': 0.5})

list(set(dataset_df.dtypes.tolist()))

df_num = dataset_df.select_dtypes(include = ['float64', 'int64'])
df_num.head()

df_num.hist(figsize=(20,24), bins = 50, xlabelsize = 8, ylabelsize=8)


import numpy as np

def split_dataset(dataset, test_ratio = 0.30):
    test_indices = np.random.rand(len(dataset)) < test_ratio
    return dataset[~test_indices], dataset[test_indices]

train_ds_pd, valid_ds_pd = split_dataset(dataset_df)
print("{} examples in training, {} examples in testing.".format(
    len(train_ds_pd), len(valid_ds_pd)
))

label = 'SalePrice'
train_df = tfdf.keras.pd_dataframe_to_tf_dataset(train_ds_pd,label=label, task = tfdf.keras.Task.REGRESSION)
valid_ds = tfdf.keras.pd_dataframe_to_tf_dataset(valid_ds_pd, label = label, task = tfdf.keras.Task.REGRESSION)




