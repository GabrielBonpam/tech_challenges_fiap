import numpy as np
import pandas as pd


import os
for dirname, _, filenames in os.walk('C:\\Users\\gabriel.bonpam\\Downloads'):
    for filename in filenames:
        print(os.path.join(dirname,filename))


df = pd.read_csv("C:/Users/gabriel.bonpam/Downloads/fashion_products.csv")
df.head(20)

df.describe()


import matplotlib.pyplot as plt

import seaborn as sns

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
sns.histplot(df['Price'], kde=True, color='skyblue')
plt.title('Price Distribution')

plt.subplot(1,2,2)
sns.histplot(df['Rating'],kde=True,color='salmon')
plt.title('Rating Distribution')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,6))
sns.countplot(data=df, x='Category')
plt.title('Distribution of Products by Category')
plt.xlabel('Category')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Brand')
plt.title('Distribution of Products by Brand')
plt.xlabel('Brand')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(12,6))
sns.boxplot(data=df, x='Category',y = 'Rating', hue = 'Brand')
plt.title('Rating Distribution by Category and Brand')
plt.xlabel('Category')
plt.ylabel('Rating')
plt.xticks(rotation = 45)
plt.legend(title = 'Brand')
plt.show()


plt.figure(figsize=(12,6))
sns.boxplot(data=df, x='Category', y='Rating', hue = 'Brand')
plt.title('Rating Distributio by Category and Brand')
plt.xlabel('Category')
plt.ylabel('Rating')
plt.xticks(rotation = 45)
plt.legend(title = 'Brand')
plt.show()


plt.figure(figsize=(10,6))
sns.countplot(data=df, x='Color')
plt.title('Distribution of Products by Color')
plt.xlabel('Color')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(8,6))
sns.heatmap(df[['Price','Rating']].corr(), annot=True,cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix (Price vs Rating)')
plt.show()