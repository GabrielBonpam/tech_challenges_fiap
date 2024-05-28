from sklearn import datasets
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

iris = datasets.load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['Target'] = iris.get('target')
df.head()

features = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
X = df[features].values
y = df['Target'].values

X = StandardScaler().fit_transform(X)

df_padronizado = pd.DataFrame(data=X, columns=features)
df_padronizado.head()

pca = PCA(n_components=2)

principalComponents = pca.fit_transform(X)

df_pca = pd.DataFrame(data = principalComponents,
                      columns=['PC1','PC2'])

target = pd.Series(iris['target'], name='target')
result_df = pd.concat([df_pca, target], axis=1)
result_df

print("Variance of each component: ", pca.explained_variance_ratio_)
print("Total Variance Explained: ", round(sum(list(pca.explained_variance_ratio_))*100,2))


import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

results = []

X = df_padronizado

for n in range(2,5):
    pca = PCA(n_components=n)
    pca.fit(X)
    explained_variance = np.sum(pca.explained_variance_ratio_)
    results.append(explained_variance)


plt.figure(figsize=(15,8))
plt.plot(range(2,5), results, marker='o', linestyle='-',color='b')
plt.xlabel('Número de Componentes')
plt.ylabel('Variância Explicada Cumulativa (%)')
plt.title('Variância Explicada Cumulativa por Componentes do PCA')
plt.grid(True)

for i, (n_components,explained_var) in enumerate(zip(range(2,5),results)):
    plt.text(n_components,explained_var, f"Componente {n_components}", ha="right", va="bottom")
    
plt.show()