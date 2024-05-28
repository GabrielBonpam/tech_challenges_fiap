import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


pd.set_option('display.max_columns', None)

df_fifa = pd.read_csv("data/players_22.csv")
df_fifa

df_fifa.describe()

df_fifa.head(10)

df_fifa.info()

df_fifa.shape

df_fifa_numerico = df_fifa.select_dtypes([np.number])

df_fifa.head(10)

# correlation_matrix = df_fifa_numerico.corr()
# correlation_matrix



# plt.figure(figsize = (10,8))
# sns.heatmap(correlation_matrix, cmap='coolwarm', annot=False, fmt='.2f', linewidths=.5)
# plt.title('Matrix de Correlação')
# plt.show()


# df_fifa_numerico.isnull().sum()

# imputer = SimpleImputer(strategy='mean')
# df_fifa_numerico = pd.DataFrame(imputer.fit_transform(df_fifa_numerico), columns=df_fifa_numerico.columns)

# scaler = StandardScaler()
# df_fifa_padronizado = scaler.fit_transform(df_fifa_numerico)

# pca = PCA()

# pca.fit(df_fifa_padronizado)
# variancia_cumulativa = np.cumsum(pca.explained_variance_ratio_)

# plt.plot(range(1, len(variancia_cumulativa)+1), variancia_cumulativa, marker='o')
# plt.xlabel('Número de Componentes Principais')
# plt.ylabel('Variância Acumulada Explicada')
# plt.title('Variância Acumulada Explicada pelo PCA')
# plt.show()


# limiar_de_variancia = 0.80

# num_de_pca = np.argmax(variancia_cumulativa >= limiar_de_variancia) + 1

# print(f"Número de Componentes para {limiar_de_variancia * 100}% da variância:{num_de_pca}")

# pca = PCA(n_components=num_de_pca)

# principal_componentes = pca.fit_transform(df_fifa_padronizado)

# explained_variance_ratio = pca.explained_variance_ratio_
# print(explained_variance_ratio)

# num_components = principal_componentes.shape[1]

# column_names = [f'PC{i}' for i in range(1, num_components + 1)]

# pca_df = pd.DataFrame(data=principal_componentes, columns=column_names)

# plt.figure(figsize=(15,8))
# for i, col in enumerate(pca_df.columns[:10]):
#     plt.subplot(2,5,i +1)
#     sns.histplot(pca_df[col],bins=20,kde=True)
#     plt.title(f'Histograma{col}')
# plt.tight_layout()
# plt.show()


# from scipy.stats import shapiro

# for column in pca_df.columns:
#     stat, p_value = shapiro(pca_df[column])
#     print(f"Variável: {column}, Estatística de teste: {stat}, Valor p: {p_value}" )

#     if p_value > 0.05:
#         print(f'A variável {column} parece seguir uma distribuição normal. \n')
    
#     else:
#         print(f'A variável {column} não parece seguir uma distribuição normal')



