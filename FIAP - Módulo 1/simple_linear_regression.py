import pandas as pd 
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

df_icecream = pd.read_excel("C:/Users/gabriel.bonpam/Downloads/CURSO_IA_ML-main/Aula 2/Sorvete.xlsx")
df_icecream.head()



plt.scatter(df_icecream['Temperatura'], df_icecream['Vendas_Sorvetes'])
plt.xlabel("Temperatura(ºC)")
plt.ylabel("Vendas de Sorvetes (Milhares)")
plt.title('Relação entre Temperatura e Vendas de Sorvetes')
plt.show()

df_icecream.corr()

X = df_icecream[['Temperatura']]
y = df_icecream['Vendas_Sorvetes']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train.shape

X_test.shape

modelo = LinearRegression()
modelo.fit(X_train, y_train)

predictions = modelo.predict(X_test)

erro_medio_quadratico = mean_squared_error(y_test,predictions)
erro_absoluto_medio = mean_absolute_error(y_test, predictions)
r_quadrado = r2_score(y_test, predictions)

print(f"Erro Médio Quadratico: {erro_medio_quadratico}")
print(f"Erro Absoluto Medio: {erro_absoluto_medio}")
print(f"R²: {r_quadrado}")

plt.scatter(X_test, y_test, label = "Real")
plt.scatter(X_test, predictions, label = "Previsto", color = "red")
plt.ylabel("Temperatura: (°C)")
plt.title("Previsões do Modelo em Regressão Linear")
plt.legend()
