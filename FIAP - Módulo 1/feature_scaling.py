import pandas as pd 
import matplotlib.pyplot as plt

df = pd.read_csv('data/Churn_Modelling.csv', delimiter=';')

df.head(5)

#df.shape

plt.boxplot(df['CreditScore'])
plt.title('CreditScore')
plt.ylabel('Valores')
plt.show()


print(df['CreditScore'].min())
print(df['CreditScore'].max())

plt.boxplot(df['Age'])
plt.title('Age')
plt.ylabel('Valores')
plt.show()

plt.boxplot(df['Tenure'])
plt.title('Tenure')
plt.ylabel('Valores')
plt.show()


plt.boxplot(df['Balance'])
plt.title('Balance')
plt.ylabel('Valores')
plt.show()

plt.boxplot(df['NumOfProducts'])
plt.title('NumOfProducts')
plt.ylabel('Valores')
plt.show()

print(df['NumOfProducts'].min())
print(df['NumOfProducts'].max())

plt.boxplot(df['EstimatedSalary'])
plt.title('EstimatedSalary')
plt.ylabel('Valores')
plt.show()

print(df['EstimatedSalary'].min())
print(df['EstimatedSalary'].max())

from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

df['Surname'] = label_encoder.fit_transform(df['Surname'])
df['Geography'] = label_encoder.fit_transform(df['Geography'])
df['Gender'] = label_encoder.fit_transform(df['Gender'])

df.head(5)

from sklearn.model_selection import train_test_split

X= df.drop(columns=['Exited'])
y= df['Exited']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.preprocessing import StandardScaler, MinMaxScaler

scaler = MinMaxScaler()

scaler.fit(X_train)

x_train_min_max_scaled = scaler.transform(X_train)
x_test_min_max_scaled = scaler.transform(X_test)

from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=3)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)
print(f'Acurácia:  {accuracy:.2f}')

model_min_max = KNeighborsClassifier(n_neighbors=3)

model_min_max.fit(x_train_min_max_scaled,y_train)

y_pred_min_max = model.predict(x_test_min_max_scaled)

accuracy_min_max = accuracy_score(y_test, y_pred_min_max)
print(f'Acurácia: {accuracy_min_max:.2f}')



