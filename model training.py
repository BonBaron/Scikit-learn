import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
#Алгоритм машинного обучения
from sklearn.ensemble import RandomForestRegressor
from sklearn import ensemble
#kNN – метод ближайших соседей
from sklearn.neighbors import KNeighborsRegressor
# Отключение предупреждений Pandas
pd.options.mode.chained_assignment = None
#Используется для записи модели в файл
import joblib
from sklearn.metrics import r2_score

data = pd.read_csv("train_data_200k.csv")

X = data.drop(['target1', 'target2', 'target3', 'target4', 'Unnamed: 0'], axis=1)  # Выбрасываем столбцы target и Unnamed: 0
y = data[['target1', 'target2', 'target3','target4']]
X.fillna(X.median(axis=0), axis=0, inplace = True)
y.fillna(0, inplace = True)
feature_names = X.columns
print (feature_names)
#print (X.head)
#print (y.head)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 11)

N_train, _ = X_train.shape 
N_test,  _ = X_test.shape 
print (N_train, N_test)


#kNN – метод ближайших соседей
knn = KNeighborsRegressor()
knn.fit(X_train, y_train)

y_true = knn.predict(X_train)
y_pred = knn.predict(X_test)

print('Model Accuracy:', r2_score(y_true, y_pred))
'''
rf = ensemble.RandomForestRegressor(n_estimators=100)
rf.fit(X_train, y_train)

importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

print("Feature importances:")
for f, idx in enumerate(indices):
    print("{:2d}. feature '{:5s}' ({:.4f})".format(f + 1, feature_names[idx], importances[idx]))
    
err_train = np.mean(y_train != rf.predict(X_train))
err_test  = np.mean(y_test  != rf.predict(X_test))
print (err_train, err_test)

print('Model Accuracy:', rf.score(X, y))
#joblib.dump(rf, "train_model", compress=9)# Запись модели в файл
'''
#print('Model Accuracy:', knn.score(X, y))
