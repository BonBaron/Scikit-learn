import numpy as np
import pandas as pd
import joblib

data = pd.read_csv('test_data.csv') # open csv

data['target1'] = np.nan
data['target2'] = np.nan
data['target3'] = np.nan
data['target4'] = np.nan

X=data.drop(['Unnamed: 0','target1', 'target2', 'target3', 'target4'], axis=1)  # Throw out the column Unnamed: 0

X.fillna(0, inplace = True)
y = data[['target1', 'target2', 'target3','target4']]

print (X.shape)
print (y.shape)

model = joblib.load("train_model")
prediction = model.predict(X)
print (prediction)
#export to file
prediction = pd.DataFrame(prediction, columns=['target1', 'target2', 'target3','target4']).to_csv('Forecasts for parameters Target1...4.csv', index=False)
