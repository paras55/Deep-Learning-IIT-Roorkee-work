# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 13:12:09 2019

@author: Vikas
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 10:32:37 2019

@author: Vikas
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense
import seaborn as sns


dataset = pd.read_excel("frequency.xlsx")
X = dataset.iloc[:, 0:2].values
y = dataset.iloc[:, 2].values
sns.pairplot(dataset)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)



regressor = Sequential()

# Adding the input layer and the first hidden layer
regressor.add(Dense(units = 2, kernel_initializer = 'uniform', activation = 'relu', input_dim = 2))

# Adding the second hidden layer
regressor.add(Dense(units = 2, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the output layer
regressor.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'linear'))


regressor.compile(optimizer = 'RMSprop', loss = 'mean_squared_error', metrics = ['mean_absolute_error', 'mean_squared_error'])

history=regressor.fit(X_train, y_train, batch_size = 10, epochs = 1000)

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()


y_pred= regressor.predict(X_test).flatten()

plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()], [y_pred.min(), y_pred.max()])
plt.set_xlabel('Measured')
plt.set_ylabel('Predicted')
plt.show()