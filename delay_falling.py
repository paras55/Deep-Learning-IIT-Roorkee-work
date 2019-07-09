# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 12:47:59 2019

@author: Vikas
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense
import seaborn as sns
from keras import optimizers


dataset = pd.read_excel("delay_falling.xlsx")
X = dataset.iloc[:, 0:2].values
y = dataset.iloc[:, 3].values
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


regressor.compile(optimizer = 'RMSprop', loss = 'mean_squared_error', metrics = ['accuracy'])
keras.optimizers.RMSprop(lr=0.0001,rho=0.9,epsilon=None,decay=0.0)

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

regressor.save('delay_falling.h5')