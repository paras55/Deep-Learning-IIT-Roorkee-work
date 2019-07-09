import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense
import seaborn as sns
from keras import optimizers
import tkinter as tk 
import statsmodels.api as sm


dataset = pd.read_excel("delay_total.xlsx")
X = dataset.iloc[:, 0:2].values
y = dataset.iloc[:, 3].values
sns.pairplot(dataset)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)



regressor = Sequential()

# Adding the input layer and the first hidden layer
regressor.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 2))

# Adding the second hidden layer
regressor.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the output layer
regressor.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'linear'))


regressor.compile(optimizer = 'RMSprop', loss = 'mean_squared_error', metrics = ['accuracy'])
keras.optimizers.RMSprop(lr=0.01,rho=0.9,epsilon=None,decay=0.0)

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

# GUI BY TKINTER
root= tk.Tk() 
 
canvas1 = tk.Canvas(root, width = 1200, height = 450)
canvas1.pack()





# New_Interest_Rate label and input box
label1 = tk.Label(root, text='value of b ')
canvas1.create_window(100, 100, window=label1)

entry1 = tk.Entry (root) # create 1st entry box
canvas1.create_window(270, 100, window=entry1)


label2 = tk.Label(root, text=' Value of a ')
canvas1.create_window(120, 120, window=label2)

entry2 = tk.Entry (root) # create 2nd entry box
canvas1.create_window(270, 120, window=entry2)





def values(): 
    global New_Interest_Rate #our 1st input variable
    b = float(entry1.get()) 
    
    global New_Unemployment_Rate #our 2nd input variable
    a = float(entry2.get()) 
    
    Prediction_result  = ('Prediction: ', history.predict([[b,a]]))
    label_Prediction = tk.Label(root, text= Prediction_result, bg='orange')
    canvas1.create_window(260, 280, window=label_Prediction)
    
button1 = tk.Button (root, text='Predict',command=values, bg='orange') # button to call the 'values' command above 
canvas1.create_window(270, 150, window=button1)


root.mainloop()
values()