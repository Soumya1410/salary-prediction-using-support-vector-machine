#1 Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

X = np.array([1,2,3,4,5,6,7,8,9,10]).astype(float)
y =np.array([45000, 50000, 60000, 80000, 110000, 150000, 200000, 300000, 500000, 1000000]).astype(float)
X = np.expand_dims(X, axis=-1)
y = np.expand_dims(y, axis=-1)

print(X.shape,' ',y.shape)

plt.scatter(X, y, color='r')
plt.show()
#3 Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)

plt.scatter(X, y, color='r')
plt.show()

# Fitting the Support Vector Regression Model to the dataset
from sklearn.svm import SVR

# most important SVR parameter is Kernel type. It can be #linear,polynomial or gaussian SVR. We have a non-linear condition #so we can select polynomial or gaussian but here we select RBF(a #gaussian type) kernel.
regressor = SVR(kernel='rbf')
regressor.fit(X,y)
X_grid = np.arange(min(X), max(X), 0.01) #this step required because data is feature scaled.
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
#5 Predicting a new result
ip = sc_X.transform(np.array([[6.5]]))
y_pred = regressor.predict (ip)
print('Predicted salary (normalized):',y_pred)
y_pred = sc_y.inverse_transform([y_pred])
print('Predicted salary',y_pred)

#output: 
#Predicted salary (normalized): [-0.27861589]
#Predicted salary [[170370.0204065]]
