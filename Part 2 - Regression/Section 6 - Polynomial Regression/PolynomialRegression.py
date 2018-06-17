#polynomial regression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#getting the dataset
dataset = pd.read_csv("Position_Salaries.csv")
X = dataset.iloc[:,1:2]
y = dataset.iloc[:,2]

#fitting linear regression in the model
from sklearn.linear_model import LinearRegression
lin_reg_1 = LinearRegression()
lin_reg_1.fit(X, y)

#fitting Polynomial Regression in the model
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X, y)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

#visualising the Linear Regression Model
plt.scatter(X, y, color = 'red')
plt.plot(X,lin_reg_1.predict(X), color = 'blue')
plt.title('Truth or Bluff(Lineqr Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

#visualising the Polyomial Regression model
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid,lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title('Truth or Bluff(Lineqr Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()