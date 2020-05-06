#Polynomial Regression

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

data = pd.read_csv(r"C:\Users\Rising IT\PycharmProjects\Pos_Salaries.csv")
#print(data)

x = data.iloc[:,1:2].values   #if only 1 written then it is vector not matrix
y = data.iloc[:,2].values
#x = x.reshape(-1,1)
#y = y.reshape(-1,1)
#print(y)

lin = LinearRegression()
lin.fit(x,y)

pol = PolynomialFeatures(degree = 5)  #default value 2, change to fit the values, 5 seemed to fit
x_pol = pol.fit_transform(x)
#print(x_pol)
pol.fit(x_pol,y)

lin2 = LinearRegression()
lin2.fit(x_pol,y)
pred_y = lin2.predict(x_pol)
pred_y1 = lin.predict(x)

plt.scatter(x,y)
plt.plot(x,pred_y)
#plt.show()

#print(pred_y[6])
print(lin2.predict(pol.fit_transform([[6.5]])))
print(lin.predict([[6.5]]))
#print(pred_y(6.5))
#print(lin2.predict(pol.fit_transform(6)))

#no test train split as the dataset is very small