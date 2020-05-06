#Decision Tree

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

data = pd.read_csv(r"C:\Users\Rising IT\PycharmProjects\Pos_Salaries.csv")
#print(data)

x = data.iloc[:,1:2].values   #if only 1 written then it is vector not matrix
y = data.iloc[:,2].values

reg = DecisionTreeRegressor(random_state=0)
reg.fit(x,y)
y_pred = reg.predict([[6.5]])
print(y_pred)

x_grid = np.arange(min(x),max(x),0.01)
x_grid = x_grid.reshape(-1,1)
plt.scatter(x,y)
plt.plot(x_grid,reg.predict(x_grid))
plt.show()

