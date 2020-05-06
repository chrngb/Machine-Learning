import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
#import statsmodels.regression.linear_model as lm

data = pd.read_csv(r"C:\Users\Rising IT\PycharmProjects\startups.csv")
#print(data.head(5))

real_x = data.iloc[:,0:4].values #column 0,1,2,3 as independent x variables
real_y = data.iloc[:,4].values  #column 4 as dependent y variable
#print(real_x)

ct = ColumnTransformer([('name', OneHotEncoder(), [3])], remainder='passthrough')
real_x = ct.fit_transform(real_x) #transform column 3 values to 0 & 1 # three diff columns so 3 dummy variables are there in 3 columns
#print(real_x)
#print(data.head(5))

real_x = real_x[:,1:]   # used to eliminate one dummy variable

#print(mlr.coef_)
#print(mlr.intercept_)

real_x = np.append(arr=np.ones((50,1)).astype(int),values=real_x,axis=1) #total 50 rows #append merges array
#print(real_x)

real_x = real_x[:,[0,3]]  #changed at last by elimination
#print(real_x)

training_x,testing_x,training_y,testing_y = train_test_split(real_x, real_y,test_size=0.2,random_state=0)
mlr = LinearRegression()
mlr.fit(training_x,training_y)

pred_y = mlr.predict(testing_x)
print(testing_y)
print(pred_y)

x_opt = np.array(real_x[:,[0,1,2,3,4,5]],dtype=float)
reg_OLS = sm.OLS(endog=real_y, exog=x_opt).fit()
#print(reg_OLS.summary())

x_opt = np.array(real_x[:,[0,1,3,4,5]],dtype=float) #index no 2 x2 has highest p value so it is removed
reg_OLS = sm.OLS(endog=real_y, exog=x_opt).fit()
#print(reg_OLS.summary())

x_opt = np.array(real_x[:,[0,3,4,5]],dtype=float) #index no 1 is removed x1
reg_OLS = sm.OLS(endog=real_y, exog=x_opt).fit()
#print(reg_OLS.summary())

x_opt = np.array(real_x[:,[0,3,5]],dtype=float) #index no 2 is removed x2
reg_OLS = sm.OLS(endog=real_y, exog=x_opt).fit()
#print(reg_OLS.summary())

x_opt = np.array(real_x[:,[0,3]],dtype=float) #index no 2 is removed x2
reg_OLS = sm.OLS(endog=real_y, exog=x_opt).fit()
#print(reg_OLS.summary())



