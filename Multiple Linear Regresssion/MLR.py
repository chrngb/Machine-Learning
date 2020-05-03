import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = pd.read_csv(r"C:\Users\Rising IT\PycharmProjects\startups.csv")
#print(data.head(5))

real_x = data.iloc[:,0:4].values #column 0,1,2,3 as independent x variables
real_y = data.iloc[:,4].values  #column 4 as dependent y variable
#print(real_x)

ct = ColumnTransformer([('name', OneHotEncoder(), [3])], remainder='passthrough')
real_x = ct.fit_transform(real_x) #transform column 3 values to 0 & 1
#print(real_x)

training_x,testing_x,training_y,testing_y = train_test_split(real_x, real_y,test_size=0.2,random_state=0)
mlr = LinearRegression()
mlr.fit(training_x,training_y)

pred_y = mlr.predict(testing_x)
print(testing_y)
print(pred_y)

#print(mlr.coef_)
#print(mlr.intercept_)
