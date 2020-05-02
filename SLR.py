import pandas as pd
import matplotlib.pyplot as plt
#import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = pd.read_csv(r"C:\Users\Rising IT\PycharmProjects\company.csv")
#print(data.head)

real_x = data.iloc[:,0].values
real_y = data.iloc[:,1].values
#print(real_y)
real_x = real_x.reshape(-1,1) #convert to 2D array
real_y = real_y.reshape(-1,1)

training_x,testing_x,training_y,testing_y = train_test_split(real_x, real_y,test_size=0.3,random_state=0)
#70% for training

Lin = LinearRegression()
Lin.fit(training_x,training_y)

pred_y = Lin.predict(testing_x)
print(testing_y[3])
print(pred_y[3])

plt.scatter(training_x,training_y)
plt.plot(training_x,Lin.predict(training_x))
plt.show()

plt.scatter(testing_x,testing_y)
plt.plot(training_x,Lin.predict(training_x))
plt.show()


