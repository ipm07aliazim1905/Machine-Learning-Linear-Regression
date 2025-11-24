# Machine-Learning-Linear-Regression
import pandas as pd
import os
import matplotlib.pyplot as plt

from sklearn import linear_model
from sklearn.linear_model import LinearRegression


project_path = os.getcwd()
print('project path=', project_path)

file_path = project_path + '\\Salary_dataset.xlsx'
print('file path= ', file_path)

df = pd.read_excel(file_path)
print(df)


y = df.drop(['YearsExperience'],axis=1)
x = df.drop(['Salary'],axis=1)

model = LinearRegression()
model.fit(x,y)

print(model.predict(x))
print("b0 =", model.intercept_)
print("b1 =", model.coef_)


plt.scatter(x,y)
plt.plot(x,y,color='green', label='Regression Line')
plt.xlabel('----Years of Experience--->')
plt.ylabel('----Salary--->')
plt.title('YearsExperience-Salary Regression Plot')
plt.grid()
plt.show()
