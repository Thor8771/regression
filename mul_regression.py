import pandas as pd
from sklearn import linear_model

data=pd.read_csv('insurance1.csv')
# print(data)
mean_height=data.height.mean()
data.height=data.height.fillna(mean_height)
model=linear_model.LinearRegression()
model.fit(data[['Age','height','weight']],data['Premium'])
mo=model.predict([[52,100,100]])
print(mo)