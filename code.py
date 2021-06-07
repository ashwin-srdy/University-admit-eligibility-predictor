# IMPORTING LIBRARIES , DATASET
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
data=pd.read_csv('Admission_Predict.csv')
print(data.columns,data.shape)

# DATA PRE-PROCESSING
print(data.isnull().sum())
print(data.corr())
print(data.info())
print(data.describe())
print("University ratings:",set(data["University Rating"]),"Research:",set(data["Research"]))

#DATA VISUALISATION
sns.barplot(data["GRE Score"],data["CGPA"])
sns.barplot(data["SOP"],data["CGPA"])
sns.barplot(data["TOEFL Score"],data["CGPA"])

plt.scatter(data['GRE Score'],data['CGPA'])
plt.title('CGPA vs GRE Score')
plt.xlabel('GRE Score')
plt.ylabel('CGPA')
plt.show()

plt.scatter(data['CGPA'],data['SOP'])
plt.title('SOP for CGPA')
plt.xlabel('CGPA')
plt.ylabel('SOP')
plt.show()

plt.figure(figsize=(10, 10))
sns.heatmap(data.corr(), annot=True, linewidths=0.05, fmt= '.2f',cmap="magma")
plt.show()

data.Research.value_counts()
sns.countplot(x="University Rating",data=data)

data.Research.value_counts()
sns.countplot(x="University Rating",data=data

sns.barplot(x="University Rating", y="Chance of Admit ", data=data)

# DATA TRANSFORMATION
x=data.drop(['Serial No.','Chance of Admit '],axis=1)
y=data['Chance of Admit ']
print(x.shape,y.shape)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

from sklearn.preprocessing import MinMaxScaler
mms=MinMaxScaler()
x_train[x_train.columns]=mms.fit_transform(x_train[x_train.columns].values)
x_test[x_test.columns]=mms.transform(x_test[x_test.columns].values)

# MODEL BUILDING(RANDOM FOREST REGRESSOR)
from sklearn.ensemble import RandomForestRegressor
model=RandomForestRegressor()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
print(y_pred)
print(y_test)

#evaluation
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error
print('model score:',model.score(x_test,y_test))
print('Mean Absolute Error:', mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test, y_pred)))
print('roc score:',roc_auc_score(y_test>0.5, y_pred>0.5))
print('recall score:',recall_score(y_test>0.5, y_pred>0.5))

# MODEL BUILDING(LINEAR REGRESSION)
x1=data.drop(['Serial No.','Chance of Admit '],axis=1)
y1=data['Chance of Admit ']
x1_train,x1_test,y1_train,y1_test=train_test_split(x1,y1,test_size=0.2)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x1_train=sc.fit_transform(x1_train)
x1_test=sc.fit_transform(x1_test)

from sklearn.linear_model import LinearRegression
model1=LinearRegression()
model1.fit(x1_train,y1_train)
y1_pred=model1.predict(x1_test)
print('model score:',model1.score(x1_test,y1_test))
print('Mean Absolute Error:', mean_absolute_error(y1_test, y1_pred))  
print('Mean Squared Error:', mean_squared_error(y1_test, y1_pred))  
print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y1_test, y1_pred)))
print('roc score:',roc_auc_score(y1_test>0.5, y1_pred>0.5))
print('recall score:',recall_score(y1_test>0.5, y1_pred>0.5))

# MODEL BUILDING (LOGISTIC REGRESSION)
x2=data.iloc[:,1:8].values
y2=data.iloc[:,-1:].values
x2_train,x2_test,y2_train,y2_test=train_test_split(x1,y1,test_size=0.2)
y2_train=y2_train>0.5
y2_test=y2_test>0.5

from sklearn.linear_model import LogisticRegression
model2=LogisticRegression()
model2.fit(x2_train,y2_train)

#evaluation
from sklearn.metrics import accuracy_score,roc_auc_score,recall_score
y2_pred=model2.predict(x2_test)
print('model score:',model2.score(x2_test,y2_test))
print('roc score:',roc_auc_score(y2_test, y2_pred))
print('recall score:',recall_score(y2_test, y2_pred))
print(type(y2_test),type(y2_pred))

# SAVING THE MODEL
#Though the accuracy of Logistic regression model is more we prefer Random forest regressor if we also want the percentage of chance or else we can use Logistic regression model.
import pickle
pickle.dump(model,open('model.pkl','wb'))