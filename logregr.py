
#importing the modules
import pandas as pd
import numpy as np

#reading data
data=pd.read_csv("Social_Network_Ads.csv")
x=data.iloc[:,2:4].values
y=data.iloc[:,4].values


#splitting,training and predicting
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)
from sklearn.linear_model import LogisticRegression
logregr=LogisticRegression()
logregr.fit(x_train,y_train)
pred=logregr.predict(x_test)

#confusion matrix andd aaccuracy
from sklearn.metrics import confusion_matrix,accuracy_score
score=confusion_matrix(y_test,pred)
print(score)
print(accuracy_score(y_test,pred))