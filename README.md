# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard libraries.

2.Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively.

3.Import LabelEncoder and encode the dataset.

4.Import LogisticRegression from sklearn and apply the model on the dataset.

5.Predict the values of array.

6.Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.

7.Apply new unknown values
## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: sanjay m
RegisterNumber: 212222110038 
*/

import pandas as pd
data=pd.read_csv('Placement_Data.csv')
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
data1


x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")#A Library for Large Linear Classification
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1=classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```

## Output:
![the Logistic Regression Model to Predict the Placement Status of Student](sam.png)
##placement data
![image](https://github.com/AkilaMohan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119406959/d956dbf5-171d-45c8-b538-dbfb94214471)

##salary data
![image](https://github.com/AkilaMohan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119406959/ea5a1dbe-5c40-48cf-87a5-cd2cd534ddbb)

##isnull()
![image](https://github.com/AkilaMohan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119406959/df9ade02-7ea2-410f-9747-cb2981767235)

##DUPLICATED()
![image](https://github.com/AkilaMohan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119406959/5705a5ab-f7be-4945-bccc-fcad4485a222)

##Print Data
![image](https://github.com/AkilaMohan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119406959/0f228b5f-a8b6-4751-a9de-a5ca3fb0e6b2)

##iloc[:,:-1]
![image](https://github.com/AkilaMohan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119406959/1397d2cf-f593-47e0-a326-408a60b0d5f5)

##Data Status
![image](https://github.com/AkilaMohan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119406959/5eb0c256-b48c-4973-9599-50d56cba67d2)

##Y Prediction array:
![image](https://github.com/AkilaMohan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119406959/1600e646-2697-42f7-b264-41261341799d)

##Accuray value:
![image](https://github.com/AkilaMohan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119406959/7c616072-c2ad-471c-848a-bfba236c1f81)

##Confusion Array:
![image](https://github.com/AkilaMohan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119406959/312f0502-c469-4e23-a881-9fc1c65fb814)

##Classification report:
![image](https://github.com/AkilaMohan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119406959/ad9956ab-2371-4dfa-bcfb-7073bf5490f9)

##Prediction of LR:
![image](https://github.com/AkilaMohan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119406959/3ff992ee-5c67-4dd6-9382-a1d3052b1d1c)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
