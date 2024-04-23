# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student->

## AIM :
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required :
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm :

1.Import the standard libraries.

2.Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively.

3.Import LabelEncoder and encode the dataset.

4.Import LogisticRegression from sklearn and apply the model on the dataset.

5.Predict the values of array.

6.Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.

7.Apply new unknown values

## PROGRAM :
#### Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
#### Developed by : sanjay m
#### RegisterNumber : 212222110038
```
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
## OUTPUT:

### Placement_data
![image](https://github.com/Pradeeppachiyappan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118707347/2d30f6e7-146a-4759-b90e-6e50676a2bc8)

### Salary_data
![image](https://github.com/Pradeeppachiyappan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118707347/4e7823f4-5a70-4635-b359-e4551c14b756)

### ISNULL()
![image](https://github.com/Pradeeppachiyappan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118707347/b9f45afa-0d93-421f-9d73-d3fc072250e6)

### DUPLICATED()
![image](https://github.com/Pradeeppachiyappan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118707347/dcf0bf6e-543f-41d2-82f8-2cb93777173e)

### Print Data
![image](https://github.com/Pradeeppachiyappan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118707347/4659883d-f53a-4d4d-8183-8b36cd480d45)

### iloc[:,:-1]
![image](https://github.com/Pradeeppachiyappan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118707347/dd8c3742-8749-4cac-ad58-cdf86433b80b)

### Data_Status
![image](https://github.com/Pradeeppachiyappan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118707347/af901a2d-25fe-493e-9d6c-ec33cbf23e90)

### Y_Prediction array:
![image](https://github.com/Pradeeppachiyappan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118707347/50a32c48-d67d-49ab-9e0c-20140d857436)

### Accuray value:
![image](https://github.com/Pradeeppachiyappan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118707347/77b946f3-a6b2-4a98-8566-ca6878cacad6)

### Confusion Array:
![image](https://github.com/Pradeeppachiyappan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118707347/7edbb883-e626-428b-9648-e8726cb252e5)

### Classification report:
![image](https://github.com/Pradeeppachiyappan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118707347/05e10820-342d-4c64-b51d-c73795e46d92)

### Prediction of LR:
![image](https://github.com/Pradeeppachiyappan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118707347/05562708-e9ca-43a3-8981-a5834e4a2948)


## RESULT :
Thus,the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
