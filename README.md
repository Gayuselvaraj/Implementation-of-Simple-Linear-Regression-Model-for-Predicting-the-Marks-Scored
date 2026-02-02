# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries

2.Set variables for assigning dataset values.

3.Import linear regression from sklearn

4.Assign the points for representing in the graph\

5.Predict the regression for marks by using the representation of the graph

6.Compare the graphs and hence we obtained the linear regression for the given datas.
## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: GAYATHRI S
RegisterNumber:  212224230073
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv("C:/Users/SMARTLINK/Downloads/student_scores.csv")
df.head()

df.tail()

X=df.iloc[:,:-1].values
X

Y=df.iloc[:,1].values
Y

#spilitting training and test data
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)

#displaying predicted values
Y_pred

Y_test

#graph plot for training data
plt.scatter(X_train,Y_train,color="red")
plt.plot(X_train,regressor.predict(X_train),color="blue")
plt.title("Hours vs Scores(Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

plt.scatter(X_test,Y_test,color='green')
plt.plot(X_train,regressor.predict(X_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)

mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)

rmse=np.sqrt(mse)
print('RMSE = ',rmse)
```

## Output:
<img width="894" height="288" alt="Screenshot 2026-02-02 083530" src="https://github.com/user-attachments/assets/780fee0c-42ff-4392-810d-3173bfde0a5f" />


<img width="815" height="257" alt="image" src="https://github.com/user-attachments/assets/00aaeb97-f66a-4415-843f-94fcd9ba7625" />


![image](https://github.com/user-attachments/assets/7896a547-4222-4cf9-8267-574447b2b3db)


<img width="490" height="121" alt="Screenshot 2026-02-02 083741" src="https://github.com/user-attachments/assets/6c8051f5-4175-45ec-825a-6e77d25b4093" />










<img width="503" height="110" alt="Screenshot 2026-02-02 083747" src="https://github.com/user-attachments/assets/57059542-c1bf-4d51-a55c-002ae78efd22" />











<img width="480" height="94" alt="Screenshot 2026-02-02 083751" src="https://github.com/user-attachments/assets/91dea0c0-8c33-4ae6-96f4-f980ecb422f1" />








<img width="967" height="693" alt="image" src="https://github.com/user-attachments/assets/c30a3d83-0938-47e2-ae15-fd23aac79b5c" />


<img width="873" height="641" alt="image" src="https://github.com/user-attachments/assets/e1fd18b2-a33e-40de-b3bc-fa86a7359977" />


<img width="339" height="66" alt="image" src="https://github.com/user-attachments/assets/517cb972-f5ef-421e-a7c7-9b447de75581" />



<img width="521" height="313" alt="image" src="https://github.com/user-attachments/assets/c7ba6841-1697-41df-be06-2492f0a6373c" />




## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
