# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Import the libraries and read the data frame using pandas.

Calculate the null values present in the dataset and apply label encoder.

Determine test and training data set and apply decison tree regression in dataset.

calculate Mean square error,data prediction and r2.

## Program:
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Manisha selvakumari.S.S.
RegisterNumber:  212223220055
*/
```
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics

data = pd.read_csv("Salary.csv")
print(data.head(), "\n")
print(data.info(), "\n")
print(data.isnull().sum(), "\n")

le = LabelEncoder()
data["Position"] = le.fit_transform(data["Position"])
print(data.head(), "\n")

x = data[["Position", "Level"]]
y = data["Salary"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

dt = DecisionTreeRegressor(random_state=2)
dt.fit(x_train, y_train)

y_pred = dt.predict(x_test)

mse = metrics.mean_squared_error(y_test, y_pred)
r2 = metrics.r2_score(y_test, y_pred)

print(mse)
print(r2)

print(dt.predict([[5, 6]])[0])
```

## Output:

![Screenshot 2025-05-25 184506](https://github.com/user-attachments/assets/d03338cf-a1b1-4e6f-afa2-1648ff2e2e6d)

![Screenshot 2025-05-25 184529](https://github.com/user-attachments/assets/65c00c39-73f7-47c1-b5e9-e921ebb7effa)

![Screenshot 2025-05-25 184520](https://github.com/user-attachments/assets/3cbc0542-f75d-4f3f-95cf-ae6656655207)

![Screenshot 2025-05-25 184546](https://github.com/user-attachments/assets/3e51d587-84df-4851-9a5f-ea30594368e7)

![Screenshot 2025-05-25 184558](https://github.com/user-attachments/assets/86f2b637-aa21-4d28-b10e-dd87a75f6f07)



## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
