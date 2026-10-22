
import pandas as pd 

import matplotlib.pyplot as plt 

from sklearn.model_selection import train_test_split 

from sklearn.linear_model import LinearRegression 

from sklearn.metrics import r2_score 

 

df = pd.read_csv("Placement_Data_Full_Class (1).csv") 

df = df[df['status'] == 'Placed'].dropna(subset=['salary']) 

 

X = df[['ssc_p', 'hsc_p', 'degree_p', 'etest_p', 'mba_p']] 

y = df['salary'] 

 

X_train, X_test, y_train, y_test = train_test_split( 

    X, y, test_size=0.2, random_state=42 

) 

 

model = LinearRegression() 

model.fit(X_train, y_train) 

pred = model.predict(X_test) 

 

print("Coefficients:", model.coef_) 

print("R2 Score:", r2_score(y_test, pred)) 

 

plt.scatter(y_test, pred) 

plt.xlabel("Actual Salary") 

plt.ylabel("Predicted Salary") 

plt.show() 