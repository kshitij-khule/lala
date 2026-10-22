import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import r2_score 

 
df = pd.read_csv("Placement_Data_Full_Class (1).csv") 

 
X = df[['ssc_p', 'hsc_p', 'etest_p']] 

y = df[['degree_p', 'mba_p']] 

 

X_train, X_test, y_train, y_test = train_test_split( 

    X, y, test_size=0.2, random_state=42 

) 

 

model = LinearRegression() 

model.fit(X_train, y_train) 

pred = model.predict(X_test) 

 

print("Coefficients:\n", model.coef_) 

print("R2 Score:", r2_score(y_test, pred, multioutput='uniform_average')) 

 

plt.scatter(y_test['degree_p'], pred[:, 0], label="Degree") 

plt.scatter(y_test['mba_p'], pred[:, 1], label="MBA") 

plt.xlabel("Actual Marks") 

plt.ylabel("Predicted Marks") 

plt.legend() 

plt.show()