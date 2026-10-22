import pandas as pd 

import matplotlib.pyplot as plt 

from sklearn.model_selection import train_test_split 

from sklearn.linear_model import LinearRegression 

from sklearn.metrics import mean_squared_error, r2_score 

 

df = pd.read_csv("placement.csv") 

X = df[['cgpa']] 

y = df['package'] 

 

X_train, X_test, y_train, y_test = train_test_split( 

    X, y, test_size=0.2, random_state=42 

) 

 

model = LinearRegression() 

model.fit(X_train, y_train) 

pred = model.predict(X_test) 

 

print("Slope:", model.coef_[0]) 

print("Intercept:", model.intercept_) 

print("MSE:", mean_squared_error(y_test, pred)) 

print("R2 Score:", r2_score(y_test, pred)) 

 

plt.scatter(X_test, y_test, label="Actual") 

plt.plot(X_test, pred, color="red", label="Regression line") 

plt.xlabel("CGPA") 

plt.ylabel("Package (LPA)") 

plt.legend() 

plt.show()