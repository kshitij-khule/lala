import pandas as pd 

import matplotlib.pyplot as plt 

from sklearn.model_selection import train_test_split 

from sklearn.linear_model import LogisticRegression 

from sklearn.metrics import accuracy_score, confusion_matrix 

 

df = pd.read_csv("Placement_Data_Full_Class (1).csv") 

 

X = df[['ssc_p', 'hsc_p', 'degree_p', 'etest_p', 'mba_p']] 

y = df['status'].map({'Not Placed': 0, 'Placed': 1}) 

 

X_train, X_test, y_train, y_test = train_test_split( 

    X, y, test_size=0.2, random_state=42, stratify=y 

) 

 

model = LogisticRegression(max_iter=1000) 

model.fit(X_train, y_train) 

pred = model.predict(X_test) 

 

print("Accuracy:", accuracy_score(y_test, pred)) 

print("Confusion Matrix:\n", confusion_matrix(y_test, pred)) 

 

plt.scatter(range(len(y_test)), y_test, label="Actual") 

plt.scatter(range(len(pred)), pred, label="Predicted") 

plt.legend() 

plt.show()