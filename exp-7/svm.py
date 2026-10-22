import pandas as pd 

import matplotlib.pyplot as plt 

from sklearn.model_selection import train_test_split 

from sklearn.svm import SVC 

from sklearn.preprocessing import StandardScaler 

from sklearn.metrics import accuracy_score 

 

df = pd.read_csv("Iris.csv") 

 

X = df[['SepalLengthCm', 'SepalWidthCm', 

        'PetalLengthCm', 'PetalWidthCm']] 

y = df['Species'] 

 

X_train, X_test, y_train, y_test = train_test_split( 

    X, y, test_size=0.2, random_state=42, stratify=y 

) 

 

scaler = StandardScaler() 

X_train = scaler.fit_transform(X_train) 

X_test = scaler.transform(X_test) 

 

model = SVC(kernel='linear') 

model.fit(X_train, y_train) 

pred = model.predict(X_test) 

 

print("Accuracy:", accuracy_score(y_test, pred)) 

 

plt.scatter(X_test[:, 0], X_test[:, 1], 

            c=pd.Categorical(pred).codes) 

plt.xlabel("Sepal Length (scaled)") 

plt.ylabel("Sepal Width (scaled)") 

plt.show() 