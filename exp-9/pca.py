import pandas as pd 

import matplotlib.pyplot as plt 

from sklearn.decomposition import PCA 

from sklearn.preprocessing import StandardScaler 

 

df = pd.read_csv("Iris.csv") 

 

X = df[['SepalLengthCm', 'SepalWidthCm', 

        'PetalLengthCm', 'PetalWidthCm']] 

y = df['Species'] 

 

X_scaled = StandardScaler().fit_transform(X) 

 

pca = PCA(n_components=2) 

X_pca = pca.fit_transform(X_scaled) 

 

print("Explained Variance Ratio:", pca.explained_variance_ratio_) 

print("Total Variance:", sum(pca.explained_variance_ratio_)) 

 

for species in y.unique(): 

    mask = y == species 

    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], label=species) 

 

plt.xlabel("Principal Component 1") 

plt.ylabel("Principal Component 2") 

plt.legend() 

plt.show() 