
import matplotlib.pyplot as plt 

from sklearn.datasets import make_blobs 

from sklearn.model_selection import train_test_split 

from sklearn.mixture import GaussianMixture 

 

X, _ = make_blobs( 

    n_samples=300, centers=3, random_state=42 

) 

 

X_train, X_test = train_test_split( 

    X, test_size=0.2, random_state=42 

) 

 

model = GaussianMixture(n_components=3, random_state=42) 

model.fit(X_train) 

 

labels = model.predict(X_test) 

 

print("Cluster Means:\n", model.means_) 

print("Converged:", model.converged_) 

print("Iterations:", model.n_iter_) 

 

plt.scatter(X_test[:, 0], X_test[:, 1], c=labels) 

plt.scatter(model.means_[:, 0], model.means_[:, 1], 

            marker='X', s=200, color='red') 

plt.xlabel("Feature 1") 

plt.ylabel("Feature 2") 

plt.show() 