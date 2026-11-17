import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_openml

X, y = fetch_openml('Fashion-MNIST', version=1, return_X_y=True, as_frame=False)
Xtr, ytr = X[:10000]/255.0, y[:10000]
Xte, yte = X[60000:62000]/255.0, y[60000:62000]

for k in [1, 3, 5, 7, 9, 15]:
    t = time.time()
    p = KNeighborsClassifier(n_neighbors=k).fit(Xtr, ytr).predict(Xte)
    print(f'K={k:2d}  Accuracy: {accuracy_score(yte, p):.4f}  Time: {time.time() - t:.2f}s')