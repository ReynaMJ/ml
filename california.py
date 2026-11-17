
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error, r2_score

data = fetch_california_housing()
X = data.data[:, [0]]
y = data.target

X = (X - X.mean()) / X.std()
X = np.c_[np.ones(len(X)), X]

# Gradient Descent
theta = np.zeros(2)
lr = 0.01
for i in range(1000):
    pred = X @ theta
    error = pred - y
    theta -= lr * (X.T @ error) / len(y)

# Normal Equation
theta2 = np.linalg.pinv(X.T @ X) @ X.T @ y

p1 = X @ theta
p2 = X @ theta2

print("GD MSE:", mean_squared_error(y, p1))
print("GD R2:", r2_score(y, p1))
print("Normal Equation MSE:", mean_squared_error(y, p2))
print("Normal Equation R2:", r2_score(y, p2))

plt.scatter(X[:, 1], y, s=2)
plt.plot(X[:, 1], p1, color='red')
plt.show()
