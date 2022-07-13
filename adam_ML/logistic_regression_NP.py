import numpy as np

class LogisitcRegressionNP():

    def __init__(self, lr=0.001, n_iters=100000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, x, y):
        # initialize parameters
        n_samples, n_features = x.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient Descent
        for i in range(self.n_iters):
            linear_model = np.dot(x, self.weights) + self.bias
            # Decimal values
            y_predicted = self.sigmoid(linear_model)

            # Derivatives
            dw = (1 / n_samples) * np.dot(x.T, (y_predicted-y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, x, label_value=1, threshold=0.5):
        linear_model = np.dot(x, self.weights) + self.bias
        y_predicted = self.sigmoid(linear_model)
        middle = (np.max(y_predicted) - np.min(y_predicted)) / 2 # change to y_predicted
        y_predicted_class = [label_value if i >= 0.85 else 0 for i in y_predicted]
        print("y_predicted is:")
        print(y_predicted)
        print('y_predicted_class is: ')
        print(y_predicted_class)
        return y_predicted_class

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

####################################################################################################

import sklearn
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

# breast_cancer = datasets.load_breast_cancer()
# X, y = breast_cancer.data, breast_cancer.target

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# def accuracy(y_true, y_pred):
#     accuracy = np.sum(y_true == y_pred) / len(y_true)
#     return accuracy

# regressor = LogisitcRegression(lr=0.001, n_iters=1000)
# # 0s and 1s
# print(y_train)
# regressor.fit(X_train, y_train)
# predictions = regressor.predict(X_test)

# my_accuracy = accuracy(y_test, predictions)
# print("Accuracy is: %s" % (my_accuracy))

from sklearn.datasets import make_classification
X, y = make_classification(n_features=3, n_redundant=0, 
                           n_informative=3, random_state=1, 
                           n_clusters_per_class=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

# regressor = LogisitcRegressionNP(lr=0.001, n_iters=1000)
# # 0s and 1s
# print(y_train)
# regressor.fit(X_train, y_train)
# predictions = regressor.predict(X_test)

# my_accuracy = accuracy(y_test, predictions)
# print("Accuracy is: %s" % (my_accuracy))