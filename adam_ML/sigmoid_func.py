import math
import matplotlib.pyplot as plt
X = []
y = []

starting_value = -10

for i in range(20):
    X.append(starting_value)
    starting_value += 1

print(X)

for i in range(len(X)):
    y_value = 1/(1+ math.e ** -X[i])
    y.append(y_value)

plt.plot(X, y)
plt.show()