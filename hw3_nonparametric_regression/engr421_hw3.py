"""ENGR 421 - Homework 3
    Nonparametric Regression
    Zeynep Öner
    ID: 64912"""

import numpy as np
import matplotlib.pyplot as plt
import math

# importing dataset
dataset = np.genfromtxt("hw03_data_set.csv", delimiter=",", skip_header=True)

# separation of x and y into the training and test sets
training = np.array([dataset[i, :] for i in range(150)])
test = np.array([dataset[i, :] for i in range(150, 272)])

# splitting x and y values
x_training = training[:, 0]
x_test = test[:, 0]
y_training = training[:, 1].astype(int)
y_test = test[:, 1].astype(int)
N_test = x_test.shape[0]  # N samples in the test set, used for rmse calculations

# parameters for the smoothers
bin_width = 0.37
origin = 1.5
max_value = max(x_training)
data_interval = np.arange(origin, max_value, 0.01)

# regressogram
left_borders = np.arange(origin, max_value, bin_width)
right_borders = np.arange(origin + bin_width, max_value + bin_width, bin_width)
p_hat = np.array([np.sum(((left_borders[b] < x_training) & (x_training <= right_borders[b])) * y_training)
                  / np.sum((left_borders[b] < x_training) & (x_training <= right_borders[b]))
                  for b in range(len(left_borders))])

plt.figure(figsize=(10, 6))
plt.plot(x_training, y_training, "b.", markersize=10)
plt.plot(x_test, y_test, "r.", markersize=10)
for b in range(len(left_borders)):
    plt.plot([left_borders[b], right_borders[b]], [p_hat[b], p_hat[b]], "k-")
for b in range(len(left_borders) - 1):
    plt.plot([right_borders[b], right_borders[b]], [p_hat[b], p_hat[b + 1]], "k-")
plt.xlabel("Eruption time (min)")
plt.ylabel("Waiting time to next eruption (min)")
plt.legend(["training", "test"])
plt.title("h=0.37")
plt.show()

# rmse calculation for regressogram
y_hat = np.zeros(N_test)
for b in range(len(left_borders)):
    for i in range(N_test):
        if left_borders[b] < x_test[i] <= right_borders[b]:
            y_hat[i] = p_hat[int((x_test[i] - origin) / bin_width)]
            # if x_test[i] is in the bin, y_hat[i] is the corresponding p_hat value
            # similar to our histogram estimator y calculations during class
rmse = np.sqrt(np.sum((y_test - y_hat) ** 2 / N_test))
print("Regressogram => RMSE is", np.round(rmse, 4), "when h is 0.37")

# running mean smoother
p_hat = np.array([np.sum((np.abs((x - x_training) / bin_width) < 1/2) * y_training)
                  / np.sum(np.abs((x - x_training) / bin_width) < 1/2) for x in data_interval])
"""In the textbook, the weight function states |u|<1; but when I do that, the graph is a little off from the one
in the description. So I am using the weight function we defined for the naive estimator in class."""

plt.figure(figsize=(10, 6))
plt.plot(x_training, y_training, "b.", markersize=10)
plt.plot(x_test, y_test, "r.", markersize=10)
plt.plot(data_interval, p_hat, "k-")
plt.xlabel("Eruption time (min)")
plt.ylabel("Waiting time to next eruption (min)")
plt.legend(["training", "test"])
plt.title("h=0.37")
plt.show()

# rmse calculation for running mean smoother
for i in range(N_test):
    y_hat[i] = p_hat[int((x_test[i] - origin) / 0.01)]  # 0.01 is the step size I chose for the data interval
rmse = np.sqrt(np.sum((y_test - y_hat) ** 2 / N_test))
print("Running Mean Smoother => RMSE is", np.round(rmse, 4), "when h is 0.37")

# kernel smoother
p_hat = np.array([np.sum((1 / np.sqrt(2 * math.pi)) * np.exp(-(((x - x_training) / bin_width) ** 2 / 2)) * y_training)
                  / np.sum((1 / np.sqrt(2 * math.pi)) * np.exp(-(((x - x_training) / bin_width) ** 2 / 2)))
                  for x in data_interval])

plt.figure(figsize=(10, 6))
plt.plot(x_training, y_training, "b.", markersize=10)
plt.plot(x_test, y_test, "r.", markersize=10)
plt.plot(data_interval, p_hat, "k-")
plt.xlabel("Eruption time (min)")
plt.ylabel("Waiting time to next eruption (min)")
plt.legend(["training", "test"])
plt.title("h=0.37")
plt.show()

# rmse calculation for the kernel smoother
for i in range(N_test):
    y_hat[i] = p_hat[int((x_test[i] - origin) / 0.01)]
rmse = np.sqrt(np.sum((y_test - y_hat) ** 2 / N_test))
print("Kernel Smoother => RMSE is", np.round(rmse, 4), "when h is 0.37")
