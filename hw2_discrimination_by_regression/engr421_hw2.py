"""ENGR 421 - Homework 2
    Discrimination by Regression
    Zeynep Öner
    ID: 64912"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def safelog(x):
    return np.log(x + 1e-100)
# safelog needed for the objective function


def sigmoid(X, w, w0):
    s = 1 / (1 + np.exp(-(np.matmul(X, w) + w0)))
    return s
# sigmoid function used for calculating y predicted


def calculate_confusion_matrix(predicted, true, true_string):
    confusion_matrix = pd.crosstab(predicted, true, rownames=['y_predicted'], colnames=[true_string])
    return confusion_matrix


# gradient functions are calculated as explained in section 10.8 of the textbook
def gradient_W(X, y_truth, y_pred):
    g = [-np.sum(np.repeat((y_truth[:, c] - y_pred[:, c])[:, None], X.shape[1], axis=1)
                 * np.repeat(y_pred[:, c][:, None], X.shape[1], axis=1)
                 * np.repeat((1 - y_pred[:, c])[:, None], X.shape[1], axis=1) * X, axis=0) for c in range(K)]
    return np.array(g).transpose()


def gradient_w0(y_truth, y_pred):
    g = -np.sum((y_truth - y_pred) * y_truth * (1 - y_pred))
    return g


# importing images, labels and initial weights
images = np.genfromtxt("hw02_images.csv", delimiter=",")
labels = np.genfromtxt("hw02_labels.csv", dtype=int)
W = np.genfromtxt("initial_W.csv", delimiter=",")
w0 = np.genfromtxt("initial_w0.csv")

K = max(labels)  # K classes

# separation of images and labels into the training and test sets
images_separated = np.split(images, 2)
images_training = images_separated[0]
images_test = images_separated[1]

labels_separated = np.split(labels, 2)
labels_training = labels_separated[0]
labels_test = labels_separated[1]

N = images_training.shape[0]  # N samples in each set
D = images_training.shape[1]  # D features

eta = 0.0001  # step size
epsilon = 1e-3  # threshold
max_iteration = 500

# one-hot encoding for the label set
Y_truth = np.zeros((N, K)).astype(int)
for n in range(N):
    for c in range(K):
        if labels_training[n] == c + 1:
            Y_truth[n][c] = 1
        # if the n'th value of the label set is (c + 1), the c'th element in the n'th row is 1

iteration = 0
objective_function = []
training_predicted = float()

while iteration < max_iteration:
    iteration += 1

    training_predicted = sigmoid(images_training, W, w0)  # prediction for the training set
    objective_function.append(-np.sum(Y_truth * safelog(training_predicted)))

    # temp variables to store the old weight values
    W_temp = W
    w0_temp = w0

    # update equations
    W = W - eta * gradient_W(images_training, Y_truth, training_predicted)
    w0 = w0 - eta * gradient_w0(Y_truth, training_predicted)

    # if threshold is acquired, break
    if np.sqrt(np.sum(W - W_temp)**2 + np.sum(w0 - w0_temp)**2) < epsilon:
        break

# with the optimized W and w0 values, calculate y_predicted for the test set
test_predicted = sigmoid(images_test, W, w0)

# objective function plotted
plt.figure(figsize=(10, 6))
plt.plot(range(iteration), objective_function, "k-")
plt.xlabel("Iteration")
plt.ylabel("Error")
plt.show()

# confusion matrix calculation for the training set
y_pred_train = np.argmax(training_predicted, axis=1) + 1
confusion_mat_train = calculate_confusion_matrix(y_pred_train, labels_training, 'y_train')
print("\nConfusion matrix for the training set:")
print(confusion_mat_train)

# confusion matrix calculation for the test set
y_pred_test = np.argmax(test_predicted, axis=1) + 1
confusion_mat_test = calculate_confusion_matrix(y_pred_test, labels_test, 'y_test')
print("\nConfusion matrix for the test set:")
print(confusion_mat_test)
