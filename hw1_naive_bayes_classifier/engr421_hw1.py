"""ENGR 421 - Homework 1
    Naive Bayes' Classifier
    Zeynep Öner
    ID: 64912"""

import numpy as np
import pandas as pd


def calculate_score_function(x, mu, prior):
    score = []
    for j in range(N):
        score.append([(np.sum(x[j] * np.log(mu[c]) + (1 - x[j]) * np.log(1 - mu[c])))
                      + np.log(prior[c]) for c in range(K)])
        # according to the Naive Bayes' discriminant function
    score = np.array(score)
    return score


def calculate_confusion_matrix(predicted, true, true_string):
    confusion_matrix = pd.crosstab(predicted, true, rownames=['y_hat'], colnames=[true_string])
    return confusion_matrix


# generate image and label data from the .csv files
image_data = np.genfromtxt("hw01_images.csv", delimiter=",")
label_data = np.genfromtxt("hw01_labels.csv", delimiter=",", dtype=int)  # we need the labels to be integers

K = np.max(label_data)  # K classes

# division of the data into training and test
image_separated = np.split(image_data, 2)  # splitting the image data into two
image_training = np.array(image_separated[0])  # first half is the training data
image_test = np.array(image_separated[1])  # second half is the test data

label_separated = np.split(label_data, 2)
label_training = np.array(label_separated[0])
label_test = np.array(label_separated[1])

N = image_training.shape[0]  # number of samples
D = image_training.shape[1]  # number of features

# sample mean vector
mu_list = [[] for i in range(K)]  # empty list with K elements
for c in range(K):
    mu_list[c] = np.mean(image_training[label_training == (c + 1)], axis=0)
    # calculate the sample mean column vector according to x_i, both D-dimensional column vectors
mu_hat = np.array(mu_list)
print("Sample mean vector output:")
print(mu_hat[:, 0])
print(mu_hat[:, 1], "\n")

# sample standard deviation vector
sigma_list = [0 for i in range(K)]
for c in range(K):
    sigma_list[c] = np.sqrt(np.mean((image_training[label_training == (c + 1)] - mu_list[c])**2, axis=0))
    # square root of var(X) = E[(X - E[X])^2]
sigma_hat = np.array(sigma_list)
print("Sample standard deviation output:")
print(sigma_hat[:, 0])
print(sigma_hat[:, 1], "\n")

# prior probability of classes
priors_list = []
for c in range(K):
    priors_list.append(np.mean(label_training == (c + 1)))
prior_probs = np.array(priors_list)
print("Prior probabilities:")
print(prior_probs, "\n")

# score function for prediction
score_training = calculate_score_function(image_training, mu_hat, prior_probs)
score_test = calculate_score_function(image_test, mu_hat, prior_probs)
print("Score function output for the training set generated",
      "\nScore function output for the test set generated")

# confusion matrix calculation for the training set
y_pred_train = np.argmax(score_training, axis=1) + 1
confusion_mat_train = calculate_confusion_matrix(y_pred_train, label_training, 'y_train')
print("\nConfusion matrix for the training set:")
print(confusion_mat_train)

# confusion matrix calculation for the test set
y_pred_test = np.argmax(score_test, axis=1) + 1
confusion_mat_test = calculate_confusion_matrix(y_pred_test, label_test, 'y_test')
print("\nConfusion matrix for the test set:")
print(confusion_mat_test)
