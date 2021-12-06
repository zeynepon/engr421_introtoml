"""ENGR 421 - Homework 4
    Nonparametric Regression
    Zeynep Öner
    ID: 64912"""

import numpy as np
import matplotlib.pyplot as plt


# 0 log2(0) = 0
def safelog2(x):
    if x == 0:
        return 0
    else:
        return np.log2(x)


def decision_tree(P, x, y):
    # necessary dictionaries for the nodes we will use
    node_indices = {}
    is_terminal = {}
    need_split = {}
    node_splits = {}
    node_frequencies = {}

    # initialization of the root node
    node_indices[1] = np.array(range(N_train))
    is_terminal[1] = False
    need_split[1] = True

    while True:
        split_nodes = [key for key, value in need_split.items() if value is True]
        # if a node needs splitting, we add it to our list
        if len(split_nodes) == 0:  # iteration ends when there are no nodes to split
            break

        for node in split_nodes:  # we go through all the nodes to be split
            data_indices = node_indices[node]  # indices for the data points that reach the node
            need_split[node] = False  # node split now, value is false
            node_frequencies[node] = [np.sum(y[data_indices] == c + 1) for c in range(K)]  # frequency of the node

            if len(np.unique(y[data_indices])) == 1 or len(x[data_indices]) <= P:
                # if the node is pure (one unique label), or the node has P or fewer data points, node is terminal
                is_terminal[node] = True
            else:
                is_terminal[node] = False
                unique_values = np.sort(np.unique(x[data_indices]))  # all the possible data points
                split_positions = (unique_values[1:len(unique_values)] +
                                   unique_values[0:len(unique_values) - 1]) / 2
                # split positions are in the middle of each unique point
                split_scores = np.repeat(0.0, len(split_positions))
                for s in range(len(split_positions)):
                    left_indices = data_indices[x_training[data_indices] < split_positions[s]]
                    right_indices = data_indices[x_training[data_indices] >= split_positions[s]]
                    indices = [left_indices, right_indices]
                    for index in indices:
                        split_scores[s] -= len(index) / len(data_indices) * \
                                           np.sum([np.mean(y_training[index] == c + 1) *
                                                   safelog2(np.mean(y_training[index] == c + 1)) for c in range(K)])
                        # impurity function from lecture notes
                best_split = split_positions[np.argmin(split_scores)]
                # best split is the one with the smallest impurity
                node_splits[node] = best_split  # split corresponding to the node

                left_indices = data_indices[x[data_indices] < best_split]
                node_indices[2 * node] = left_indices
                is_terminal[2 * node] = False
                need_split[2 * node] = True  # left child: 2 * parent

                right_indices = data_indices[x[data_indices] >= best_split]
                node_indices[2 * node + 1] = right_indices
                is_terminal[2 * node + 1] = False
                need_split[2 * node + 1] = True  # right child: 2 * parent + 1
    return is_terminal, node_frequencies, node_splits


def traverse(x, is_terminal, node_frequencies, node_splits):
    # traversing the tree
    index = 1
    while True:
        if is_terminal[index]:
            y = np.argmax(node_frequencies[index]) + 1  # if it's a terminal node, return the frequency
            break
        else:
            if x < node_splits[index]:
                index = index * 2  # go over to the left or the right child according to the x value
            else:
                index = index * 2 + 1
    return y


def rmse(y, y_hat, P):
    r = np.sqrt(np.sum((y - y_hat) ** 2 / N_test))
    print("RMSE is", np.round(r, 4), "when P is", P)
    return r


# importing dataset
dataset = np.genfromtxt("hw04_data_set.csv", delimiter=",", skip_header=True)
# separation of x and y into the training and test sets
training = np.array([dataset[i, :] for i in range(150)])
test = np.array([dataset[i, :] for i in range(150, 272)])

# splitting x and y values
x_training = training[:, 0]
x_test = test[:, 0]
y_training = training[:, 1].astype(int)
y_test = test[:, 1].astype(int)

# necessary parameters
K = np.max(y_training)  # K classes
N_train = x_training.shape[0]  # N samples in the training data
N_test = x_test.shape[0]  # N samples in the test data
# P = int(input("Enter P: "))
P = 25

# learn decision tree with pre-pruning parameter P = 25
is_terminal, node_frequencies, node_splits = decision_tree(P, x_training, y_training)

# plot data and fit
min_value = np.min(x_training)
max_value = np.max(x_training)
data_interval = np.arange(min_value, max_value, 0.01)
y_predicted = [traverse(x, is_terminal, node_frequencies, node_splits) for x in data_interval]

plt.figure(figsize=(10, 6))
plt.plot(x_training, y_training, "b.", markersize=10)
plt.plot(x_test, y_test, "r.", markersize=10)
plt.plot(data_interval, y_predicted, "k-")
plt.xlabel("Eruption time (min)")
plt.ylabel("Waiting time to next eruption (min)")
plt.legend(["training", "test"])
plt.title("P=25")
plt.show()

y_hat_test = [traverse(x, is_terminal, node_frequencies, node_splits) for x in x_test]
rmse(y_test, y_hat_test, P)

# different rmse values for different P's
P_list = np.arange(5, 55, 5)
rmse_list = []
for p in P_list:
    it, nf, ns = decision_tree(p, x_training, y_training)
    y_hat = [traverse(x, it, nf, ns) for x in x_test]
    rmse_list.append(rmse(y_test, y_hat, p))

plt.figure(figsize=(10, 6))
plt.plot(P_list, rmse_list, "ko-")
plt.xlabel("Pre-pruning size (P)")
plt.ylabel("RMSE")
plt.show()
