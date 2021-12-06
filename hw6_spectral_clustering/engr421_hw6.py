import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial as spa

X = np.genfromtxt("hw06_data_set.csv", delimiter=",", skip_header=True)  # data points extracted from the csv file

N = X.shape[0]  # N samples
K = 5  # K clusters
threshold = 1.25
R = 5

class_means = np.array([[+2.5, +2.5],
                        [-2.5, +2.5],
                        [-2.5, -2.5],
                        [+2.5, -2.5],
                        [0, 0]])  # class means already given
class_deviations = np.array([[[+0.8, -0.6], [-0.6, +0.8]],
                             [[+0.8, +0.6], [+0.6, +0.8]],
                             [[+0.8, -0.6], [-0.6, +0.8]],
                             [[+0.8, +0.6], [+0.6, +0.8]],
                             [[+1.6, 0], [0, +1.6]]])  # class covariances already given
class_sizes = np.array([50, 50, 50, 50, 100])  # sample sizes for each cluster

# data points plotted
plt.figure(figsize=(6, 6))
plt.plot(X[:, 0], X[:, 1], '.', markersize=10, color="black")
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()

# calculate distances between pairs of data points
distances = np.zeros((N, N))
B = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        distances[i][j] = np.sqrt((X[j][0] - X[i][0]) ** 2 + (X[j][1] - X[i][1]) ** 2)  # euclidean distance
        if distances[i][j] > threshold or distances[i][j] == 0:
            B[i][j] = 0  # if the distance is above the threshold, B_ij = 0
        else:
            B[i][j] = 1  # otherwise, B_ij = 1

# plot connectivity matrix
plt.figure(figsize=(6, 6))
plt.plot(X[:, 0], X[:, 1], '.', markersize=10, color="black")
for i in range(N):
    for j in range(N):
        if B[i][j] == 1:
            plt.plot([X[i][0], X[j][0]], [X[i][1], X[j][1]], "k", linewidth=0.5)
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()

# calculate D
D = np.zeros((N, N))
for i in range(N):
    b_count = 0  # to count how many times b_ij returns 1 in a particular row
    for j in range(N):
        if B[i][j] == 1:
            b_count += 1  # if b_ij is 1, increase d_ii
    D[i][i] = b_count

# calculate the symmetric laplacian matrix according to the formula
I = np.eye(N, dtype=int)
L_symmetric = I - np.matmul(np.sqrt(np.linalg.inv(D)), np.matmul(B, np.sqrt(np.linalg.inv(D))))

L_eigenvalues, L_eigenvectors = np.linalg.eig(L_symmetric)  # extract eigenvalues and eigenvectors
eigenvalues_sorted = L_eigenvalues.argsort()
smallest_eigenvectors = L_eigenvectors[:, eigenvalues_sorted]
smallest_eigenvectors = smallest_eigenvectors[:R]  # take first R values
Z = np.transpose(smallest_eigenvectors)  # transpose and generate a new NxR matrix
initial_centroids = np.vstack([Z[84], Z[128], Z[166], Z[186], Z[269]])

"""# plot X and Z together
plt.figure(figsize=(6, 6))
plt.plot(Z[:, 0], Z[:, 1], '.', markersize=10, color="black")
plt.plot(X[:, 0], X[:, 1], '.', markersize=5, color="red")
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()"""


def update_centroids(memberships, data):
    if memberships is None:
        centroids = initial_centroids
    else:
        centroids = np.vstack([np.mean(data[memberships == k,], axis=0) for k in range(K)])  # updated centroids
    return centroids


def update_memberships(centroids, data):
    D = spa.distance_matrix(centroids, data)  # distances between centroids and data points
    memberships = np.argmin(D, axis=0)  # smallest distance finds the nearest centroid
    return memberships


def plot_current_state(centroids, memberships, data):
    plt.figure(figsize=(6, 6))
    cluster_colors = np.array(["#377eb8", "#984ea3", "#4daf4a", "#e41a1c", "#ff7f00"])
    for c in range(K):
        plt.plot(data[memberships == c, 0], data[memberships == c, 1], ".", markersize=10,
                 color=cluster_colors[c])
        plt.plot(centroids[c, 0], centroids[c, 1], "s", markersize=12,
                 markerfacecolor=cluster_colors[c], markeredgecolor="black")
    plt.xlabel("x1")
    plt.ylabel("x2")


# k-means clustering algorithm
centroids = None
memberships = None
iteration = 1
while True:
    print("Iteration#{}:".format(iteration))

    old_centroids = centroids
    centroids = update_centroids(memberships, Z)  # updating the centroids at each iteration
    # k-means algorithm stops when centroids stop changing
    if np.all(centroids == old_centroids):
        break

    old_memberships = memberships
    memberships = update_memberships(centroids, Z)  # updating the memberships at each iteration
    # k-means algorithm stops when memberships stop changing
    if np.all(memberships == old_memberships):
        plot_current_state(centroids, memberships, X)
        plt.show()
        break

    iteration = iteration + 1
