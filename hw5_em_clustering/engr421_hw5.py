import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial as spa
import scipy.stats as stats


X = np.genfromtxt("hw05_data_set.csv", delimiter=",", skip_header=True)  # data points extracted from the csv file
N = X.shape[0]  # N samples
D = X.shape[1]
K = 5  # K clusters

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


def update_centroids(memberships, X):
    if memberships is None:
        # centroids initialized from given csv file
        centroids = np.genfromtxt("hw05_initial_centroids.csv", delimiter=",")
    else:
        centroids = np.vstack([np.mean(X[memberships == k,], axis=0) for k in range(K)])  # updated centroids
    return centroids


def update_memberships(centroids, X):
    D = spa.distance_matrix(centroids, X)  # distances between centroids and data points
    memberships = np.argmin(D, axis=0)  # smallest distance finds the nearest centroid
    return memberships


def plot_current_state(centroids, memberships, X):
    cluster_colors = np.array(["#377eb8", "#e41a1c", "#4daf4a", "#984ea3", "#ff7f00"])
    if memberships is None:
        plt.plot(X[:, 0], X[:, 1], ".", markersize=10, color="black")
    else:
        for c in range(K):
            plt.plot(X[memberships == c, 0], X[memberships == c, 1], ".", markersize=10,
                     color=cluster_colors[c])
    for c in range(K):
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
    centroids = update_centroids(memberships, X)  # updating the centroids at each iteration
    # k-means algorithm stops when centroids stop changing
    if np.all(centroids == old_centroids):
        break
    else:
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plot_current_state(centroids, memberships, X)

    old_memberships = memberships
    memberships = update_memberships(centroids, X)  # updating the memberships at each iteration
    # k-means algorithm stops when memberships stop changing
    if np.all(memberships == old_memberships):
        plt.show()
        break
    else:
        plt.subplot(1, 2, 2)
        plot_current_state(centroids, memberships, X)
        plt.show()

    iteration = iteration + 1

# one-hot encoding for the memberships set
m = np.zeros((N, K))
for n in range(N):
    for c in range(K):
        if memberships[n] == c + 1:
            m[n][c] = 1
memberships = m

sample_means = np.zeros((K, D))
sample_covariances = []
priors = []
for k in range(K):
    sample_covariances.append(np.eye(D))
    priors.append(class_sizes[k] / K)
    # prior probabilities calculated as number of data points in cluster k/total number of data points
sample_covariances = np.array(sample_covariances)
priors = np.array(priors)
if centroids.all() is not None:
    sample_means = centroids
h = np.zeros((N, K))  # success probability
for i in range(100):  # 100 iterations
    for k in range(K):
        for n in range(N):
            # E-step, requires the calculation of success probability
            comp_dens = stats.multivariate_normal.pdf(X[n], sample_means[k], sample_covariances[k])  # multivariate gaussian
            h[n][k] = priors[k] * comp_dens
            m[n][k] = h[n][k] / np.sum(h[n, :])
        sum_sprob = np.sum(m[:, k])
        # M-step to find sample mean, covariance and priors
        priors[k] = sum_sprob / N
        sample_means[k] = m[:, k].dot(X) / sum_sprob
        cov_num = np.sum(m[n, k] * np.outer(X[n] - sample_means[k], X[n] - sample_means[k]) for n in range(N))
        sample_covariances[k] = cov_num / sum_sprob + np.eye(D) * 10e-4

memberships = update_memberships(centroids, X)  # final update for the memberships

plt.figure(figsize=(6, 6))
plot_current_state(centroids, memberships, X)
x, y = np.mgrid[-6:6:.05, -6:6:.05]  # mesh grid for the coordinates of the contour lines
coordinates = np.empty(x.shape + (2,))
coordinates[:, :, 0] = x
coordinates[:, :, 1] = y
for i in range(K):
    initial_pdf = stats.multivariate_normal(class_means[i], class_deviations[i])
    final_pdf = stats.multivariate_normal(sample_means[i], sample_covariances[i])
    plt.contour(x, y, initial_pdf.pdf(coordinates), linestyles='dashed', levels=[0.05])
    plt.contour(x, y, final_pdf.pdf(coordinates), levels=[0.05])
plt.show()
print(sample_means)