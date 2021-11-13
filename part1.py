import numpy as np
from scipy.stats import multivariate_normal, norm
from scipy.spatial import distance
import matplotlib.pyplot as plt
from math import pi

# --------------------------------------------------- QUESTION 1 ----------------------------------------------------- #
E1 = [[1, 0], [0, 1]]
E2 = [[2, 0], [0, 2]]

x1 = [2, 4]
x2 = [-1, -4]
x3 = [-1, 2]
x4 = [4, 0]

variables = [x1, x2, x3, x4]


# Prior probabilities
phi_1 = 0.7
phi_2 = 0.3

p1 = multivariate_normal(x1, E1)
p2 = multivariate_normal(x2, E2)


# (1) Expectation-phase

# For cluster = 1
# Likelihood probabilitie
likelihood_p1_x1 = p1.pdf(x1)
likelihood_p1_x2 = p1.pdf(x2)
likelihood_p1_x3 = p1.pdf(x3)
likelihood_p1_x4 = p1.pdf(x4)

# Joint probabilitie for c = 1
joint_p1_x1 = likelihood_p1_x1 * phi_1
joint_p1_x2 = likelihood_p1_x2 * phi_1
joint_p1_x3 = likelihood_p1_x3 * phi_1
joint_p1_x4 = likelihood_p1_x4 * phi_1

# For cluster = 2
# Likelihood probabilitie for c = 2
likelihood_p2_x1 = p2.pdf(x1)
likelihood_p2_x2 = p2.pdf(x2)
likelihood_p2_x3 = p2.pdf(x3)
likelihood_p2_x4 = p2.pdf(x4)

# Joint probabilitie for c = 2
joint_p2_x1 = likelihood_p2_x1 * phi_2
joint_p2_x2 = likelihood_p2_x2 * phi_2
joint_p2_x3 = likelihood_p2_x3 * phi_2
joint_p2_x4 = likelihood_p2_x4 * phi_2

# Posterior probabilities for c = 1
posterior_p1_x1 = joint_p1_x1/(joint_p1_x1 + joint_p2_x1)
posterior_p1_x2 = joint_p1_x2/(joint_p1_x2 + joint_p2_x2)
posterior_p1_x3 = joint_p1_x3/(joint_p1_x3 + joint_p2_x3)
posterior_p1_x4 = joint_p1_x4/(joint_p1_x4 + joint_p2_x4)

# Posterior probabilities for c = 2
posterior_p2_x1 = joint_p2_x1/(joint_p1_x1 + joint_p2_x1)
posterior_p2_x2 = joint_p2_x2/(joint_p1_x2 + joint_p2_x2)
posterior_p2_x3 = joint_p2_x3/(joint_p1_x3 + joint_p2_x3)
posterior_p2_x4 = joint_p2_x4/(joint_p1_x4 + joint_p2_x4)

print('For c = 1, x1 likelihood probabilitie = ', likelihood_p1_x1, 'joint probabilitie = ', joint_p1_x1,
      'posterior = ', posterior_p1_x1)
print('For c = 2, x1 likelihood probabilitie = ', likelihood_p2_x1, 'joint probabilitie = ', joint_p2_x1,
      'posterior = ', posterior_p2_x1)
print('For c = 1, x2 likelihood probabilitie = ', likelihood_p1_x2, 'joint probabilitie = ', joint_p1_x2,
      'posterior = ', posterior_p1_x2)
print('For c = 2, x2 likelihood probabilitie = ', likelihood_p2_x2, 'joint probabilitie = ', joint_p2_x2,
      'posterior = ', posterior_p2_x2)
print('For c = 1, x3 likelihood probabilitie = ', likelihood_p1_x3, 'joint probabilitie = ', joint_p1_x3,
      'posterior = ', posterior_p1_x3)
print('For c = 2, x3 likelihood probabilitie = ', likelihood_p2_x3, 'joint probabilitie = ', joint_p2_x3,
      'posterior = ', posterior_p2_x3)
print('For c = 1, x4 likelihood probabilitie = ', likelihood_p1_x4, 'joint probabilitie = ', joint_p1_x4,
      'posterior = ', posterior_p1_x4)
print('For c = 2, x4 likelihood probabilitie = ', likelihood_p2_x4, 'joint probabilitie = ', joint_p2_x4,
      'posterior = ', posterior_p2_x4)


# (2) Maximization-phase
posterior_p1_sum = posterior_p1_x1 + posterior_p1_x2 + posterior_p1_x3 + posterior_p1_x4
posterior_p2_sum = posterior_p2_x1 + posterior_p2_x2 + posterior_p2_x3 + posterior_p2_x4

# New centroid means
u1 = (posterior_p1_x1 * np.transpose(x1) + posterior_p1_x2 * np.transpose(x2) + posterior_p1_x3 * np.transpose(x3) +
      posterior_p1_x4 * np.transpose(x4)) / posterior_p1_sum

u2 = (posterior_p2_x1 * np.transpose(x1) + posterior_p2_x2 * np.transpose(x2) + posterior_p2_x3 * np.transpose(x3) +
      posterior_p2_x4 * np.transpose(x4)) / posterior_p2_sum

# New covariance matrices
E1_new_00 = (posterior_p1_x1 * np.power(np.subtract(x1[0], u1[0]), 2) + (posterior_p1_x2 * np.power(np.subtract(x2[0], u1[0]), 2)) + (posterior_p1_x3 * np.power(np.subtract(x3[0], u1[0]), 2)) + (posterior_p1_x4 * np.power(np.subtract(x4[0], u1[0]), 2))) / posterior_p1_sum
E1_new_01 = (posterior_p1_x1 * np.dot(np.subtract(x1[0], u1[0]), np.subtract(x1[1], u1[1])) + (posterior_p1_x2 * np.dot(np.subtract(x2[0], u1[0]), np.subtract(x2[1], u1[1]))) + (posterior_p1_x3 * np.dot(np.subtract(x3[0], u1[0]), np.subtract(x3[1], u1[1]))) + (posterior_p1_x4 * np.dot(np.subtract(x4[0], u1[0]), np.subtract(x4[1], u1[1])))) / posterior_p1_sum
E1_new_10 = (posterior_p1_x1 * np.dot(np.subtract(x1[1], u1[1]), np.subtract(x1[0], u1[0])) + (posterior_p1_x2 * np.dot(np.subtract(x2[1], u1[1]), np.subtract(x2[0], u1[0]))) + (posterior_p1_x3 * np.dot(np.subtract(x3[1], u1[1]), np.subtract(x3[0], u1[0]))) + (posterior_p1_x4 * np.dot(np.subtract(x4[1], u1[1]), np.subtract(x4[0], u1[0])))) / posterior_p1_sum
E1_new_11 = (posterior_p1_x1 * np.power(np.subtract(x1[1], u1[1]), 2) + (posterior_p1_x2 * np.power(np.subtract(x2[1], u1[1]), 2)) + (posterior_p1_x3 * np.power(np.subtract(x3[1], u1[1]), 2)) + (posterior_p1_x4 * np.power(np.subtract(x4[1], u1[1]), 2))) / posterior_p1_sum

E1_new = [[E1_new_00, E1_new_01], [E1_new_10, E1_new_11]]


E2_new_00 = (posterior_p2_x1 * np.power(np.subtract(x1[0], u2[0]), 2) + (posterior_p2_x2 * np.power(np.subtract(x2[0], u2[0]), 2)) + (posterior_p2_x3 * np.power(np.subtract(x3[0], u2[0]), 2)) + (posterior_p2_x4 * np.power(np.subtract(x4[0], u2[0]), 2))) / posterior_p2_sum
E2_new_01 = (posterior_p2_x1 * np.dot(np.subtract(x1[0], u2[0]), np.subtract(x1[1], u2[1])) + (posterior_p2_x2 * np.dot(np.subtract(x2[0], u2[0]), np.subtract(x2[1], u2[1]))) + (posterior_p2_x3 * np.dot(np.subtract(x3[0], u2[0]), np.subtract(x3[1], u2[1]))) + (posterior_p2_x4 * np.dot(np.subtract(x4[0], u2[0]), np.subtract(x4[1], u2[1])))) / posterior_p2_sum
E2_new_10 = (posterior_p2_x1 * np.dot(np.subtract(x1[1], u2[1]), np.subtract(x1[0], u2[0])) + (posterior_p2_x2 * np.dot(np.subtract(x2[1], u2[1]), np.subtract(x2[0], u2[0]))) + (posterior_p2_x3 * np.dot(np.subtract(x3[1], u2[1]), np.subtract(x3[0], u2[0]))) + (posterior_p2_x4 * np.dot(np.subtract(x4[1], u2[1]), np.subtract(x4[0], u2[0])))) / posterior_p2_sum
E2_new_11 = (posterior_p2_x1 * np.power(np.subtract(x1[1], u2[1]), 2) + (posterior_p2_x2 * np.power(np.subtract(x2[1], u2[1]), 2)) + (posterior_p2_x3 * np.power(np.subtract(x3[1], u2[1]), 2)) + (posterior_p2_x4 * np.power(np.subtract(x4[1], u2[1]), 2))) / posterior_p2_sum

E2_new = [[E2_new_00, E2_new_01], [E2_new_10, E2_new_11]]

# New priors
phi_1_new = posterior_p1_sum / 4
phi_2_new = posterior_p2_sum / 4

print('For c = 1, new centroid mean: ', u1, 'new E1: ', E1_new, 'new prior: ', phi_1_new)
print('For c = 2, new centroid mean: ', u2, 'new E2: ', E2_new, 'new prior: ', phi_2_new)


# Plot Clustering Solutions
x_coordinates = [1, 2, 3]
y_coordinates = [4, 5, 6]

# Create figure canvas
plt.figure(figsize=(12, 12))

plt.scatter(x=[u1[0], u2[0]], y=[u1[1], u2[1]], c='r', label='One Point')  # use this to plot a single point
plt.scatter(x=[x1[0], x2[0], x3[0], x4[0]], y=[x1[1], x2[1], x3[1], x4[1]], c='g', label='Multiple Points')

t = np.linspace(0, 2*pi, 100)
plt.plot(u1[0]+E1_new_00*np.cos(t), u1[1]+E1_new_11*np.sin(t))
plt.plot(u2[0]+E2_new_00*np.cos(t), u2[1]+E2_new_11*np.sin(t))

plt.show()
# --------------------------------------------------- QUESTION 2 ----------------------------------------------------- #

#From question 1, x1, x3 and x4 are in cluster 1 and x2 in c2

# Average distance of xi to the points of its clusters
a_x1 = (distance.euclidean(x1, x3) + distance.euclidean(x1, x4)) / 2
a_x3 = (distance.euclidean(x3, x1) + distance.euclidean(x3, x4)) / 2
a_x4 = (distance.euclidean(x4, x1) + distance.euclidean(x4, x3)) / 2
a_x2 = 0

# Min(average distance of xi to points in another cluster)
b_x1 = distance.euclidean(x1, x2)
b_x3 = distance.euclidean(x3, x2)
b_x4 = distance.euclidean(x4, x2)
b_x2 = min(distance.euclidean(x2, x1),distance.euclidean(x2, x3), distance.euclidean(x2, x4)) / 3

silhouette_x1 = (b_x1 - a_x1) / max(b_x1, a_x1)
silhouette_x3 = (b_x3 - a_x3) / max(b_x3, a_x3)
silhouette_x4 = (b_x4 - a_x4) / max(b_x4, a_x4)

silhouette_c1 = (silhouette_x1 + silhouette_x3 + silhouette_x4) / 3

silhouette_C = silhouette_c1 /2

print("a_x1: ", a_x1, "\nb_x1:", b_x1, "\nsilhouette x1:", silhouette_x1, '\n')
print("a_x3: ", a_x3, "\nb_x3:", b_x3, "\nsilhouette x3:", silhouette_x3, '\n')
print("a_x4: ", a_x4, "\nb_x4:", b_x4, "\nsilhouette x4:", silhouette_x4, '\n')
print("silhouette c1: ", silhouette_c1, "\n")
print("silhouette C: ", silhouette_C, "\n")