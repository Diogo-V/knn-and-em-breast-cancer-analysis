import numpy as np
from scipy.stats import multivariate_normal, norm

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
joint_p2_x1 = likelihood_p2_x1 * phi_1
joint_p2_x2 = likelihood_p2_x2 * phi_1
joint_p2_x3 = likelihood_p2_x3 * phi_1
joint_p2_x4 = likelihood_p2_x4 * phi_1

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
E1_new = ((posterior_p1_x1 * np.dot(np.subtract(x1, u1), np.transpose(np.subtract(x1, u1)))) + (posterior_p1_x2 * np.dot(np.subtract(x2, u1), np.transpose(np.subtract(x2, u1)))) + (posterior_p1_x3 * np.dot(np.subtract(x3, u1), np.transpose(np.subtract(x3, u1)))) + (posterior_p1_x4 * np.dot(np.subtract(x4, u1), np.transpose(np.subtract(x4, u1))))) / posterior_p1_sum
E2_new = ((posterior_p2_x1 * np.dot(np.subtract(x1, u2), np.transpose(np.subtract(x1, u2)))) + (posterior_p2_x2 * np.dot(np.subtract(x2, u2), np.transpose(np.subtract(x2, u2)))) + (posterior_p2_x3 * np.dot(np.subtract(x3, u2), np.transpose(np.subtract(x3, u2)))) + (posterior_p2_x4 * np.dot(np.subtract(x4, u2), np.transpose(np.subtract(x4, u2))))) / posterior_p2_sum


# New priors
phi_1_new = posterior_p1_sum / 4
phi_2_new = posterior_p2_sum / 4

print('For c = 1, new centroid mean: ', u1, 'new E1: ', E1_new, 'new prior: ', phi_1_new)
print('For c = 2, new centroid mean: ', u2, 'new E1: ', E2_new, 'new prior: ', phi_2_new)
