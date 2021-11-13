import matplotlib.pyplot as plt


def mlp(n):
    inner_layers = 3 * (n * n + n)
    output_layers = n * 2 + 2
    return inner_layers + output_layers


def tree(n):
    return 3 ** n


def bayes(n):
    priors = 1
    n_params_gauss = sum([i for i in range(1, n + 1)]) + n  # Covariance matrix plus n_vars in mean vector
    return priors + 2 * n_params_gauss


M = [2, 5, 10, 30, 100, 300, 1000]

mlp_res = []
tree_res = []
bayes_res = []


for m in M:
    mlp_res.append(mlp(m))
    # tree_res.append(tree(m))
    bayes_res.append(bayes(m))

print(mlp_res)
# print(tree_res)
print(bayes_res)

# Plots graph
plt.plot(M, mlp_res, marker="o")
# plt.plot(M, tree_res, marker="o")
plt.plot(M, bayes_res, marker="o")
# plt.title("Question 3.b")
plt.title("Question 3.c")
# plt.legend(["MLP", "Tree", "Bayes"])
plt.legend(["MLP", "Bayes"])
plt.ylabel("VC dimension")
plt.xlabel("Data dimensionality")
plt.show()
