import numpy as np
import pandas as pd
from scipy.io import arff
from scipy.stats import multivariate_normal
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.metrics import silhouette_score

# Data
x1, x2, x3, x4 = [[2, 4], [-1, -4], [-1, 2], [4, 0]]

# Priors
priors = [0.7, 0.3]

# Parâmetros iniciais das gaussianas
mu1, mu2 = x1, x2
mus = [mu1, mu2]
sigma1, sigma2 = [[1, 0], [0, 1]], [[2, 0], [0, 2]]
sigmas = [sigma1, sigma2]

posteriors = []



# i = índice da observação
# k = índice do cluster
# Likelihood (verosimilhança): P(x = x_i | C = k) = f.d.(x_i) de N(mu_k, sigma_k)
# Joint probability: P(C = k) * P(x = x_i | C = k) = P(x = x_i,  C = k)
# Posterior: P(C = k | x = x_i)
# Responsability r_i,k resulta de normalizar os posteriors


# E-Step
def responsabilities(X, priors, mus, sigmas):
    N = len(X)  # número de observações / de pontos no nosso dataset
    K = len(mus)  # Número de clusters / Número de gaussianas no nosso mixture model
    # Matriz com n linhas e K colunas: a entrada i,j é igual a P(C = j | x = x_i)
    posteriors = np.array([[priors[j] * multivariate_normal.pdf(x, mean=mus[j], cov=sigmas[j]) for j in range(0, K)] for x in X])
    # print(posteriors)
    normalized_post = posteriors / posteriors.sum(axis=1)[:, np.newaxis]
    return normalized_post

# 3, Lab 9
print('3 do Lab 9: ', responsabilities([4, 0, 1], [0.5, 0.5], [1, 0], [1, 1]))
print('\n'*4)


# M-Step
def reestimate_param(X, priors, mus, sigmas):
    X = np.array(X)
    mus = np.array(mus)
    N = len(X)
    K = len(mus)
    A = responsabilities(X, priors, mus, sigmas)
    updated_mus = np.transpose(np.array(X).T @ A / A.sum(axis=0))  # Matriz cuja linha j é mu_j
    print('updated mus: ', updated_mus)
    if len(X.shape) > 1:  # número de variáveis > 1
        updated_sigmas = np.array([np.array([(A[n, j] * (X[n] - updated_mus[j])[:, np.newaxis] @ (X[n] - updated_mus[j])[np.newaxis, :]) for n in range(0, N)]).sum(axis=0) for j in range(0, K)]) / A.sum(axis=0)[:, np.newaxis, np.newaxis]
        print('\nupdated sigmas: ', updated_sigmas)
    else:
        # Neste caso, devolvemos o vector com os desvios-padrão
        updated_sigmas = np.sqrt([np.array([(A[n, j] * (X[n] - updated_mus[j])**2) for n in range(0, N)]).sum(axis=0) for j in range(0, K)] / A.sum(axis=0))
        print('\nnovos desvios padrão: ', updated_sigmas)
    updated_priors = A.sum(axis=0) / sum(sum(A))  # sum(sum(A)) é suposto ser a soma de todas as entradas da matriz
    print('\nupdated priors: ', updated_priors)
    return [updated_mus, updated_sigmas, updated_priors]




updated_mus, updated_sigmas, updated_priors = reestimate_param([x1, x2, x3, x4], priors, mus, sigmas)


# 2
updated_responsabilities = responsabilities([x1, x2, x3, x4], updated_priors, updated_mus, updated_sigmas)
print('responsabilities com parâmetros actualizados: ', updated_responsabilities)
clusters1 = [np.argmax(x) for x in updated_responsabilities]  # clusters aproximados
print('\nCoeficiente da silhueta: ', silhouette_score([x1, x2, x3, x4], clusters1, metric='euclidean'))



# 4
dataii, meta = arff.loadarff('./data/breast.w.arff')
df = pd.DataFrame(dataii).dropna()
X = df.iloc[:, :-1]

# Preprocessing the targets
le = preprocessing.LabelEncoder()
y = le.fit_transform(df.iloc[:, -1])

# k = 2
kmeans2 = KMeans(n_clusters=2, random_state=0).fit(X)
labels2 = kmeans2.labels_

# k = 3
kmeans3 = KMeans(n_clusters=3, random_state=0).fit(X)
labels3 = kmeans3.labels_

print('\n\n')

# 4a.
setlabelsy = set(y)   # labels possíveis para y (em princípio, será = [0, 1] porque há 2 classes mas assim fica mais "automático")
setlabels2 = set(labels2)
ecr2 = np.mean([labels2.tolist().count(a) - max([np.array([labels2, y]).T.tolist().count([a, b]) for b in setlabelsy]) for a in setlabels2])
print('ECR para k = 2: ', ecr2)  # 13.5


ecr3 = np.mean([labels3.tolist().count(a) - max([np.array([labels3, y]).T.tolist().count([a, b]) for b in setlabelsy]) for a in set(labels3)])
print('ECR para k = 3: ', ecr3)  # 6.666666666666667




# 4b.
print('\nCoeficiente da silhueta para k = 2: ', silhouette_score(X, labels2, metric='euclidean'))
print('\nCoeficiente da silhueta para k = 3: ', silhouette_score(X, labels3, metric='euclidean'))



# 5
fittedselector = SelectKBest(mutual_info_classif, k=2).fit(X, y)
top2features = fittedselector.get_feature_names_out()
Xnew = fittedselector.transform(X)



newdf = pd.DataFrame(np.c_[Xnew, labels3], columns=np.append(top2features, 'cluster'))
colors = {0: 'red', 1: 'green', 2: 'blue'}
import matplotlib.pyplot as plt
plt.scatter(newdf.iloc[:, 0], newdf.iloc[:, 1], c=newdf['cluster'].map(colors), alpha=0.6, s=20)
plt.scatter(*zip(*[x[1:3] for x in kmeans3.cluster_centers_]), c='pink', s=100)  # Plottar os centros
plt.show()




# i = índice da observação
# k = índice do cluster
# Likelihood (verosimilhança): P(x = x_i | C = k) = f.d.(x_i) de N(mu_k, sigma_k)
# Joint probability: P(C = k) * P(x = x_i | C = k) = P(x = x_i,  C = k)
# Posterior: P(C = k | x = x_i)
# Responsability r_i,k resulta de normalizar os posteriors


# E-Step
def responsabilities(X, priors, mus, sigmas):
    N = len(X)  # número de observações / de pontos no nosso dataset
    K = len(mus)  # Número de clusters / Número de gaussianas no nosso mixture model
    # Matriz com n linhas e K colunas: a entrada i,j é igual a P(C = j | x = x_i)
    posteriors = np.array([[priors[j] * multivariate_normal.pdf(x, mean=mus[j], cov=sigmas[j]) for j in range(0, K)] for x in X])
    # print(posteriors)
    normalized_post = posteriors / posteriors.sum(axis=1)[:, np.newaxis]
    return normalized_post

# 3, Lab 9
print('3 do Lab 9: ', responsabilities([4, 0, 1], [0.5, 0.5], [1, 0], [1, 1]))
print('\n'*4)


# M-Step
def reestimate_param(X, priors, mus, sigmas):
    X = np.array(X)
    mus = np.array(mus)
    N = len(X)
    K = len(mus)
    A = responsabilities(X, priors, mus, sigmas)
    updated_mus = np.transpose(np.array(X).T @ A / A.sum(axis=0))  # Matriz cuja linha j é mu_j
    print('updated mus: ', updated_mus)
    if len(X.shape) > 1:  # número de variáveis > 1
        updated_sigmas = np.array([np.array([(A[n, j] * (X[n] - updated_mus[j])[:, np.newaxis] @ (X[n] - updated_mus[j])[np.newaxis, :]) for n in range(0, N)]).sum(axis=0) for j in range(0, K)]) / A.sum(axis=0)[:, np.newaxis, np.newaxis]
        print('\nupdated sigmas: ', updated_sigmas)
    else:
        # Neste caso, devolvemos o vector com os desvios-padrão
        updated_sigmas = np.sqrt([np.array([(A[n, j] * (X[n] - updated_mus[j])**2) for n in range(0, N)]).sum(axis=0) for j in range(0, K)] / A.sum(axis=0))
        print('\nnovos desvios padrão: ', updated_sigmas)
    updated_priors = A.sum(axis=0) / sum(sum(A))  # sum(sum(A)) é suposto ser a soma de todas as entradas da matriz
    print('\nupdated priors: ', updated_priors)
    return [updated_mus, updated_sigmas, updated_priors]




updated_mus, updated_sigmas, updated_priors = reestimate_param([x1, x2, x3, x4], priors, mus, sigmas)


# 2
updated_responsabilities = responsabilities([x1, x2, x3, x4], updated_priors, updated_mus, updated_sigmas)
print('responsabilities com parâmetros actualizados: ', updated_responsabilities)
clusters1 = [np.argmax(x) for x in updated_responsabilities]  # clusters aproximados
print('\nCoeficiente da silhueta: ', silhouette_score([x1, x2, x3, x4], clusters1, metric='euclidean'))



# 4
dataii, meta = arff.loadarff('breast.w.arff')
df = pd.DataFrame(dataii).dropna()
X = df.iloc[:, :-1]

# Preprocessing the targets
le = preprocessing.LabelEncoder()
y = le.fit_transform(df.iloc[:, -1])

# k = 2
kmeans2 = KMeans(n_clusters=2, random_state=0).fit(X)
labels2 = kmeans2.labels_

# k = 3
kmeans3 = KMeans(n_clusters=3, random_state=0).fit(X)
labels3 = kmeans3.labels_

print('\n\n')

# 4a.
setlabelsy = set(y)   # labels possíveis para y (em princípio, será = [0, 1] porque há 2 classes mas assim fica mais "automático")
setlabels2 = set(labels2)
ecr2 = np.mean([labels2.tolist().count(a) - max([np.array([labels2, y]).T.tolist().count([a, b]) for b in setlabelsy]) for a in setlabels2])
print('ECR para k = 2: ', ecr2)  # 13.5


ecr3 = np.mean([labels3.tolist().count(a) - max([np.array([labels3, y]).T.tolist().count([a, b]) for b in setlabelsy]) for a in set(labels3)])
print('ECR para k = 3: ', ecr3)  # 6.666666666666667




# 4b.
print('\nCoeficiente da silhueta para k = 2: ', silhouette_score(X, labels2, metric='euclidean'))
print('\nCoeficiente da silhueta para k = 3: ', silhouette_score(X, labels3, metric='euclidean'))



# 5
fittedselector = SelectKBest(mutual_info_classif, k=2).fit(X, y)
top2features = fittedselector.get_feature_names_out()
Xnew = fittedselector.transform(X)



newdf = pd.DataFrame(np.c_[Xnew, labels3], columns=np.append(top2features, 'cluster'))
colors = {0: 'red', 1: 'green', 2: 'blue'}
import matplotlib.pyplot as plt
plt.scatter(newdf.iloc[:, 0], newdf.iloc[:, 1], c=newdf['cluster'].map(colors), alpha=0.6, s=20)
plt.scatter(*zip(*[x[1:3] for x in kmeans3.cluster_centers_]), c='pink', s=100)  # Plottar os centros
plt.show()