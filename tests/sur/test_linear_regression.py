import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import BayesianRidge

# training data
xtrain = np.array(
      [[0.5       , 0.33333333],
       [0.25      , 0.66666667],
       [0.75      , 0.11111111],
       [0.125     , 0.44444444],
       [0.625     , 0.77777778],
       [0.375     , 0.22222222],
       [0.875     , 0.55555556],
       [0.0625    , 0.88888889],
       [0.5625    , 0.03703704],
       [0.3125    , 0.37037037],
       [0.8125    , 0.7037037 ],
       [0.1875    , 0.14814815],
       [0.6875    , 0.48148148],
       [0.4375    , 0.81481481],
       [0.9375    , 0.25925926],
       [0.03125   , 0.59259259],
       [0.53125   , 0.92592593],
       [0.28125   , 0.07407407],
       [0.78125   , 0.40740741],
       [0.15625   , 0.74074074]])

ytrain = np.array(
      [-0.10316614,  0.16585917, -0.29220324,  0.15771768,
        0.06451588, -0.13031895, -0.12112112,  0.43093146,
       -0.34623426, -0.01752642, -0.03618134, -0.11282839,
       -0.08835435,  0.15300116, -0.26063376,  0.45675513,
        0.19451957, -0.23697354, -0.14697874,  0.26922018])

# ======================================================================================== #
# ===================================== GPySurrogate ===================================== #
# ======================================================================================== #

# from profit.sur.gaussian_process import GPySurrogate
#
# gp = GPySurrogate()
# gp.train(xtrain, ytrain.reshape([-1, 1]))
#
# gp.plot()
# plt.savefig('GPySurrogate')


# ======================================================================================== #
# ============================ Sklearn linear regression 2D ============================== #
# ======================================================================================== #

from profit.sur.linreg.linear_regression import SklearnLinReg

model = SklearnLinReg()
model.set_transformation({'polynomial': 4})
model.train(xtrain, ytrain, tol=1e-6, fit_intercept=False)

# generate prediction points
min_val = xtrain.min(axis=0)
max_val = xtrain.max(axis=0)
npoints = [50] * len(min_val)
xpredict = np.array([np.linspace(minv, maxv, n) for minv, maxv, n in zip(min_val, max_val, npoints)])
Xpredict = np.hstack([xx.flatten().reshape([-1, 1]) for xx in np.meshgrid(*xpredict)])
Ypredict, Ystd = model.predict(Xpredict)

# plot
ax = plt.axes(projection='3d')
ax.scatter(xtrain[:, 0], xtrain[:, 1], ytrain, color='black', marker='x', alpha=0.8)
ax.plot_trisurf(Xpredict[:, 0], Xpredict[:, 1], Ypredict, color='red', alpha=0.8)
ax.plot_trisurf(Xpredict[:, 0], Xpredict[:, 1], Ypredict+2*Ystd, color='grey', alpha=0.6)
ax.plot_trisurf(Xpredict[:, 0], Xpredict[:, 1], Ypredict-2*Ystd, color='grey', alpha=0.6)
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('y')
# plt.savefig('SklearnLinReg')

# ======================================================================================== #
# ========================= Sklearn linear regression 1D ver2 ============================ #
# ======================================================================================== #

def func(x: np.ndarray = None):
    return x + np.cos(10 * x)

def gaussian(x, mu, s):
    return np.array([[np.exp(-(xi - mui)**2 / (2 * s**2)) for mui in mu] for xi in x])

def sigmoid(x, mu, s):
    return np.array([[1 / (1 + np.exp(-(xi - mui) / s)) for mui in mu] for xi in x])

size = 10
sigma_n = 0.2
# rng = np.random.RandomState(148)
rng = np.random.RandomState(147)
xtrain = np.linspace(0, 1, size)
mutrain = np.linspace(0, 1, int(size/2+1))
ytrain = func(xtrain) + rng.normal(loc=0, scale=sigma_n, size=size)
x = np.linspace(0, 1, 300)
y = func(x)

n_order = 8
Xtrain = np.vander(xtrain, n_order + 1, increasing=True)
X = np.vander(x, n_order + 1, increasing=True)

# s = 0.2
# Xtrain = gaussian(xtrain, mutrain, s)
# X = gaussian(x, mutrain, s)

# s = 0.1
# Xtrain = sigmoid(xtrain, mutrain, s)
# X = sigmoid(x, mutrain, s)

model1D = BayesianRidge(tol=1e-3, fit_intercept=False, compute_score=True, alpha_init=1/sigma_n**2, lambda_init=1e-4)
model1D.fit(Xtrain, ytrain)
ypredict, ystd = model1D.predict(X, return_std=True)
print(f'mean std deviation: {ystd.mean()}')
print(f"log marginal likelihood: {model1D.scores_[-1]}")


# plot
fig, ax = plt.subplots()
ax.plot(x, y, 'k')
ax.plot(xtrain, ytrain, 'xk')
ax.plot(x, ypredict, 'r')
ax.fill_between(x, ypredict-2*ystd, ypredict+2*ystd, color='r', alpha=0.4)
ax.set_ylim([y.min() - 0.6, y.max() + 0.6])
ax.set_title('polynomial basis functions')
ax.set_xlabel('x')
ax.set_ylabel('f(x)')
plt.savefig('SklearnLinReg1D')

# ======================================================================================== #
# ============================ Sklearn linear regression 1D ============================== #
# ======================================================================================== #

# def func(x: np.ndarray = None):
#     return x + np.cos(10 * x)
#
# size = 10
# sigma_n = 0.2
# rng = np.random.RandomState(148)
# xtrain = np.linspace(0, 1, size)
# ytrain = func(xtrain) + rng.normal(loc=0, scale=sigma_n, size=size)
#
# n_order = 8
# Xtrain = np.vander(xtrain, n_order + 1, increasing=True)
#
# n_order = 8
# model1D = SklearnLinReg()
# model1D.set_transformation({'polynomial': n_order})
# model1D.train(xtrain.reshape([-1, 1]), ytrain, n_iter=300, tol=1e-3, fit_intercept=False,
#               alpha_init=1/(sigma_n**2), lambda_init=1e-4)
#
# x = np.linspace(0, 1, 300)
# y = func(x)
# ypredict, ystd = model1D.predict(x.reshape([-1, 1]))
# print(f'mean std deviation: {ystd.mean()}')
#
# # plot
# fig, ax = plt.subplots()
# ax.plot(x, y, 'k')
# ax.plot(xtrain, ytrain, 'xk')
# ax.plot(x, ypredict, 'r')
# ax.fill_between(x, ypredict-2*ystd, ypredict+2*ystd, color='r', alpha=0.4)
# ax.set_ylim([y.min() - 0.3, y.max() + 0.3])
# ax.set_xlabel('x')
# ax.set_ylabel('f(x)')
# plt.savefig('SklearnLinReg1D')

# ======================================================================================== #
# ==================== Linear regression without transformation ========================== #
# ======================================================================================== #

# from sklearn.linear_model import LinearRegression
#
# reg = LinearRegression().fit(xtrain, ytrain)
#
# min_val = xtrain.min(axis=0)
# max_val = xtrain.max(axis=0)
# npoints = [50] * len(min_val)
# xpredict = np.array([np.linspace(minv, maxv, n) for minv, maxv, n in zip(min_val, max_val, npoints)])
# Xpredict = np.hstack([xx.flatten().reshape([-1, 1]) for xx in np.meshgrid(*xpredict)])
# Ypredict = reg.coef_ @ Xpredict.T + reg.intercept_
# Ypredict2 = reg.predict(Xpredict)
#
# ax = plt.axes(projection='3d')
# ax.scatter(xtrain[:, 0], xtrain[:, 1], ytrain, color='black', marker='x', alpha=0.8)
# ax.plot_trisurf(Xpredict[:, 0], Xpredict[:, 1], Ypredict, color='red', alpha=0.8)
# ax.set_xlabel('x1')
# ax.set_ylabel('x2')
# ax.set_zlabel('y')
# plt.savefig('LinReg_without_transformation')
#
#
# # ======================================================================================== #
# # ============ Linear Regression with transformation to polynomial basis ================= #
# # ======================================================================================== #
#
# from sklearn.preprocessing import PolynomialFeatures
# # from sklearn.compose import TransformedTargetRegressor
#
# # transformer = PolynomialFeatures(3)
# # regression = LinearRegression()
# # reg = TransformedTargetRegressor(regressor=regression, transformer=transfomr)
#
# poly = PolynomialFeatures(3)
# xtrans = poly.fit_transform(xtrain)
# Xtrans = poly.fit_transform(Xpredict)
# reg = LinearRegression()
# reg.fit(xtrans, ytrain)
# Ypredict = reg.predict(Xtrans)
#
# # plot
# ax = plt.axes(projection='3d')
# ax.scatter(xtrain[:, 0], xtrain[:, 1], ytrain, color='black', marker='x', alpha=0.8)
# ax.plot_trisurf(Xpredict[:, 0], Xpredict[:, 1], Ypredict, color='red', alpha=0.8)
# ax.set_xlabel('x1')
# ax.set_ylabel('x2')
# ax.set_zlabel('y')
# plt.savefig('LinReg_with_transformation')
#
#
# ======================================================================================== #
# ==== Linear regression with transformation to polynomial basis and Bayesian Ridge ====== #
# ======================================================================================== #

# from sklearn.linear_model import BayesianRidge
# from sklearn.preprocessing import PolynomialFeatures
#
# poly = PolynomialFeatures(4)
# xtrans = poly.fit_transform(xtrain)
# Xtrans = poly.fit_transform(Xpredict)
# reg = BayesianRidge(tol=1e-6, fit_intercept=False)
# # reg.set_params(alpha_init=1., lambda_init=1e1)
# reg.fit(xtrans, ytrain)
# Ypredict, Ystd = reg.predict(Xtrans, return_std=True)
#
# ax = plt.axes(projection='3d')
# ax.scatter(xtrain[:, 0], xtrain[:, 1], ytrain, color='black', marker='x', alpha=0.8)
# ax.plot_trisurf(Xpredict[:, 0], Xpredict[:, 1], Ypredict, color='red', alpha=0.8)
# ax.plot_trisurf(Xpredict[:, 0], Xpredict[:, 1], Ypredict+Ystd, color='grey', alpha=0.6)
# ax.plot_trisurf(Xpredict[:, 0], Xpredict[:, 1], Ypredict-Ystd, color='grey', alpha=0.6)
# ax.set_xlabel('x1')
# ax.set_ylabel('x2')
# ax.set_zlabel('y')
# plt.savefig('BayesianRidge')


# ======================================================================================== #
# ====  ====== #
# ======================================================================================== #

# import numpy as np
# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.pyplot as plt
# from scipy.special import binom
# import chaospy
#
# def func(x: np.ndarray = None) -> np.ndarray:
#     """This is a docstring
#
#     Here are some extra lines for additional explanation on the
#     function, its arguments and results
#
#     Args:
#         x: a float or array of floats
#
#     Returns:
#         Output of the function
#     """
#     return x + np.cos(10*x)
#
# np.random.seed(100)
# n, m = 17, 8
# sigma_n = 0.2
# distribution = chaospy.Uniform(0, 1)
# x_sample = distribution.sample(n, rule="halton")
# orthogonal_expansion = chaospy.generate_expansion(m, distribution)
# print(orthogonal_expansion.round(2))
# # print(orthogonal_expansion)
# y_sample = func(x_sample) + np.random.normal(0, sigma_n, n)
# approx_model = chaospy.fit_regression(orthogonal_expansion, x_sample, y_sample)
#
# P = orthogonal_expansion(x_sample)
# A_inv = sigma_n**2 * np.linalg.inv(P @ P.T)
# coeff = 1/sigma_n**2 * A_inv @ P @ y_sample
# x = np.linspace(0, 1, 300)
# y = func(x)
# y_fit = approx_model(x)
# cov = orthogonal_expansion(x).T @ A_inv @ orthogonal_expansion(x)
# std = np.sqrt(np.diag(cov))
#
# # plot
# fig, ax = plt.subplots()
# ax.plot(x, y, 'k')
# ax.plot(x_sample, y_sample, 'xk')
# ax.plot(x, y_fit, 'r')
# ax.fill_between(x, y_fit-2*std, y_fit+2*std, color='r', alpha=0.4)
# ax.set_ylim([y.min()-0.3, y.max()+0.3])
# ax.set_xlabel('x')
# ax.set_ylabel('f(x)')
# plt.show()
