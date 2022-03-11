import numpy as np
import matplotlib.pyplot as plt

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

# ==================================== #
# ============= chaospy 1D ============== #
# ==================================== #

# from mpl_toolkits.mplot3d import Axes3D
# from scipy.special import binom
# import chaospy
#
def func(x: np.ndarray = None) -> np.ndarray:
    """This is a docstring

    Here are some extra lines for additional explanation on the
    function, its arguments and results

    Args:
        x: a float or array of floats

    Returns:
        Output of the function
    """
    return x + np.cos(10*x)
#
# np.random.seed(100)
# n, m = 17, 7
# sigma_n = 0.2
# distribution = chaospy.Uniform(0, 1)
# x_sample = distribution.sample(n, rule="halton")
# orthogonal_expansion = chaospy.generate_expansion(m, distribution)
# print(orthogonal_expansion.round(2))
# y_sample = func(x_sample) + np.random.normal(0, sigma_n, n)
# approx_model = chaospy.fit_regression(orthogonal_expansion, x_sample, y_sample)
#
# P = orthogonal_expansion(x_sample)
# A_inv = sigma_n**2 * np.linalg.inv(P @ P.T)
# coeff = 1/sigma_n**2 * A_inv @ P @ y_sample
# x = np.linspace(0, 1, 300)
# y = func(x)
# y_fit = approx_model(x)
# # y_fit2 = orthogonal_expansion(x).T @ coeff
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
# plt.savefig("./figures/chaospy")

# ==================================== #
# ============= chaospy 1D ============== #
# ==================================== #

from mpl_toolkits.mplot3d import Axes3D
from scipy.special import binom

def predict(phi_sample, phi, sigma_n, sigma_p, y_sample):
    A_inv = np.linalg.inv(1/sigma_n**2 * phi_sample @ phi_sample.T + np.diag(np.ones(phi_sample.shape[0]) / sigma_p**2))
    coeff = 1/sigma_n**2 * A_inv @ phi_sample @ y_sample
    y_fit = coeff @ phi.T
    cov = sigma_n**2 + phi @ A_inv @ phi.T
    y_std = np.sqrt(np.diag(cov))
    return y_fit, y_std

def monomials(x, m):
    return np.column_stack([x**(m-1-i) for i in range(m)])

def gaussians(x, mu, s):
    return np.array([[np.exp(-(xi - mui)**2 / (2 * s**2)) for mui in mu] for xi in x])

def sigmoids(x, mu, s):
    return np.array([[1 / (1 + np.exp(-(xi - mui) / s)) for mui in mu] for xi in x])

def LinReg1D():
    import chaospy

    # np.random.seed(147)
    n, m = 17, 9  # number of sample points, order of polynomial
    sigma_n = 0.2  # noise
    sigma_p = 1e2  # prior
    x = np.linspace(0, 1, 300)
    y = func(x)

    # create sample data
    distribution = chaospy.Uniform(0, 1)
    x_sample = distribution.sample(n, rule='halton')
    y_sample = func(x_sample) + np.random.normal(0, sigma_n, n)

    # orthogonal_expansion = chaospy.generate_expansion(m, distribution)
    # approx_model = chaospy.fit_regression(orthogonal_expansion, x_sample, y_sample)
    # P = orthogonal_expansion(x_sample)
    # y_fit = approx_model(x)
    # y_fit = orthogonal_expansion(x).T @ coeff
    # cov = sigma_n**2 + orthogonal_expansion(x).T @ A_inv @ orthogonal_expansion(x)

    # monomials
    phi_sample = monomials(x_sample, m).T
    phi = monomials(x, m)
    y_fit, y_std = predict(phi_sample, phi, sigma_n, sigma_p, y_sample)

    # gaussians
    mu = np.linspace(0, 1, int(n / 2))
    s = 0.2
    phi_sample = gaussians(x_sample, mu, s).T
    phi = gaussians(x, mu, s)
    y_fit, y_std = predict(phi_sample, phi, sigma_n, sigma_p, y_sample)

    # sigmoids
    # mu = np.linspace(0, 1, int(n/2))
    # s = 0.2
    # phi_sample = sigmoids(x_sample, mu, s).T
    # phi = sigmoids(x, mu, s)
    # y_fit, y_std = predict(phi_sample, phi, sigma_n, sigma_p, y_sample)

    # plot
    fig, ax = plt.subplots()
    ax.plot(x, y, 'k')
    ax.plot(x_sample, y_sample, 'xk')
    ax.plot(x, y_fit, 'r')
    ax.fill_between(x, y_fit - 2 * y_std, y_fit + 2 * y_std, color='r', alpha=0.4)
    ax.set_ylim([y.min() - 0.3, y.max() + 0.3])
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    plt.savefig("./figures/chaospy")

# LinReg1D()

# ==================================== #
# ============= chaospy 2D ============== #
# ==================================== #

def predict2D(phi_sample, phi, sigma_n, sigma_p, y_sample):
    # see Gaussian Process for Machine Learning
    A_inv = np.linalg.inv(1/sigma_n**2 * phi_sample @ phi_sample.T + np.diag(np.ones(phi_sample.shape[0]) / sigma_p**2))
    y_fit = 1/sigma_n**2 * phi.T @ A_inv @ phi_sample @ y_sample
    # y_fit = coeff.T @ phi
    cov = sigma_n**2 + phi.T @ A_inv @ phi
    y_std = np.sqrt(np.diag(cov))
    return y_fit, y_std

def gaussian2D(x, mu, s):
    return np.array([[np.exp(-np.sqrt(np.sum((xi - mui)**2)) / (2 * s ** 2)) for mui in mu] for xi in x])

def chaospy2D():
    import chaospy

    # parameters
    m = 4           # polynomial order
    sigma_n = 0.02  # noise
    sigma_p = 0.5   # prior

    # generate prediction points
    min_val = xtrain.min(axis=0)
    max_val = xtrain.max(axis=0)
    npoints = [50] * len(min_val)
    xpredict = np.array([np.linspace(minv, maxv, n) for minv, maxv, n in zip(min_val, max_val, npoints)])
    Xpredict = np.hstack([xx.flatten().reshape([-1, 1]) for xx in np.meshgrid(*xpredict)])

    # === Monomials === #
    # phi = chaospy.monomial(start=0, stop=m, dimensions=2)
    # title = f'Monomial expansion \n ' \
    #         f'order $m = 4, \\sigma_{{noise}} = {sigma_n}, \\sigma_{{prior}} = {sigma_p}$'
    # file_location = './figures/Chaospy2DMonomial'
    # old
    # approx_model = chaospy.fit_regression(phi, xtrain.T, ytrain)
    # Ypredict = approx_model(Xpredict[:, 0], Xpredict[:, 1])

    # === Lagrange polynomials === #
    # phi = chaospy.lagrange_polynomial(xtrain.T)
    # title = f'Lagrange polynomials expansion \n ' \
    #         f'$\\sigma_{{noise}} = {sigma_n}, \\sigma_{{prior}} = {sigma_p}$'
    # file_location = './figures/Chaospy2DLagrange'

    # === Orthogonal Stieltjes === #
    # distribution = chaospy.J(chaospy.Uniform(0, 1), chaospy.Uniform(0, 1))
    # phi = chaospy.generate_expansion(m, distribution)
    # title = f'Orthogonal expansion using Stieltjes \n ' \
    #         f'order $m = 4, \\sigma_{{noise}} = {sigma_n}, \\sigma_{{prior}} = {sigma_p}$'
    # file_location = './figures/Chaospy2DOrthogonal'

    # === Chebishev 1 === #
    # phi1D = chaospy.expansion.chebyshev_1(m, lower=0, upper=1)
    # phi1D2 = chaospy.expansion.chebyshev_1(m, lower=0, upper=1)
    # phi1D2.names = ('q1',)
    # phi = chaospy.outer(phi1D, phi1D2).reshape(-1)  # broadcast_arrays?
    # title = f'Chebishev_1 polynomial expansion \n ' \
    #         f'order $m = 4, \\sigma_{{noise}} = {sigma_n}, \\sigma_{{prior}} = {sigma_p}$'
    # file_location = './figures/Chaospy2DChebishev_1'

    # === Chebishev 2 === #
    # phi1D = chaospy.expansion.chebyshev_2(m, lower=0, upper=1)
    # phi1D2 = chaospy.expansion.chebyshev_2(m, lower=0, upper=1)
    # phi1D2.names = ('q1',)
    # phi = chaospy.outer(phi1D, phi1D2).reshape(-1)  # broadcast_arrays?
    # title = f'Chebishev_2 polynomial expansion \n ' \
    #         f'order $m = 4, \\sigma_{{noise}} = {sigma_n}, \\sigma_{{prior}} = {sigma_p}$'
    # file_location = './figures/Chaospy2DChebishev_2'

    # === Hermite === #
    # phi1D = chaospy.expansion.hermite(m, mu=0.5, sigma=0.5)
    # phi1D2 = chaospy.expansion.hermite(m, mu=0.5, sigma=0.5)
    # phi1D2.names = ('q1',)
    # phi = chaospy.outer(phi1D, phi1D2).reshape(-1)
    # title = f'Hermite polynomial expansion \n ' \
    #         f'order $m = 4, \\sigma_{{noise}} = {sigma_n}, \\sigma_{{prior}} = {sigma_p}$'
    # file_location = './figures/Chaospy2DHermite'

    # === Laguerre === #
    # phi1D = chaospy.expansion.laguerre(m)
    # phi1D2 = chaospy.expansion.laguerre(m)
    # phi1D2.names = ('q1',)
    # phi = chaospy.outer(phi1D, phi1D2).reshape(-1)
    # title = f'Laguerre polynomial expansion \n ' \
    #         f'order $m = 4, \\sigma_{{noise}} = {sigma_n}, \\sigma_{{prior}} = {sigma_p}$'
    # file_location = './figures/Chaospy2DLaguerre'

    # === Legendre === #
    phi1D = chaospy.expansion.legendre(m, lower=0.0, upper=1.0)
    phi1D2 = chaospy.expansion.legendre(m, lower=0.0, upper=1.0)
    phi1D2.names = ('q1',)
    phi = chaospy.outer(phi1D, phi1D2).reshape(-1)
    title = f'Legendre polynomial expansion \n ' \
            f'order $m = 4, \\sigma_{{noise}} = {sigma_n}, \\sigma_{{prior}} = {sigma_p}$'
    file_location = './figures/Chaospy2DLegendre'

    # === gaussian === #
    # s = 0.4
    # phitrain = gaussian2D(xtrain, xtrain, s)
    # phipredict = gaussian2D(Xpredict, xtrain, s)
    # Ypredict, Ystd = predict2D(phitrain.T, phipredict.T, sigma_n, sigma_p, ytrain)
    # title = f'Gaussian functions \n ' \
    #         f'$s = {s}, \\sigma_{{noise}} = {sigma_n}, \\sigma_{{prior}} = {sigma_p}$'
    # file_location = './figures/Chaospy2DGaussian'

    # train and predict
    phitrain = phi(xtrain[:, 0], xtrain[:, 1])
    phipredict = phi(Xpredict[:, 0], Xpredict[:, 1])
    Ypredict, Ystd = predict2D(phitrain, phipredict, sigma_n, sigma_p, ytrain)

    # plot
    ax = plt.axes(projection='3d')
    ax.scatter(xtrain[:, 0], xtrain[:, 1], ytrain, color='black', marker='x', alpha=0.8)
    ax.plot_trisurf(Xpredict[:, 0], Xpredict[:, 1], Ypredict, color='red', alpha=0.8)
    ax.plot_trisurf(Xpredict[:, 0], Xpredict[:, 1], Ypredict + 2 * Ystd, color='grey', alpha=0.6)
    ax.plot_trisurf(Xpredict[:, 0], Xpredict[:, 1], Ypredict - 2 * Ystd, color='grey', alpha=0.6)
    ax.set(xlabel='x1', ylabel='x2', zlabel='y(x1, x2)',
           title=title)
    # plt.savefig(file_location)

chaospy2D()
