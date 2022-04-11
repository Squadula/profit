import numpy as np
import matplotlib.pyplot as plt
from profit.sur.linreg import ChaospyLinReg
from profit.sur.linreg import CustomLinReg

# training data
Xtrain = np.array(
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

# generate prediction points
min_val = Xtrain.min(axis=0)
max_val = Xtrain.max(axis=0)
npoints = [50] * len(min_val)
xpred = np.array([np.linspace(minv, maxv, n) for minv, maxv, n in
                     zip(min_val, max_val, npoints)])
Xpred = np.hstack(
    [xx.flatten().reshape([-1, 1]) for xx in np.meshgrid(*xpred)])

def plot(Xtrain, ytrain, Xpred, ymean, ycov, title, model):
    ystd = np.sqrt(np.diag(ycov))
    # plot
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(Xtrain[:, 0], Xtrain[:, 1], ytrain, color='black', marker='x', alpha=0.8)
    ax.plot_trisurf(Xpred[:, 0], Xpred[:, 1], ymean, color='red', alpha=0.8)
    ax.plot_trisurf(Xpred[:, 0], Xpred[:, 1], ymean + 2 * ystd, color='grey', alpha=0.6)
    ax.plot_trisurf(Xpred[:, 0], Xpred[:, 1], ymean - 2 * ystd, color='grey', alpha=0.6)
    ax.set(xlabel='x1', ylabel='x2', zlabel='y(x1, x2)',
           title=title)
    fig.savefig('figures/ChaospyLinReg' + model)

# =============================================================== #
# ======================== ChaospyLinReg ======================== #
# =============================================================== #

sigma_n, sigma_p = 0.05, 5
model_name = 'monomial'
# legendre polynomial kwargs:
# kwargs = {
#     'lower': -1,
#     'upper': 1
# }
kwargs = {
}
model = ChaospyLinReg(model_name, 3, **kwargs)
model.train(Xtrain, ytrain, sigma_n, sigma_p)
ymean, ycov = model.predict(Xpred)

plot(Xtrain, ytrain, Xpred, ymean.flatten(), ycov,
     'Linear Regression using ' + model_name + ' polynomials', model_name)


# ============================================================== #
# ======================== CustomLinReg ======================== #
# ============================================================== #

def RBF2D(x, mu, s):
    return np.array([[np.exp(-np.sqrt(np.sum((xi - mui)**2)) / (2 * s ** 2)) for mui in mu] for xi in x]).T

sigma_n, sigma_p = 0.05, 0.5
np.random.seed(1234)
mutest = np.random.uniform(0, 1, [10, 2])

model_name = 'RBF'
kwargs = {
    'mu': mutest,
    's': 0.7
}
model = CustomLinReg(RBF2D, **kwargs)
model.train(Xtrain, ytrain, sigma_n, sigma_p)
ymean, ycov = model.predict(Xpred)

plot(Xtrain, ytrain, Xpred, ymean.flatten(), ycov,
     'Linear Regression using ' + model_name, model_name)
