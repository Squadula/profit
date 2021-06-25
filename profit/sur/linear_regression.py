r"""This module contains the backend for Linear Regression models

Work in progress
"""

from abc import ABC, abstractmethod
from .sur import Surrogate

class LinearRegression(Surrogate, ABC):
    """Base class for all linear Linear Regression models.

    Attributes:

    Parameters:

    """

    _defaults = {}  # standard linear model or polynomial basis model

    def __init__(self):
        super().__init__()
        self.transformer = None

    def prepare_train(self, X, y, others):
        """

        """
        pass

    def set_transformation(self, params):
        """"

        """
        for key, value in params.items():
            if key == 'polynomial':
                from sklearn.preprocessing import PolynomialFeatures

                self.transformer = PolynomialFeatures(value)


@Surrogate.register('SklearnLinReg')
class SklearnLinReg(LinearRegression):
    """

    """

    def __init__(self):
        super().__init__()
        self.model = None

    def train(self, X, y, fixed_sigma_n=False, multi_output=False):
        from sklearn.linear_model import LinearRegression as LinReg

        self.Xtrain = X
        self.ytrain = y
        Xtransformed = self.transformer.fit_transform(X)

        self.model = LinReg()
        self.model.fit(Xtransformed, y)

        self.trained = True
        self.decode_training_data()

    def predict(self, Xpred, add_data_variance=True):
        XpredTransformed = self.transformer.fit_transform(Xpred)

        return self.model.predict(XpredTransformed)

    def save_model(self, path):
        pass

    @classmethod
    def load_model(cls, path):
        pass

    @classmethod
    def from_config(cls, config, base_config):
        pass

    @classmethod
    def handle_subconfig(cls, config, base_config):
        pass

