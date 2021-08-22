import numpy as np
from tqdm import tqdm
from abc import ABC

"""
TODO: Write extensive doc.
"""


class ActiveLearning(ABC):
    _models = {}

    def __init__(self, runner, surrogate, variables, ntrain, nwarm=1, batch_size=1,
                 acquisition_function='simple_exploration', convergence_criterion=1e-5, nsearch_points=50, make_plot=False):
        from profit.al.aquisition_functions import AcquisitionFunction

        self.runner = runner
        self.surrogate = surrogate
        self.variables = variables

        self.search_space = {var.name: np.linspace(*var.constraints, nsearch_points)
                             for var in variables.list if var.kind.lower() in 'activelearning'}

        al_keys = [var.name for var in variables.list if var.kind.lower() == 'activelearning']
        Xpred = [np.linspace(*var.constraints, nsearch_points) if var.name in al_keys else np.unique(var.value)
                 for var in variables.input_list]
        self.Xpred = np.hstack([xi.flatten().reshape(-1, 1) for xi in np.meshgrid(*Xpred)])

        self.nwarm = nwarm
        self.ntrain = ntrain
        self.batch_size = batch_size
        self.convergence_criterion = convergence_criterion
        self.make_plot = make_plot

        self.krun = 0

        if issubclass(acquisition_function.__class__, AcquisitionFunction):
            self.acquisition_function = acquisition_function
        elif isinstance(acquisition_function, dict):
            label = acquisition_function['class']
            params = {key: value for key, value in acquisition_function.items() if key != 'class'}
            self.acquisition_function = AcquisitionFunction[label](self.Xpred, self.surrogate, self.variables, **params)
        else:
            self.acquisition_function = AcquisitionFunction[acquisition_function](self.Xpred, self.surrogate,
                                                                                  self.variables)

    def warmup(self):
        """
        To get data for active learning, sample initial points randomly.
        """
        from profit.util.variable_kinds import halton
        params_array = [{} for _ in range(self.nwarm)]
        halton_seq = halton(size=(self.nwarm, self.variables.input.shape[-1]))
        for idx, values in enumerate(self.variables.named_input[:self.nwarm]):
            names = values.dtype.names
            for col, key in enumerate(names):
                if key in self.search_space:
                    minv = self.search_space[key][0]
                    maxv = self.search_space[key][-1]
                    rand = minv + (maxv - minv) * halton_seq[idx, col]
                else:
                    rand = values[key][0]
                params_array[idx][key] = rand

        self.runner.spawn_array(params_array, blocking=True)
        self.update_data()

        self.surrogate.train(self.variables.input[:self.nwarm], self.variables.output[:self.nwarm])
        if self.make_plot:
            self.plot()

    def learn(self):
        """
        Main loop for active learning.
        """
        from time import time

        st = time()
        for krun in tqdm(range(self.nwarm, self.ntrain, self.batch_size)):
            """
            1. find next candidates (and assign input)
            2. update runs
            3. assign output (if mcmc is accepted, else delete input)
            4. optimize surrogate
            """
            self.krun = krun

            candidates = self.find_next_candidates()
            self.update_run(candidates)
            self.surrogate.set_ytrain(self.variables.output[:krun+self.batch_size])
            self.surrogate.optimize()
            if self.make_plot:
                self.plot()

        print("Runtime main loop: {}".format(time() - st))
        if self.make_plot:
            from matplotlib.pyplot import show
            show()

    def find_next_candidates(self):
        """
        This function should be implemented as abstract method, as every acqisition function needs different inputs (variance of the surrogate, etc.)
        1. evaluate acquisition function
        2. find the indices of the first batch_size maximum (or minimum?) values
        3. return these indices
        """

        """
        if loss.max() - loss.min() < 1e-5:
            rand = np.random.randint(0, len(self.Xpred), size=self.batch_size)
            candidates = self.Xpred[rand]
            print("\nNo preference. Next random: {}".format(candidates))
        else:
        """
        candidates = self.acquisition_function.find_next_candidates(self.batch_size)
        print("\nNext candidates: {}".format(candidates))
        return candidates

    def update_run(self, candidates):
        params_array = [{} for _ in range(self.batch_size)]

        for key, values in zip(self.variables.named_input.dtype.names, candidates.T):
            for idx, value in enumerate(values):
                params_array[idx][key] = value
        # Start batch
        self.runner.spawn_array(params_array, blocking=True)
        self.update_data()

    def update_data(self):
        for key in self.runner.input_data.dtype.names:
            self.variables[key].value = self.runner.input_data[key].reshape(-1, 1)
        for key in self.runner.output_data.dtype.names:
            self.variables[key].value = self.runner.output_data[key].reshape(-1, 1)

    def save(self, path):
        self.surrogate.save_model(path)

    def plot(self):
        from matplotlib.pyplot import figure, scatter
        figure()
        self.surrogate.plot(self.Xpred)
        #scatter(self.variables.input[self.krun:self.krun + self.batch_size],
        #        self.variables.output[self.krun:self.krun + self.batch_size],
        #        marker='x', c='r')

    @classmethod
    def from_config(cls, runner, surrogate, variables, config, base_config):
        """Instantiates an ActiveLearning object from the configuration parameters.

        Parameters:
            runner (profit.run.runner.Runner): Runner instance.
            surrogate (profit.sur.sur.Surrogate): Surrogate instance.
            variables (profit.util.variable_kinds.VariableGroup): Variables.
            config (dict): Only the 'active_learning' part of the base_config.
            base_config (dict): The whole configuration parameters.

        Returns:
            profit.al.active_learning.ActiveLearning: AL instance.
        """
        self = cls(runner, surrogate, variables, ntrain=base_config['ntrain'], nwarm=config['nwarm'],
                   batch_size=config['batch_size'], acquisition_function=config['acquisition_function'],
                   convergence_criterion=config['convergence_criterion'], nsearch_points=config['nsearch_points'],
                   make_plot=base_config['ui']['plot'])
        return self

    @classmethod
    def register(cls, label):
        """Decorator to register new active learning classes."""
        def decorator(surrogate):
            if label in cls._models:
                raise KeyError(f'registering duplicate label {label} for {cls.__name__}.')
            cls._models[label] = surrogate
            return surrogate
        return decorator
