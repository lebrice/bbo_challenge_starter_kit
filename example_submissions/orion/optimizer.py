import time

from bayesmark.abstract_optimizer import AbstractOptimizer
from bayesmark.experiment import experiment_main
from bayesmark.np_util import random as np_random

from orion.client import create_experiment
from orion.core.utils.format_trials import dict_to_trial
from orion.core.utils.exceptions import SampleTimeout

import pprint


def bayesmark_to_orion_space(api_config):
    """Help routine to setup hyperopt search space in constructor.

    Take api_config as argument so this can be static.
    """
    # The ordering of iteration prob makes no difference, but just to be
    # safe and consistnent with space.py, I will make sorted.
    param_list = sorted(api_config.keys())

    space = {}
    round_to_values = {}
    for param_name in param_list:
        param_config = api_config[param_name]

        param_type = param_config["type"]

        param_space = param_config.get("space", None)
        param_range = param_config.get("range", None)
        param_values = param_config.get("values", None)

        # Some setup for case that whitelist of values is provided:
        # TODO(Xavier): What is this???
        # values_only_type = param_type in ("cat", "ordinal")
        # if (param_values is not None) and (not values_only_type):
        #     assert param_range is None
        #     param_values = np.unique(param_values)
        #     param_range = (param_values[0], param_values[-1])
        #     round_to_values[param_name] = interp1d(
        #         param_values, param_values, kind="nearest", fill_value="extrapolate"
        #     )

        if param_type == "int":
            low, high = param_range
            if param_space in ("log", "logit"):
                space[param_name] = f'loguniform({low}, {high}, discrete=True)'
            else:
                space[param_name] = f'uniform({low}, {high}, discrete=True)'
        elif param_type == "bool":
            assert param_range is None
            assert param_values is None
            space[param_name] = 'choices([True, False])'
        elif param_type in ("cat", "ordinal"):
            assert param_range is None
            space[param_name] = f'choices({param_values})'
        elif param_type == "real":
            low, high = param_range
            if param_space in ("log", "logit"):
                space[param_name] = f'loguniform({low}, {high})'
            else:
                space[param_name] = f'uniform({low}, {high})'
        else:
            assert False, "type %s not handled in API" % param_type

    return space


class OrionOptimizer(AbstractOptimizer):
    primary_import = 'orion'

    def __init__(self, api_config, random=np_random):
        """
        Parameters
        ----------
        api_config : dict-like of dict-like
            Configuration of the optimization variables. See API description.
        """
        AbstractOptimizer.__init__(self, api_config)
        self.random = random

        space = bayesmark_to_orion_space(api_config)
        self.experiment = create_experiment(
            'bbo',
            space=space,
            debug=True,
            # algorithms={
            #     'BayesianOptimizer': {
            #         'acq_func': 'EI',  # 'gp_hedge'
            #         'alpha': 1e-10,
            #         'n_initial_points': 5,
            #         'n_restarts_optimizer': 0,
            #         'noise': 1e-10,
            #         'normalize_y': True,
            #         'seed': None,
            #         'strategy': None
            #  }},
        )

        # TODO: Remove when done debugging
        pprint.pprint(self.experiment.configuration)

    def suggest(self, n_suggestions=1):
        """Get suggestion from the optimizer.

        Parameters
        ----------
        n_suggestions : int
            Desired number of parallel suggestions in the output

        Returns
        -------
        next_guess : list of dict
            List of `n_suggestions` suggestions to evaluate the objective
            function. Each suggestion is a dictionary where each key
            corresponds to a parameter being optimized.
        """
        start = time.clock()
        try:
            rvals = [self.experiment.suggest().params for _ in range(n_suggestions)]
        except SampleTimeout:
            raise
        except:
            import traceback
            traceback.print_exc()
            import pdb
            pdb.set_trace()
        return rvals

    def observe(self, X, y):
        """Feed an observation back.

        Parameters
        ----------
        X : list of dict-like
            Places where the objective function has already been evaluated.
            Each suggestion is a dictionary where each key corresponds to a
            parameter being optimized.
        y : array-like, shape (n,)
            Corresponding values where objective has been evaluated
        """
        try:
            for params, objective in zip(X, y):
                trial = dict_to_trial(params, self.experiment.space)
                trial.experiment = self.experiment.id
                self.experiment.observe(
                    trial,
                    [{'name': 'objective', 'type': 'objective', 'value': objective}])
        except:
            import traceback
            traceback.print_exc()
            import pdb
            pdb.set_trace()


if __name__ == "__main__":
    start = time.clock()
    experiment_main(OrionOptimizer)
    print('task time', time.clock() - start)
