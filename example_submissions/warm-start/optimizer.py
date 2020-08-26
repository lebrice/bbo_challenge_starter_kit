import hashlib
import json
import logging
from dataclasses import Field, InitVar, dataclass, fields, make_dataclass
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Type, Union

import bayesmark.random_search as rs
import numpy as np
from bayesmark import np_util
from bayesmark.abstract_optimizer import AbstractOptimizer
from bayesmark.experiment import experiment_main
from orion.core.worker.trial import Trial
from simple_parsing import Serializable, list_field

from hpo import BayesianHPO
from hyperparameters import HyperParameters, loguniform, uniform

logging.getLogger("bayesmark.experiment").setLevel(logging.WARNING)
# taken from the hyperopt example submission.
dtype_map = {"real": float, "int": int, "bool": bool, "cat": str, "ordinal": str}

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


@dataclass
class Database(Serializable):
    api_configs: List[Dict[str, Dict[str, Any]]] = list_field()

    observations: List[Tuple[HyperParameters, float]] = list_field()

    def restore(self, file: Path) -> "Database":
        # TODO: re-create the classes from the api configs so they get
        # registered as subclasses of Serializable, making it possible to
        # load the object from storage using the Serializable methods.
        with open(file) as f:
            obj = json.load(f)
            api_configs = obj["api_configs"]
            print(f"Number of pre-existing configs: {len(api_configs)}")
            for api_config in api_configs:
                _ = hparams_class_from_api_config(api_config)
        return type(self).load(file)


@dataclass
class RandomOptimizer(AbstractOptimizer, Serializable):
    # Unclear what is best package to list for primary_import here.
    primary_import = "GPyOpt"

    def __init__(self, api_config: Dict[str, Dict], random=np_util.random):
        """Build wrapper class to use random search function in benchmark.

        Settings for `suggest_dict` can be passed using kwargs.

        Parameters
        ----------
        api_config : dict-like of dict-like
            Configuration of the optimization variables. See API description.
        """
        task_hash = compute_identity(**api_config)
        HParams = hparams_class_from_api_config(api_config)

        self.hparam_class: Type[HyperParameters] = HParams
        assert self.hparam_class in HyperParameters.subclasses
        self.save_path: Path = Path("observations.json")

        # This 'database' holds all the <HyperParameters, float> pairs seen so
        # far.
        self.database = Database()

        if self.save_path.exists():
            self.database.restore(self.save_path)

        if api_config not in self.database.api_configs:
            self.database.api_configs.append(api_config)

        super().__init__(api_config)
        self.random = random

        from orion.client import create_experiment
        database: Dict[str, str] = {
            "type": 'EphemeralDB',
            "host": 'database.pkl',
        }
        self.experiment = create_experiment(
            name=f"Task_{task_hash}",
            space=self.hparam_class.get_orion_space_dict(),
            storage={"database": database},
        )
        print(f"Experiment name: {self.experiment.name}")

        self.trials: List[Trial] = []     

    def suggest(self, n_suggestions=1):
        """Get suggestion.

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
        # x_guess = rs.suggest_dict([], [], self.api_config, n_suggestions=n_suggestions, random=self.random)
        print(f"# of observations in 'database': {len(self.database.observations)}")
        self.trials.clear()
        
        suggestions: List[Dict] = []

        # Random search:
        # suggestions = [
        #     self.hparam_class.sample() for _ in range(n_suggestions)
        # ]

        for i in range(n_suggestions):
            trial: Trial = self.experiment.suggest()
            assert trial is not None

            self.trials.append(trial)

            suggestions.append(trial.params)

            hp = self.hparam_class.from_dict(trial.params)
            logger.debug(f"trial: {trial}")
            logger.debug(f"trial params: {trial.params}")
            logger.debug(f"HP: {hp}")

        
        return suggestions

    def observe(self, X: List[Dict], y: Union[List[float], np.ndarray]):
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
        assert len(self.trials) == len(X)
        for trial, x_i, y_i in zip(self.trials, X, y):
            print(f"Observing {x_i}: {y_i}")
            hp: HyperParameters = self.hparam_class.from_dict(x_i)
            result = dict(
                name='performance',
                type='objective',
                value=y_i,
            )
            self.experiment.observe(trial, [result])
            self.database.observations.append((hp, y_i))
        self.database.save(self.save_path)


def hparams_class_from_api_config(api_config: Dict[str, Dict],
                                  cls_name: str="") -> Type[HyperParameters]:
    """Dynamically creates a subclass of HyperParameters from `api_config`.

    Args:
        api_config (Dict[str, Dict]): the api config (given by bayesbench?)
        cls_name (str, optional): Name to give to that class. When not given,
        defaults to get_class_name(api_config). Defaults to "".

    Returns:
        Type[HyperParameters]: A new subclass of HyperParameters.
    """
    fields: List[Tuple[str, Type, Field]] = []
    for field_name, v in api_config.items():
        type = v["type"]
        space = v["space"]
        range = v["range"]
        field = make_field(type=type, space=space, range=range)
        field_type = dtype_map[type]
        fields.append((field_name, field_type, field))
    

    cls_name = cls_name or get_class_name(api_config)
    print(f"class name: {cls_name}")
    # TODO: Check if the class already exists, and if so, reuse it instead of
    # re-creating it.
    for cls in HyperParameters.subclasses:
        if cls.__name__ == cls_name:
            return cls
    cls = make_dataclass(
        cls_name=cls_name,
        fields=fields,
        bases=(HyperParameters,),
    )
    assert cls in HyperParameters.subclasses
    return cls


def make_field(type: str, space: str, range: Tuple[float, float]) -> Field:
    discrete = not (type == "real") 
    min_v, max_v = range
    if space == "log":
        return loguniform(min_v, max_v, discrete=discrete)
    else:
        # TODO: make sure that weirder things like categoricals also work.
        return uniform(min_v, max_v, discrete=discrete)


def get_class_name(api_config: Dict[str, Dict]) -> str:
    # TODO: Determine a unique name for this 'space'.
    cls_name = "HParams_" + compute_identity(**api_config)
    return cls_name


def compute_identity(size: int=16, **sample) -> str:
    """Compute a unique hash out of a dictionary

    Parameters
    ----------
    size: int
        size of the unique hash

    **sample:
        Dictionary to compute the hash from

    """
    sample_hash = hashlib.sha256()

    for k, v in sorted(sample.items()):
        sample_hash.update(k.encode('utf8'))

        if isinstance(v, dict):
            sample_hash.update(compute_identity(size, **v).encode('utf8'))
        else:
            sample_hash.update(str(v).encode('utf8'))

    return sample_hash.hexdigest()[:size]


if __name__ == "__main__":
    experiment_main(RandomOptimizer)
