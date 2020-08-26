import json
from collections import OrderedDict, defaultdict
from dataclasses import MISSING, dataclass
from functools import singledispatch
from pathlib import Path
from typing import (Any, Callable, Dict, Generic, Iterable, Iterator, List,
                    Optional, Set, Tuple, Type, TypeVar, Union)

import GPy
import GPyOpt
import numpy as np
from GPyOpt.methods import BayesianOptimization, ModularBayesianOptimization
from simple_parsing import Serializable

from hyperparameters import (HP, HyperParameters, log_uniform,
                                        uniform)
from logging_utils import get_logger
from utils import field_dict

from recycling import Recycler, SimpleRecycler

logger = get_logger(__file__)

OtherHP = TypeVar("OtherHP", bound=HyperParameters)

class BayesianHPO(BayesianOptimization, Generic[HP]):
    def __init__(self,  f: Callable[[Union[HP, np.ndarray]], np.ndarray],
                        hp_type: Type[HP],
                        domain: List[Dict]=None,
                        previous_runs: List[Tuple[Union[OtherHP, HP], float]]=None,
                        recycler: Recycler=None,
                        old_observation_noise_std: float=0.1,
                        **kwargs):
        # Save the args and kwargs that were used.
        self.hp_type = hp_type
        domain = domain or self.hp_type.get_bounds_dicts()
        super().__init__(f=f, domain=domain, **kwargs)

        self.recycler: Recycler = SimpleRecycler(
            target_type=self.hp_type,
            old_observation_noise_std=old_observation_noise_std,
        )
        
        self.new_runs: List[Tuple[HP, float]] = []
        self.old_runs: List[Tuple[HyperParameters, float]] = []
        
        if previous_runs:
            for hp, perf in previous_runs:
                if type(hp) is self.hp_type:
                    self.new_runs.append((hp, perf))
                else:
                    self.old_runs.append((hp, perf))
            self.warm_start()

    def warm_start(self) -> None:
        if not self.old_runs:
            # Doesn't make sense to call warm_start when there aren't any old points to recycle.
            return
        logger.debug(f"Warm-starting using {len(self.old_runs)} old and "
                     f"{len(self.new_runs)} new runs.")
        recycled_points = self.recycler.recycle(self.old_runs + self.new_runs)
        xs: List[np.ndarray] = []
        ys: List[float] = []
        for hp, perf in recycled_points:
            xs.append(hp.to_array())
            ys.append(perf)
        self.X = np.vstack(xs)
        self.Y = np.vstack(ys)


    def evaluate_objective(self):
        """
        Evaluates the objective.

        Here we just add some code to update the `self.new_runs` attribute,
        and to call warm_start at each step if there are any old runs we could
        reuse/recycle.
        """
        new_hps: List[HP] = []
        for new_x in self.suggested_sample:
            new_hp: HP = self.hp_type.from_array(new_x)
            new_hps.append(new_hp)
        
        super().evaluate_objective()  # This performs the following:
        # self.Y_new, cost_new = self.objective.evaluate(self.suggested_sample)
        # self.cost.update_cost_model(self.suggested_sample, cost_new)
        # self.Y = np.vstack((self.Y,self.Y_new))

        for new_hp, new_y in zip(new_hps, self.Y_new):
            self.new_runs.append((new_hp, new_y))

        # We call warm_start at each step.
        self.warm_start()

    @property
    def suggested_hparams(self) -> HP:
        return self.hp_type.from_array(self.suggested_sample)

    @property
    def best_hparams(self) -> HP:
        return self.hp_type.from_array(self.x_opt)
