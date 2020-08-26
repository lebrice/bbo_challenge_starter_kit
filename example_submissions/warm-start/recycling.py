import dataclasses
import json
from abc import abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import *

import numpy as np

from hyperparameters import (HP, HyperParameters, log_uniform,
                                        uniform)
from logging_utils import get_logger
from utils import field_dict

from translation import Translator, SimpleTranslator

logger = get_logger(__file__)

OldHP = TypeVar("OldHP", bound=HyperParameters)
NewHP = TypeVar("NewHP", bound=HyperParameters)


class Recycler(Generic[NewHP]):
    def __init__(self, target_type: Type[NewHP]):
        self.target_type = target_type
        assert self.target_type is not dict, "can't recycle dicts (yet)."

    @abstractmethod
    def recycle(self, old_points: List[Tuple[HP, float]]) -> List[Tuple[NewHP, float]]:
        """ 'Recycles'/adapt old observations into pseudo-new points. """
        pass


class SimpleRecycler(Recycler):
    def __init__(self, target_type: Type[NewHP],
                       old_observation_noise_std: float=0.1,
                       translator: Translator=None):
        super().__init__(target_type=target_type)
        self.old_noise_std = old_observation_noise_std
        if translator is None:
            translator = SimpleTranslator(
                target=self.target_type,
                equivalent_keys_file=Path("warm_start/equivalent_keys.json"),
            )
        self.translator: Translator = translator

    def recycle(self, observations: List[Tuple[Union[HP, NewHP], float]]) -> List[Tuple[NewHP, float]]:
        """Recycles old points into pseudo-new points of the target type.

        TODO: flatten everything in the dict before converting?
        TODO: Adapt to use dicts/history objects for Orion eventually.
        """
        # List that will hold the 'recycled' points to be returned. 
        recycled_points: List[Tuple[NewHP, float]] = []
        
        # Sort the observations into 'new' and 'old' points.
        # NOTE: Any point which isn't exactly of type `self.target_type` is
        # considered 'old' (even if it is a subclass of `self.target_type`).
        old_points: List[Tuple[HP, float]] = []
        new_points: List[Tuple[NewHP, float]] = []

        for hp, perf in observations:
            if type(hp) is self.target_type:
                new_points.append((hp, perf))
            else:
                old_points.append((hp, perf))

        # No need to recycle these points, as they are of the right type.
        recycled_points.extend(new_points)

        # Group the old observations by the type of hyperparameters used.
        old_observations_per_type: Dict[Type[HP], List[Tuple[HP, float]]] = defaultdict(list)
        for hp, perf in old_points:
            old_observations_per_type[type(hp)].append((hp, perf))
       
        # Iterate over the groups, adapting the "old" observations if needed. 
        for hparam_type, old_observations in old_observations_per_type.items():
            if (issubclass(self.target_type, hparam_type) or 
                  issubclass(hparam_type, self.target_type)):
                """
                Case 1: Known default value.
                
                The 'old' and 'new' types are related through inheritance, i.e.
                there is a known default value for any differing attribute.
                
                This means there are only two possible options:
                1. The new hparam type is derived from the old one; 
                   - Reuse the common attributes, and use the default values for
                     the new attributes.
                   - Add multiplicative gaussian noise to the performances.
                2. The old hparam type is derived from the new one; 
                   - Drop the extra keys, and create an instance of the
                     target_type populated with the values of the old instance.
                   - Add multiplicative gaussian noise to the performances.
                
                In practice, since we use use from_dict from the Serializable
                class (from the simple_parsing package) to create instances of
                the desired HyperParameter class from a dictionary, both of
                these cases are taken care of in the same way.
                """
                for old_x, old_y in old_observations:
                    # NOTE: There should always be a default value we can reuse
                    # here because required arguments need to precede
                    # non-required arguments when creating a dataclass.
                    recycled_x: NewHP = self.target_type.from_dict(old_x.to_dict(), drop_extra_fields=True)
                    recycled_y: float = old_y
                    if self.old_noise_std:
                        recycled_y *= np.random.normal(loc=1, scale=self.old_noise_std)
                    recycled_points.append((recycled_x, recycled_y))
            else:
                """
                Case 2: No Known default value.

                The old and new hyperparameter classes aren't related through
                inheritance, hence there isn't necessarily an easy way to reuse
                a 'default value'.

                1. 'Translate' the attribute names from the 'old' hparam type to
                    the corresponding names in the 'new' type.
                    For instance, translate `learning_rate` --> `lr`, such that
                    we can reuse points from previous runs using that attribute
                    but under a different name.
                2. TODO: Perform some kind of weighted nearest-neighbour search
                    to find the nearest examples within the new points and
                    assign a value to any missing attributes based on the value
                    of the nearest point + some noise (Maybe using some kind of
                    distance function between HyperParameters or dicts.
                """
                # Here we 'translate' the old x's into dicts with any keys
                # matching those in `self.target_type` renamed. 
                old_observation_dicts: List[Tuple[Dict[str, Any], float]] = [
                    (self.translator.translate(old_x, drop_rest=True), old_y)
                    for old_x, old_y in old_observations
                ]
                new_keys = set(field_dict(self.target_type))
                
                for old_x_dict, old_y in old_observation_dicts:
                    old_keys = set(old_x_dict.keys())
                    
                    common_keys = new_keys.intersection(old_keys)
                    missing_keys = new_keys - old_keys
                    extra_keys = old_keys - new_keys
                    assert not extra_keys, "used translate with drop_rest=True but there are still extra keys!"

                    if missing_keys:
                        raise NotImplementedError("TODO: Missing Keys, do some kind of weighted nearest-neighbour among the classes with have that key.")
                        for hp, perf in new_points:
                            distance = ...
                    
                    recycled_x = self.target_type.from_dict(old_x_dict)
                    recycled_y = old_y
                    if self.old_noise_std:
                        recycled_y *= np.random.normal(loc=1, scale=self.old_noise_std)
                    recycled_points.append((recycled_x, recycled_y))

        return recycled_points
