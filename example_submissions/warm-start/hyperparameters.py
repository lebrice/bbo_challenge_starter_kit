import copy
import dataclasses
import inspect
import itertools
import logging
import math
import pickle
import random
from abc import ABC, abstractmethod
from collections import OrderedDict, defaultdict
from contextlib import contextmanager
from dataclasses import InitVar, dataclass, fields
from functools import singledispatch
from pathlib import Path
from typing import (Any, Callable, ClassVar, Dict, List, Optional, Tuple, Type,
                    TypeVar, Union, cast, overload)

import matplotlib.pyplot as plt
import numpy as np
from simple_parsing import field
from simple_parsing.helpers import Serializable, encode
from simple_parsing.helpers.serialization import register_decoding_fn

from logging_utils import get_logger
from priors import LogUniformPrior, NormalPrior, Prior, UniformPrior
from translation import Translator, SimpleTranslator
from utils import dict_intersection, field_dict, zip_dicts

logger = get_logger(__file__)
T = TypeVar("T")
HP = TypeVar("HP", bound="HyperParameters")


class BoundType:
    continuous: str = "continuous"
    discrete: str = "discrete"
    bandit: str = "bandit"


@dataclass
class BoundInfo(Serializable):
    name: str
    type: str = "continuous"
    domain: Tuple[float, float] = (np.NINF, np.Infinity)


@dataclass
class HyperParameters(Serializable, decode_into_subclasses=True):  # type: ignore
    """ Base class for dataclasses of HyperParameters. """
    
    sample_from_priors: ClassVar[bool] = False
    
    translator: ClassVar[SimpleTranslator] = None  #type: ignore
    
    @classmethod
    def get_bounds(cls) -> List[BoundInfo]:
        """Returns the bounds of the search domain for this type of HParam.

        Returns them as a list of dictionaries, in the format expected by
        GPyOpt. 
        """
        bounds: List[BoundInfo] = []
        for f in fields(cls):
            # TODO: handle a hparam which is categorical (i.e. choices)
            min = f.metadata.get("min")
            max = f.metadata.get("max")
            if min is None or max is None:
                continue
            if f.type is float:
                bound = BoundInfo(name=f.name, type=BoundType.continuous, domain=(min, max))
            elif f.type is int:
                bound = BoundInfo(name=f.name, type=BoundType.discrete, domain=(min, max))
            else:
                raise NotImplementedError(f"Unsupported type for field {f.name}: {f.type}")
            bounds.append(bound)
        return bounds
    
    @classmethod
    def get_bounds_dicts(cls) -> List[Dict[str, Any]]:
        return [b.to_dict() for b in cls.get_bounds()]

    @classmethod
    def get_orion_space_dict(cls) -> Dict[str, str]:
        result: Dict[str, str] = {}
        for field in fields(cls):
            prior: Optional[Prior] = field.metadata.get("prior")
            if prior:
                result[field.name] = prior.get_orion_space_string()
        return result

    @classmethod
    def sample(cls: Type[HP]) -> HP:
        kwargs: Dict[str, Any] = {}
        for field in dataclasses.fields(cls):
            prior: Optional[Prior] = field.metadata.get("prior")
            if prior is not None:
                value = prior.sample()
                kwargs[field.name] = value
        return cls(**kwargs)

    @classmethod
    @contextmanager
    def use_priors(cls, value: bool=True):
        temp = cls.sample_from_priors
        cls.sample_from_priors = value
        yield
        cls.sample_from_priors = temp

    def to_array(self) -> np.ndarray:
        values: List[float] = []
        for k, v in self.to_dict(dict_factory=OrderedDict).items():
            try:
                v = float(v)
            except Exception as e:
                logger.warning(f"Ignoring field {k} because we can't make a float out of it.")
            else:
                values.append(v)
        return np.array(values, dtype=float)

    @classmethod
    def from_array(cls: Type[HP], array: np.ndarray) -> HP:
        if len(array.shape) == 2 and array.shape[0] == 1:
            array = array[0]

        keys = list(field_dict(cls))
        # idea: could use to_dict and to_array together to determine how many
        # values to get for each field. For now we assume that each field is one
        # variable.
        # cls.sample().to_dict()
        # assert len(keys) == len(array), "assuming that each field is dim 1 for now."
        assert len(keys) == len(array), "assuming that each field is dim 1 for now."
        d = OrderedDict(zip(keys, array))
        logger.debug(f"Creating an instance of {cls} using args {d}")
        d = OrderedDict(
            (k, v.item()) for k, v in d.items()
        )        
        return cls.from_dict(d)

    def distance_to(self, other: Union["HyperParameters", Dict],
                          weights: Dict[str, float]=None) -> float:
        """Computes a 'distance' to another hyperparameter object or dictionary.

        Args:
            other (Union[): Another HyperParameters object
            weights (Dict[str, float], optional): Optional coefficients used to
            scale the distance with respect to each attribute/dimension. 
            Defaults to None.

        Returns:
            float: the distance as a float.
        """
        if weights is None:
            weights = {}

        x1: Dict[str, float] = self.to_dict()
        # wether self and other are related through inheritance
        related = isinstance(other, type(self)) or isinstance(self, type(other))
        if isinstance(other, HyperParameters) and related:
            # Easiest case: the two are of the same type or related through
            # inheritance.
            x2: Dict[str, float] = other.to_dict()
        else:
            print(f"x2 before: {other}")
            # 'Translate' the other into a dict with the same keys as 'self'
            translator = self.get_translator()
            print(f"translator target type: {translator}")
            x2 = translator.translate(other, drop_rest=True)
            weights = translator.translate(weights, drop_rest=True)
            print(f"x2 after: {x2}")

        distance: float = 0.
        for k, (v1, v2) in dict_intersection(x1, x2):
            distance += weights.get(k, 1) * abs(v1 - v2) 
        return distance

    def get_translator(self) -> Translator[Union[Dict, "HyperParameters"], Dict]:
        cls = type(self)
        if cls.translator is None or cls.translator.target is not cls:
            cls.translator = SimpleTranslator(cls)
        return cls.translator


def field_file_proxy(*args, path: Path, **kwargs):
    """ Adds custom serialization and deserialization functions to save/read
    the value to a file instead of keeping it in the dict.
    """
    encoding_fn = save_to_path(path)
    decoding_fn = load_from_path(path)
    return field(*args, encoding_fn=encoding_fn, decoding_fn=decoding_fn, **kwargs)


@singledispatch
def save(obj: object, path: Path) -> None:
    """ Saves the object `obj` at path `path`.

    Uses pickle at the moment, regardless of the path name or object type.
    TODO: Choose the serialization function depending on the path's extension.
    """
    with open(path, "wb") as f:
        pickle.dump(obj, f)


@save.register
def save_serializable(obj: Serializable, path: Path) -> None:
    obj.save(path)


def save_to_path(path: Union[str,Path]) -> Callable[[plt.Figure], str]:
    """Saves the path to a file containing the value instead of the value itself in the dict.

    Args:
        path (Path): Path to save the figure to.

    Returns:
        Callable[[T], str]: Callable that when given the figure, saves it to the
          path and returns the path as a string.
    """
    path = Path(path)
    def _save_to_path(fig: plt.Figure) -> str:
        save(fig, path)
        return str(path)
    return _save_to_path



def load_from_path(path: Path) -> Callable[[plt.Figure], str]:
    """Saves the path to a file containing the value instead of the value itself in the dict.

    Args:
        path (Path): Path to save the figure to.

    Returns:
        Callable[[T], str]: Callable that when given the figure, saves it to the
          path and returns the path as a string.
    """
    path = Path(path)
    def _load_from_path(fig: plt.Figure) -> str:
        suffix = path.suffix
        logger.debug(f"Loading from path {path} (suffix={suffix})")
        if suffix == "json":
            return Serializable.load_json(path, drop_extra_fields=False)
        elif suffix == "yaml":
            return Serializable.load_yaml(path, drop_extra_fields=False)
        else:
            with open(path, "rb") as fp:
                return pickle.load(fp)

    return _load_from_path
    

def get_distance(d1: Dict,
                 d2: Dict,
                 weights: Dict[str, float]=None,
                 match_equivalent_keys: bool=True) -> float:
    """ TODO: unused for now, but could use something like this later down the
    line when using dictionaries and no dataclasses (perhaps for Orion integration)
    """
    if weights:
        assert set(weights) == set(d1), "weights, if given, should match d1!"
    else:
        weights = defaultdict(lambda: 1)
    
    if set(d1) != set(d2) and match_equivalent_keys:
        translator: Translator[Dict, Dict] = SimpleTranslator(target_keys=d1)
        d2 = translator.translate(d2, drop_rest=True)
    
    distance = 0.
    for k, (v1, v2) in dict_intersection(d1, d2):
        distance += weights[k] * (v1 - v2)
    return distance




@overload
def uniform(min: int, max: int, discrete: bool=True, **kwargs) -> int:
    pass

@overload
def uniform(min: float, max: float, discrete: bool=False, **kwargs) -> float:
    pass

def uniform(min: Union[int, float],
            max: Union[int, float],
            discrete: bool=False,
            default: Union[int, float]=None,
            **kwargs) -> Union[int, float]:
    prior = UniformPrior(min=min, max=max, discrete=discrete)
    if default is None:
        default = (min + max) / 2
    return hparam(
        default=default,
        prior=prior,
        **kwargs
    )


@overload
def log_uniform(min: int, max: int, discrete: bool=True, **kwargs) -> int:
    pass

@overload
def log_uniform(min: float, max: float, discrete: bool=False, **kwargs) -> float:
    pass

def log_uniform(min: Union[int,float],
                max: Union[int,float],
                discrete: bool=False,
                default: Union[int, float]=None,
                **kwargs) -> Union[int, float]:
    prior = LogUniformPrior(min=min, max=max, discrete=discrete)
    if default is None:
        log_min = math.log(min, prior.base)
        log_max = math.log(max, prior.base)
        default = math.pow(prior.base, (log_min + log_max) / 2)
    return hparam(
        default=default,
        prior=prior,
        **kwargs,
    )

loguniform = log_uniform

def hparam(default: T,
          *args,
          prior: Union[Type[Prior[T]], Prior[T]]=None,
          **kwargs) -> T:
    metadata = kwargs.get("metadata", {})
    min: Optional[float] = kwargs.get("min", kwargs.get("min"))
    max: Optional[float] = kwargs.get("max", kwargs.get("max"))

    if prior is None:
        assert min is not None and max is not None
        # if min and max are passed but no Prior object, assume a Uniform prior.
        prior = UniformPrior(min=min, max=max)
        metadata.update({
            "min": min,
            "max": max,
            "prior": prior,
        })

    elif isinstance(prior, type) and issubclass(prior, (UniformPrior, LogUniformPrior)):
        # use the prior as a constructor.
        assert min is not None and max is not None
        prior = prior(min=min, max=max)
    
    elif isinstance(prior, Prior):
        metadata["prior"] = prior
        if isinstance(prior, (UniformPrior, LogUniformPrior)):
            metadata.update(dict(
                min=prior.min,
                max=prior.max,
            ))
        elif isinstance(prior, (NormalPrior)):
            metadata.update(dict(
                mu=prior.mu,
                sigma=prior.sigma,
            ))

    else:
        # TODO: maybe support an arbitrary callable?
        raise RuntimeError(
            "hparam should receive either: \n"
            "- `min` and `max` kwargs, \n"
            "- `min` and `max` kwargs and a type of Prior to use, \n"
            "- a `Prior` instance."
        )

    kwargs["metadata"] = metadata
    return field(
        default=default,
        *args, **kwargs, 
    )

