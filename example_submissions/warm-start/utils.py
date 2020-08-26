import dataclasses
import itertools
import random
from collections import OrderedDict
from dataclasses import Field
from typing import Dict, Iterable, List, Optional, Tuple, TypeVar

import numpy as np
import torch
from simple_parsing.helpers.serialization import encode, register_decoding_fn

Dataclass = TypeVar("Dataclass")

register_decoding_fn(np.ndarray, np.asarray)

@encode.register
def encode_ndarray(obj: np.ndarray) -> List:
    return obj.tolist()


def field_dict(dataclass: Dataclass) -> Dict[str, Field]:
    result: Dict[str, Field] = OrderedDict()
    for field in dataclasses.fields(dataclass):
        result[field.name] = field
    return result

def set_seed(seed: int):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

def dict_union(*dicts: Dict, dict_factory=OrderedDict) -> Dict:
    """ Simple dict union until we use python 3.9
    
    >>> from collections import OrderedDict
    >>> a = OrderedDict(a=1, b=2, c=3)
    >>> b = OrderedDict(c=5, d=6, e=7)
    >>> dict_union(a, b)
    OrderedDict([('a', 1), ('b', 2), ('c', 5), ('d', 6), ('e', 7)])
    """
    result: Dict = None  # type: ignore
    for d in dicts:
        if result is None:
            result = type(d)()
        result.update(d)
    assert result is not None
    return result

K = TypeVar("K")
V = TypeVar("V")


def zip_dicts(*dicts: Dict[K, V]) -> Iterable[Tuple[K, Tuple[Optional[V], ...]]]:
    # If any attributes are common to both the Experiment and the State,
    # copy them over to the Experiment.
    keys = set(itertools.chain(*dicts))
    for key in keys:
        yield (key, tuple(d.get(key) for d in dicts))

def dict_intersection(*dicts: Dict[K, V]) -> Iterable[Tuple[K, Tuple[V, ...]]]:
    common_keys = set(dicts[0])
    for d in dicts:
        common_keys.intersection_update(d)
    for key in common_keys:
        yield (key, tuple(d[key] for d in dicts))
