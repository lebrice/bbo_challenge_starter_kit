from dataclasses import dataclass

import numpy as np
import pytest

from hyperparameters import HyperParameters, log_uniform, uniform
from recycling import Recycler, SimpleRecycler


@dataclass
class A(HyperParameters):
    lr: float = 0.01


@dataclass
class B(A):
    momentum: float = 0.


def test_recycle_known_default_value():
    # Turn off noise just for testing purposes.
    recycler = SimpleRecycler(target_type=B, old_observation_noise_std=0.)
    old_points: List[Tuple[Union[A, B], float]] = [
        (B(lr=0.5, momentum=0.5), 0.5),
        (A(lr=0.1), 0.1),
        (A(lr=0.2), 0.2),
        (A(lr=0.3), 0.3),
    ]
    recycled = recycler.recycle(old_points)
    # recycled_x, recycled_y = zip(*recycled)
    assert recycled == [
        (B(lr=0.5, momentum=0.5), 0.5),
        (B(lr=0.1, momentum=0.), 0.1), # default value of 0 was added.
        (B(lr=0.2, momentum=0.), 0.2), # default value of 0 was added.
        (B(lr=0.3, momentum=0.), 0.3), # default value of 0 was added.
    ]


def test_recycle_known_default_value_subclass():
    recycler: SimpleRecycler[A] = SimpleRecycler(target_type=A, old_observation_noise_std=0.)
    old_points: List[Tuple[Union[A, B], float]] = [
        (A(lr=0.1), 0.1),
        (A(lr=0.2), 0.2),
        (A(lr=0.3), 0.3),
        (B(lr=0.5, momentum=0.5), 0.5),
    ]
    recycled = recycler.recycle(old_points)
    assert recycled == [
        (A(lr=0.1), 0.1),
        (A(lr=0.2), 0.2),
        (A(lr=0.3), 0.3),
        (A(lr=0.5), 0.5), # extra attributes are dropped.
    ]


@dataclass
class C(HyperParameters):
    learning_rate: float = 0.01
    momentum: float = 0.5


def test_recycle_not_related():
    """Test the case where the parameters are not related through inheritance,
    and where the recycler therefore has to use a Translator object to figure
    out which names are equivalent, and 'translate' the old hparams into the
    new type. 
    """
    recycler = SimpleRecycler(target_type=C, old_observation_noise_std=0.)
    old_points: List[Tuple[Union[A, B, C], float]] = [
        (B(lr=0.1, momentum=0.),  0.1),
        (B(lr=0.2, momentum=0.),  0.2),
        (B(lr=0.3, momentum=0.),  0.3),
        (B(lr=0.5, momentum=0.5), 0.5),
    ]
    recycled = recycler.recycle(old_points)
    assert recycled == [
        (C(learning_rate=0.1, momentum=0.),  0.1),
        (C(learning_rate=0.2, momentum=0.),  0.2),
        (C(learning_rate=0.3, momentum=0.),  0.3),
        (C(learning_rate=0.5, momentum=0.5), 0.5),
    ]


@pytest.mark.xfail(reason="TODO: Missing keys in the 'no known default' case.")
def test_recycle_not_related_missing_keys():
    """Test the same scenario as above, but where some attributes are missing.

    As a result, the recycler must do some kind of 'nearest neighbour-ish' kind
    of search to figure out what would be the closest value
    """
    recycler: Recycler[C] = SimpleRecycler(target_type=C, old_observation_noise_std=0.)
    old_points: List[Tuple[Union[A, B, C], float]] = [
        (C(learning_rate=0.5, momentum=0.5), 0.5),
        (A(lr=0.5), 0.5),  # Missing a 'momentum' key.
    ]
    recycled = recycler.recycle(old_points)
    assert recycled == [
        (C(learning_rate=0.5, momentum=0.5), 0.5),
        # This next example here is constructed by finding the 'closest'
        # neighbour (in terms of attributes and performance) within the given
        # points which has the missing key, and using its value + some noise
        # (which we set to zero just for testing purposes).
        (C(learning_rate=0.1, momentum=0.5),  0.1),
    ]
