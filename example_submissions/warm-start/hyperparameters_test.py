from dataclasses import dataclass

import numpy as np

from .hyperparameters import HyperParameters, hparam, log_uniform, uniform


@dataclass
class A(HyperParameters):
    learning_rate: float = uniform(0., 1.)


@dataclass
class B(A):
    momentum: float = uniform(0., 1.)


@dataclass
class C(HyperParameters):
    lr: float = uniform(0., 1.)
    momentum: float = uniform(0., 1.)


def test_to_array():
    b: B = B.sample()
    array = b.to_array()
    assert array[0] == b.learning_rate
    assert array[1] == b.momentum


def test_from_array():
    array = np.arange(2)
    b: B = B.from_array(array)
    assert b.learning_rate == 0.
    assert b.momentum == 1.


def test_distance_between_same_object():
    x1 = A(learning_rate=1.2)
    assert x1.distance_to(x1) == 0


def test_distance_between_same_type():
    x1 = A(learning_rate=0.)
    x2 = A(learning_rate=1.)
    assert x1.distance_to(x2) == 1.


def test_distance_between_same_type_with_weights():
    x1 = A(learning_rate=0)
    x2 = A(learning_rate=0.8)
    weights = {"learning_rate": 0.5}
    assert x1.distance_to(x2, weights=weights) == 0.4
    assert x2.distance_to(x1, weights=weights) == 0.4


def test_distance_between_different_types():
    x1 = A(learning_rate=0.)
    x2 = B(learning_rate=0.5, momentum=0.2)
    assert x1.distance_to(x2) == 0.5
    assert x2.distance_to(x1) == 0.5


def test_distance_between_different_types_with_weights():
    x1 = A(learning_rate=0.)
    x2 = B(learning_rate=0.5, momentum=0.2)
    weights = {"learning_rate": 0.2}
    assert x1.distance_to(x2, weights=weights) == 0.1
    assert x2.distance_to(x1, weights=weights) == 0.1


def test_distance_between_different_type_with_equivalent_names():
    x1 = A(learning_rate=0.)
    x2 = C(lr=2.)
    assert x1.distance_to(x2) == 2.

    x1 = B(learning_rate=0., momentum=1)
    x2 = C(lr=1, momentum=0.5)

    assert x2.distance_to(x1) == 1.5
    assert x1.distance_to(x2) == 1.5


def test_distance_between_different_types_with_equivalent_names_with_weights():
    x1 = A(learning_rate=0.)
    x2 = C(lr=2.)
    weights = dict(learning_rate=0.5)
    assert x1.distance_to(x2, weights=weights) == 1.
    assert x2.distance_to(x1, weights=weights) == 1.


def test_distance_between_different_types_and_equivalent_names():
    x1 = A(learning_rate=0.)
    x2 = C(lr=0.6, momentum=0.2)
    assert x1.distance_to(x2) == 0.6
    assert x2.distance_to(x1) == 0.6
