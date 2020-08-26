import copy
import dataclasses
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import *

from simple_parsing import Serializable

from logging_utils import get_logger
from utils import dict_union, field_dict

logger = get_logger(__file__)
Dataclass = TypeVar("Dataclass")
T = TypeVar("T")
SourceType = TypeVar("SourceType")
TargetType = TypeVar("TargetType")

class Translator(ABC, Generic[SourceType, TargetType]):
    @abstractmethod
    def translate(self, value: SourceType, drop_rest: bool=False) -> TargetType:
        pass

class DictTranslator(Translator[Dict[str, Any], Dict[str, Any]]):
    """ Object responsible for 'translating' the keys with equivalent names
    between two dictionaries.
    """
    def __init__(self, target: Dict[str, Any], equivalent_keys_file: Path=None):
        self.target: Dict[str, Any] = target
        if equivalent_keys_file is None:
            # TODO: Download it from the internet and save it in a temp
            # directory or something like that? 
            equivalent_keys_file = Path("warm_start/equivalent_keys.json")
        self.equivalent_keys_file = equivalent_keys_file
        self.equivalent_base_key: Dict[str, str] = self.load_equivalences()

    def load_equivalences(self) -> Dict[str, str]:
        with open(self.equivalent_keys_file) as f:
            equivalences = {k: set(v) for k, v in json.load(f).items()}
        # a dictionary mapping from any 'name'xz
        corresponding_base_name: Dict[str, str] = {}
        for k, values in equivalences.items():
            for v in values:
                assert v not in corresponding_base_name, "shouldn't have any collisions here!"
                corresponding_base_name[v] = k
        return corresponding_base_name

    def get_base_name(self, k: str) -> Optional[str]:
        """Gets the equivalent 'base' key for a given key `k`.

        For instance, when given 'LearningRate', 'lr', 'learn_rate', etc, 
        will return 'learning_rate', which is the 'base' key for that 'group'
        or 'concept'.

        Args:
            k (str): a key.

        Returns:
            Optional[str]: The corresponding 'base' key for the key `k`.
        """
        # TODO: maybe preprocess the keys (lower(), etc) so we don't have to
        # store as many equivalences?
        k = k.lower()
        return self.equivalent_base_key.get(k)

    def translate(self, value: Dict[str, Any], drop_rest: bool=False) -> Dict[str, Any]:
        """'Translates' a dict by renaming the keys to their equivalent name in
        the target type.

        TODO: handle nested translation (maybe with each nested dicts having its own translator!)

        Args:
            value (Dict[str, Any]): A dictionary to translate.
            drop_rest (bool): Wether to drop the keys that don't match or to
                keep them in the resulting dict. Defaults to False.

        Returns:
            Dict[str, Any]: The 'translated' dict, containing either just the
            renamed keys (if `drop_rest` is `True`) or both the renamed and
            remaining keys which didn't match any of the target keys if
            `drop_rest` is False (default).
        """
        assert isinstance(value, dict)

        value = copy.deepcopy(value)
        if set(self.target) == set(value):
            # The dict has the same keys as the target, so we can just return it
            # directly to save some unneeded computation.
            return value

        equivalent_keys = self.find_equivalent_keys(value, self.target)

        new_dict: Dict[str, Any] = {} if drop_rest else value
        for old_key, new_key in equivalent_keys.items():
            # Take the value from the old hparams and assign it
            # to the corresponding attribute in the new type.
            old_value = value.pop(old_key)
            if new_key in new_dict and drop_rest:
                raise RuntimeError(
                    f"Colliding keys! Key {new_key} is already in {new_dict}!"
                    f"(translating {value} to use keys {self.target})"
                ) 
            new_dict[new_key] = old_value
        return new_dict

    def find_equivalent_keys(self, a_keys: Iterable[str], b_keys: Iterable[str]) -> Dict[str, str]:
        """Finds keys from a_keys and b_keys that are identical or equivalent.

        Two keys are equivalent if they relate to the same 'concept' but have a
        different spelling. For example, "learning_rate" and "lr". 

        Args:
            a_keys (Iterable[str]): A set of keys (attribute names) of a hparam class.
            b_keys (Iterable[str]): A set of keys (attribute names) of another (could
                be the same) hparam class.

        Returns:
            Dict[str, str]: A mapping from keys in `a_keys` to their equivalent
                key in `b_keys`. If the dict is empty, this means that there is
                no common/equivalent keys between the two given sets of keys.
        """
        result: Dict[str, str] = {}
        a_keys = set(a_keys)
        b_keys = set(b_keys)
        
        common_keys = a_keys.intersection(b_keys)
        for k in common_keys:
            result[k] = k

        logger.info(f"Common keys: {common_keys}")
        
        a: Set[str] = a_keys - common_keys
        b: Set[str] = b_keys - common_keys
        
        logger.info(f"leftover keys: {a}, {b}")

        # Dict that maps from the key in A to the 'root name' of that key in the
        # dict of equivalences.
        group_for_key_a: Dict[str, str] = {}
        # same for B.
        group_for_key_b: Dict[str, str] = {}
        
        # Find the containing 'group' for each key in `a`. 
        for key_a in a:
            root_name = self.get_base_name(key_a)
            if root_name:
                if root_name in group_for_key_a:
                    # TODO: We don't currently support this, for example having
                    # `lr` and `learning_rate` as two attributes on the same
                    # hparams class.
                    previous_key = group_for_key_a[root_name]
                    raise NotImplementedError(
                        f"TODO: attributes '{key_a}' and '{previous_key}' map "
                        f"to the same 'base' name ({root_name})!"
                    )
                group_for_key_a[root_name] = key_a
                break

        # Same for `b`. 
        for key_b in b:
            root_name = self.get_base_name(key_b)
            if root_name:
                assert root_name not in group_for_key_b, "TODO: two attributes have the same root name.."
                group_for_key_b[root_name] = key_b
                break
        
        for group, key_a in group_for_key_a.items():
            logger.info(f"Key {key_a} is in the group {group}.")
            if group in group_for_key_b:
                key_b = group_for_key_b[group]
                logger.info(f"Key {key_b} is in the same group, so they are equivalent!")
                result[key_a] = key_b

        return result     


class SimpleTranslator(DictTranslator, Translator[Union[Dict, Dataclass], Dict]):
    """ Adds the ability to translate dataclasses to the dict translator.
    """   
    def __init__(self, target: Union[Dict, Dataclass, Type[Dataclass]], equivalent_keys_file: Path=None):
        # convert the target to a dict.
        if dataclasses.is_dataclass(target):
            target = field_dict(target)
        assert isinstance(target, dict)
        super().__init__(target=target, equivalent_keys_file=equivalent_keys_file)

    def translate(self, value: Union[Dataclass, Dict[str, Any]], drop_rest: bool=False) -> Dict[str, Any]:
        if isinstance(value, Serializable):
            value = value.to_dict()
        elif dataclasses.is_dataclass(value):
            value = dataclasses.asdict(value)
        assert isinstance(value, dict)
        return super().translate(value, drop_rest=drop_rest)
