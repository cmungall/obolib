from abc import ABC
from typing import Dict, List, Tuple, Iterable

from oaklib.interfaces.basic_ontology_interface import BasicOntologyInterface
import sssom
from oaklib.types import CURIE


class MappingProviderInterface(BasicOntologyInterface, ABC):
    """
    An ontology provider that provides SSSOM mappings

    TODO: move code from mapping-walker
    """

    def all_mappings(self) -> Iterable[sssom.Mapping]:
        raise NotImplementedError

    def get_sssom_mappings_by_curie(self, curie: CURIE) -> Iterable[sssom.Mapping]:
        raise NotImplementedError

    def get_transitive_mappings_by_curie(self, curie: CURIE) -> Iterable[sssom.Mapping]:
        raise NotImplementedError

