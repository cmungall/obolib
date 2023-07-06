from dataclasses import dataclass
from typing import Any

from kgcl_schema.datamodel.kgcl import (
    ChangeLanguageElement,
    EdgeChange,
    NodeChange,
    NodeMove,
)
from linkml_runtime import CurieNamespace

from oaklib.interfaces.obograph_interface import OboGraphInterface
from oaklib.interfaces.taxon_constraint_interface import TaxonConstraintInterface
from oaklib.io.streaming_writer import StreamingWriter
from oaklib.utilities.obograph_utils import DEFAULT_PREDICATE_CODE_MAP

predicate_code_map = DEFAULT_PREDICATE_CODE_MAP


def _keyval(x: Any) -> str:
    if isinstance(x, CurieNamespace):
        return str(x.curie())
    # if isinstance(x, EnumDefinitionImpl):
    #    if x.curie:
    #        return str(x.curie)
    return str(x)


@dataclass
class StreamingMarkdownWriter(StreamingWriter):
    """
    A writer that streams curies as markdown
    """

    # TODO: replace with linkml-renderer

    def emit(self, curie, label=None, **kwargs):
        oi = self.ontology_interface
        if isinstance(curie, ChangeLanguageElement):
            if isinstance(curie, NodeChange):
                change = curie
                curie = change.about_node

            elif isinstance(curie, EdgeChange):
                change = curie
                # !Here we only have only SPO e.g.: 
                # ! subject='PATO:0045073', predicate='RO:0015010', object='PATO:0000461'
                # !It needs to have old and new values to show in the release notes.
                if isinstance(change, NodeMove):
                    # ! Here we have old_value but no new_value e.g. old_value='rdfs:subPropertyOf', new_value=None
                    curie = change.old_value

                else:
                    curie = change.predicate
                # import pdb; pdb.set_trace()

        if label is None:
            label = oi.label(curie)
        self.file.write(f"## {curie} {label}\n\n")
        defn = oi.definition(curie)
        if defn:
            self.file.write(f"_{defn}_\n\n")
        self.file.write("### Xrefs\n\n")

        for _, x in oi.simple_mappings_by_curie(curie):
            self.file.write(f" * {x}\n")
        self.file.write("\n")
        if isinstance(oi, OboGraphInterface):
            self.file.write("### Relationships\n\n")
            for k, vs in oi.outgoing_relationship_map(curie).items():
                p = predicate_code_map.get(k, None)
                if p is None:
                    p = oi.label(k)
                    if p is None:
                        p = k
                self.file.write(f"* {p}\n")
                for v in vs:
                    self.file.write(f'    * {v} "{oi.label(curie)}"\n')
        if (
            self.display_options
            and "t" in self.display_options
            and isinstance(oi, TaxonConstraintInterface)
        ):
            self.file.write("### Taxon Constraints\n\n")
            tc_subj = oi.get_term_with_taxon_constraints(curie)
            for tc in tc_subj.never_in:
                self.file.write(f"* {tc}\n")

        self.file.write("\n")
