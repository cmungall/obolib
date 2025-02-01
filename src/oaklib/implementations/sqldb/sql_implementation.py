import logging
import math
import re
import shutil
import typing
from collections import defaultdict
from dataclasses import dataclass, field
from operator import or_
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    Union,
)

import rdflib
import semsql.builder.builder as semsql_builder
import sqlalchemy.orm
from kgcl_schema.datamodel import kgcl
from linkml_runtime import SchemaView
from linkml_runtime.dumpers import json_dumper
from linkml_runtime.utils.introspection import package_schemaview
from linkml_runtime.utils.metamodelcore import URIorCURIE
from semsql.sqla.semsql import (  # HasMappingStatement,
    AnnotationPropertyNode,
    Base,
    ClassNode,
    DeprecatedNode,
    Edge,
    EntailedEdge,
    HasSynonymStatement,
    HasTextDefinitionStatement,
    IriNode,
    NamedIndividualNode,
    Node,
    NodeIdentifier,
    ObjectPropertyNode,
    OntologyNode,
    OwlAxiomAnnotation,
    OwlDisjointClassStatement,
    OwlEquivalentClassStatement,
    OwlHasValue,
    OwlSomeValuesFrom,
    Prefix,
    RdfFirstStatement,
    RdfRestStatement,
    RdfsLabelStatement,
    RdfsSubclassOfStatement,
    RdfTypeStatement,
    Statements,
    TermAssociation,
    TransitivePropertyNode,
)
from sqlalchemy import and_, create_engine, delete, distinct, func, insert, select, text, update
from sqlalchemy.orm import aliased, sessionmaker
from sssom_schema import Mapping

import oaklib.datamodels.ontology_metadata as om
import oaklib.datamodels.validation_datamodel as vdm
from oaklib.constants import FILE_CACHE
from oaklib.datamodels import obograph, ontology_metadata
from oaklib.datamodels.association import Association
from oaklib.datamodels.obograph import (
    DisjointClassExpressionsAxiom,
    ExistentialRestrictionExpression,
    LogicalDefinitionAxiom,
)
from oaklib.datamodels.search import SearchConfiguration
from oaklib.datamodels.search_datamodel import SearchProperty, SearchTermSyntax
from oaklib.datamodels.similarity import TermPairwiseSimilarity
from oaklib.datamodels.summary_statistics_datamodel import (
    ContributorStatistics,
    FacetedCount,
    GroupedStatistics,
    UngroupedStatistics,
)
from oaklib.datamodels.vocabulary import (
    ALL_CONTRIBUTOR_PREDICATES,
    ALL_MATCH_PREDICATES,
    DEPRECATED_PREDICATE,
    DISJOINT_WITH,
    ENTITY_LEVEL_DEFINITION_PREDICATES,
    EQUIVALENT_CLASS,
    HAS_DBXREF,
    HAS_DEFINITION_CURIE,
    HAS_EXACT_SYNONYM,
    HAS_OBSOLESCENCE_REASON,
    HAS_SYNONYM_TYPE,
    IN_CATEGORY_PREDS,
    IN_SUBSET,
    INVERSE_OF,
    IS_A,
    LABEL_PREDICATE,
    OBSOLETION_RELATIONSHIP_PREDICATES,
    OWL_CLASS,
    OWL_META_CLASSES,
    OWL_NAMED_INDIVIDUAL,
    OWL_NOTHING,
    OWL_PROPERTY_CHAIN_AXIOM,
    OWL_THING,
    OWL_VERSION_IRI,
    PREFIX_PREDICATE,
    RDF_TYPE,
    RDFS_COMMENT,
    RDFS_DOMAIN,
    RDFS_LABEL,
    RDFS_RANGE,
    SEMAPV,
    STANDARD_ANNOTATION_PROPERTIES,
    SUBPROPERTY_OF,
    SYNONYM_PREDICATES,
    TERM_REPLACED_BY,
    TERMS_MERGED,
    omd_slots,
)
from oaklib.implementations.sqldb import SEARCH_CONFIG
from oaklib.interfaces import SubsetterInterface, TextAnnotatorInterface
from oaklib.interfaces.association_provider_interface import EntityNormalizer
from oaklib.interfaces.basic_ontology_interface import (
    ALIAS_MAP,
    DEFINITION,
    LANGUAGE_TAG,
    METADATA_MAP,
    METADATA_STATEMENT,
    PRED_CURIE,
    PREFIX_MAP,
    RELATIONSHIP,
    BasicOntologyInterface,
)
from oaklib.interfaces.class_enrichment_calculation_interface import (
    ClassEnrichmentCalculationInterface,
)
from oaklib.interfaces.differ_interface import DifferInterface
from oaklib.interfaces.dumper_interface import DumperInterface
from oaklib.interfaces.mapping_provider_interface import MappingProviderInterface
from oaklib.interfaces.merge_interface import MergeInterface
from oaklib.interfaces.metadata_interface import MetadataInterface
from oaklib.interfaces.obograph_interface import GraphTraversalMethod, OboGraphInterface
from oaklib.interfaces.owl_interface import OwlInterface
from oaklib.interfaces.patcher_interface import PatcherInterface
from oaklib.interfaces.relation_graph_interface import RelationGraphInterface
from oaklib.interfaces.search_interface import SearchInterface
from oaklib.interfaces.semsim_interface import SemanticSimilarityInterface
from oaklib.interfaces.summary_statistics_interface import SummaryStatisticsInterface
from oaklib.interfaces.taxon_constraint_interface import TaxonConstraintInterface
from oaklib.interfaces.usages_interface import UsagesInterface
from oaklib.interfaces.validator_interface import ValidatorInterface
from oaklib.types import CATEGORY_CURIE, CURIE, SUBSET_CURIE
from oaklib.utilities.axioms.logical_definition_utilities import (
    logical_definition_matches,
)
from oaklib.utilities.format_utilities import (
    OBOGRAPHS_SYNTAX_ALIAS_MAP,
    RDFLIB_SYNTAX_ALIAS_MAP,
)
from oaklib.utilities.graph.relationship_walker import walk_down, walk_up
from oaklib.utilities.identifier_utils import (
    string_as_base64_curie,
    synonym_type_code_from_curie,
)

__all__ = [
    "get_range_xsd_type",
    "regex_to_sql_like",
    "SqlImplementation",
]

from oaklib.utilities.iterator_utils import chunk
from oaklib.utilities.mapping.sssom_utils import inject_mapping_sources

SUBJECT_REL_KEY = Tuple[CURIE, Optional[List[PRED_CURIE]], Tuple]


class SqlSchemaError(Exception):
    """Raised when there are issues with the version of the SQL DDL uses"""

    pass


class ViewNotFoundError(SqlSchemaError):
    """Raised when a SQL view is not found"""

    pass


def _is_blank(curie: CURIE) -> bool:
    return curie and curie.startswith("_:")


def _python_value(val: Any, datatype: CURIE = None) -> Any:
    if datatype == "xsd:integer":
        return int(val)
    elif datatype == "xsd:float":
        return float(val)
    elif datatype == "xsd:boolean":
        return bool(val)
    else:
        return val


def get_range_xsd_type(sv: SchemaView, rng: str) -> Optional[URIorCURIE]:
    t = sv.get_type(rng)
    if t.uri:
        return t.uri
    elif t.typeof:
        return get_range_xsd_type(sv, t.typeof)
    else:
        raise ValueError(f"No xsd type for {rng}")


def regex_to_sql_like(regex: str) -> str:
    """
    convert a regex to a LIKE

    * ``.*`` => ``%``
    * ``.`` => ``_``
    * ``^`` => ``%`` (at start of string)
    * ``$`` => ``%`` (at end of string)

    TODO: implement various different DBMS flavors
    https://stackoverflow.com/questions/20794860/regex-in-sql-to-detect-one-or-more-digit

    :param regex:
    :return:
    """
    for c in r"()[]{}|":
        if c in regex:
            raise NotImplementedError(
                f"Regex engine not implemented for SQL and cannot parse char {c} in {regex}"
            )
    like = regex.replace(".*", "%")
    like = like.replace(".", "_")
    if like.startswith("^"):
        like = like[1:]
    else:
        like = f"%{like}"
    if like.endswith("$"):
        like = like[0:-1]
    else:
        like = f"{like}%"
    logging.info(f"Translated {regex} => LIKE {like}")
    return like


def _is_quoted_url(curie: CURIE):
    return curie.startswith("<")


def _remove_uri_quotes(curie: CURIE):
    if _is_quoted_url(curie):
        return curie[1:-1]
    else:
        return curie


@dataclass
class SqlImplementation(
    RelationGraphInterface,
    OboGraphInterface,
    ValidatorInterface,
    SearchInterface,
    SubsetterInterface,
    MappingProviderInterface,
    PatcherInterface,
    SemanticSimilarityInterface,
    MetadataInterface,
    DifferInterface,
    # AssociationProviderInterface,
    ClassEnrichmentCalculationInterface,
    TaxonConstraintInterface,
    TextAnnotatorInterface,
    SummaryStatisticsInterface,
    OwlInterface,
    DumperInterface,
    MergeInterface,
    UsagesInterface,
):
    """
    A :class:`OntologyInterface` implementation that wraps a SQL Relational Database.

    Currently this must be a SQLite database. PostgreSQL support is planned.

    To connect, either use SqlImplementation directly:

    >>> from oaklib.implementations.sqldb.sql_implementation import SqlImplementation
    >>> from oaklib.resource import OntologyResource
    >>> adapter = SqlImplementation(OntologyResource("tests/input/go-nucleus.db"))

    or

    >>> from oaklib import get_adapter
    >>> adapter = get_adapter("sqlite:tests/input/go-nucleus.db")

    you can also load from the semantic-sql repository:

    >>> adapter = get_adapter("sqlite:obo:obi")

    The schema is assumed to follow the `semantic-sql <https://github.com/incatools/semantic-sql>`_ schema.

    This uses SQLAlchemy ORM Models:

    - :class:`Statements`
    - :class:`Edge`

    See Also
    --------
    - `Tutorial <https://incatools.github.io/ontology-access-kit/intro/tutorial07.html>`_
    - `SQL Implementation <https://incatools.github.io/ontology-access-kit/implementations/sqldb.html>`_

    """

    # TODO: use SQLA types
    engine: Any = None
    _session: Any = None
    _connection: Any = None
    _ontology_metadata_model: SchemaView = None
    _prefix_map: PREFIX_MAP = None
    _information_content_cache: Dict[Tuple, float] = None
    _relationships_by_subject_index: Dict[CURIE, List[RELATIONSHIP]] = None

    max_items_for_in_clause: int = field(default_factory=lambda: 100)

    can_store_associations: bool = False
    """True if the underlying sqlite database has term_association populated."""

    def __post_init__(self):
        if self.engine is None:
            locator = str(self.resource.slug)
            logging.info(f"Locator: {locator}")
            if locator.startswith("obo:"):
                # easter egg feature, to be documented:
                # The selector 'sqlite:obo:ONTOLOGY' will use a pre-generated
                # sqlite db of an OBO ontology after downloading from S3.
                # Note: this can take some time
                prefix = locator[len("obo:") :]
                # Option 1 uses direct URL construction:
                url = f"https://s3.amazonaws.com/bbop-sqlite/{prefix}.db.gz"
                logging.info(f"Ensuring gunzipped for {url}")
                db_path = FILE_CACHE.ensure_gunzip(url=url, autoclean=False)
                # Option 2 uses botocore to interface with the S3 API directly:
                # db_path = OAKLIB_MODULE.ensure_from_s3(s3_bucket="bbop-sqlite", s3_key=f"{prefix}.db")
                locator = f"sqlite:///{db_path}"
            if locator.endswith(".owl"):
                # this is currently an "Easter Egg" feature. It allows you to specify a locator
                # such as sqlite:/path/to/my.owl
                # then semsql will be invoked to build a sqlite db from this.
                # the same sqlite db will be reused until the timestamp of the owl file changes.
                # the catch is that EITHER the user must have BOTH rdftab and relation-graph installed, OR
                # they should be running through ODK docker
                locator = locator.replace(".owl", ".db").replace("sqlite:", "")
                logging.info(f"Building {locator} using semsql")
                semsql_builder.make(locator)
                locator = f"sqlite:///{locator}"
            else:
                path = Path(locator.replace("sqlite:///", "")).absolute()
                if not path.exists():
                    raise FileNotFoundError(f"File does not exist: {path}")
                locator = f"sqlite:///{path}"
            logging.info(f"Locator, post-processed: {locator}")
            self.engine = create_engine(locator)

            if logging.root.level <= logging.DEBUG:
                # If oaklib is running at DEBUG level logging; record all queries from sqlachemy on execution.
                # This line is equivalent to create_engine(locator,echo=True)
                logging.getLogger("sqlalchemy.engine.Engine").setLevel(logging.INFO)

    @property
    def session(self):
        if self._session is None:
            session_cls = sessionmaker(self.engine)
            self._session = session_cls()
        return self._session

    @property
    def connection(self):
        if self._connection is None:
            self._connection = self.engine.connect()
        return self._connection

    @property
    def ontology_metadata_model(self):
        if self._ontology_metadata_model is None:
            self._ontology_metadata_model = package_schemaview(ontology_metadata.__name__)
        return self._ontology_metadata_model

    def is_mysql(self):
        # TODO
        return False

    def is_postgres(self):
        # TODO
        return False

    def _check_has_view(
        self, sqla_class: Type[Base], minimum_version=None, fail_if_absent=True
    ) -> bool:
        engine = self.session.get_bind()
        tn = sqla_class.__tablename__
        cn = sqla_class.__name__
        has_view = sqlalchemy.inspect(engine).has_table(tn)
        if fail_if_absent and not has_view:
            raise ViewNotFoundError(
                f"View {tn} does not exist (required semsql v{minimum_version})"
                f"""
                              Potential remedies:
                              (1) obtain a new ready-made copy of the database, OR
                              (2) rebuild the database from source OWL, OR
                              (3) add the missing table to the database using CREATE VIEW {tn} AS ...
                              Using the definition in https://incatools.github.io/semantic-sql/{cn}"""
            )
        return has_view

    def prefix_map(self) -> PREFIX_MAP:
        if self._prefix_map is None:
            self._prefix_map = {row.prefix: row.base for row in self.session.query(Prefix)}
        return self._prefix_map

    def languages(self) -> Iterable[LANGUAGE_TAG]:
        for row in self.session.query(Statements.language).distinct():
            if row.language:
                yield row.language

    def entities(self, filter_obsoletes=True, owl_type=None) -> Iterable[CURIE]:
        # TODO: figure out how to pass through ESCAPE at SQL Alchemy level
        # s = text('SELECT id FROM class_node WHERE id NOT LIKE "\_:%" ESCAPE "\\"')  # noqa W605
        q = self.session.query(Node)
        if owl_type:
            subquery = self.session.query(RdfTypeStatement.subject).filter(
                RdfTypeStatement.object == owl_type
            )
            q = q.filter(Node.id.in_(subquery))
        if filter_obsoletes:
            obs_subq = self.session.query(DeprecatedNode.id)
            q = q.filter(Node.id.not_in(obs_subq))
        logging.info(f"Query: {q}")
        for row in q:
            if row:
                if not _is_blank(row.id) and not row.id.startswith("<urn:swrl"):
                    yield row.id

    def owl_types(self, entities: Iterable[CURIE]) -> Iterable[Tuple[CURIE, CURIE]]:
        q = self.session.query(RdfTypeStatement).filter(RdfTypeStatement.subject.in_(entities))
        for row in q:
            yield row.subject, row.object

    def obsoletes(self, include_merged=True) -> Iterable[CURIE]:
        q = self.session.query(DeprecatedNode)
        if not include_merged:
            subq = (
                self.session.query(Statements.subject)
                .filter(Statements.predicate == HAS_OBSOLESCENCE_REASON)
                .filter(Statements.object == TERMS_MERGED)
            )
            q = q.filter(DeprecatedNode.id.not_in(subq))
        for row in q:
            yield row.id

    def obsoletes_migration_relationships(
        self, entities: Iterable[CURIE]
    ) -> Iterable[RELATIONSHIP]:
        q = (
            self.session.query(Statements)
            .filter(Statements.subject.in_(entities))
            .filter(Statements.predicate.in_(OBSOLETION_RELATIONSHIP_PREDICATES))
        )
        for row in q:
            yield row.subject, row.predicate, row.object if row.object else row.value

    def all_relationships(self) -> Iterable[RELATIONSHIP]:
        for row in self.session.query(Edge):
            yield row.subject, row.predicate, row.object

    def _add_language_filter(self, q, lang, typ: Type[Statements] = Statements):
        if lang:
            if not self.multilingual:
                logging.warning(
                    f"Source does not appear to be multilingual, filtering by @{lang} may not work"
                )
            if lang == self.default_language:
                q = q.filter(typ.language.is_(None))
            else:
                q = q.filter(typ.language == lang)
        return q

    def label(self, curie: CURIE, lang: Optional[LANGUAGE_TAG] = None) -> Optional[str]:
        q = self.session.query(RdfsLabelStatement.value).filter(RdfsLabelStatement.subject == curie)
        q = self._add_language_filter(q, lang, RdfsLabelStatement)
        for (lbl,) in q:
            return lbl
        if lang and lang != self.default_language:
            return self.label(curie, lang=self.default_language)

    def labels(
        self, curies: Iterable[CURIE], allow_none=True, lang: LANGUAGE_TAG = None
    ) -> Iterable[Tuple[CURIE, str]]:
        for curie_it in chunk(curies, self.max_items_for_in_clause):
            curr_curies = list(curie_it)

            has_label = set()
            q = self.session.query(RdfsLabelStatement).filter(
                RdfsLabelStatement.subject.in_(tuple(curr_curies))
            )
            q = self._add_language_filter(q, lang, RdfsLabelStatement)
            for row in q:
                yield row.subject, row.value
                has_label.add(row.subject)
            if lang and lang != self.default_language:
                # fill missing
                curies_with_no_labels = set(curr_curies) - has_label
                yield from self.labels(curies_with_no_labels, allow_none=allow_none, lang=None)
                allow_none = False
            if allow_none:
                for curie in curr_curies:
                    if curie not in has_label:
                        yield curie, None

    def multilingual_labels(
        self, curies: Iterable[CURIE], allow_none=True, langs: Optional[List[LANGUAGE_TAG]] = None
    ) -> Iterable[Tuple[CURIE, str, LANGUAGE_TAG]]:
        if isinstance(curies, str):
            logging.warning(f"multilingual_labels called with a single curie: {curies}")
            curies = [curies]
        for curie_it in chunk(curies, self.max_items_for_in_clause):
            curr_curies = list(curie_it)

            has_label = defaultdict(set)
            q = self.session.query(RdfsLabelStatement).filter(
                RdfsLabelStatement.subject.in_(tuple(curr_curies))
            )
            if langs is not None:
                if self.default_language in langs:
                    q = q.filter(
                        or_(
                            RdfsLabelStatement.language.is_(None),
                            RdfsLabelStatement.language.in_(langs),
                        )
                    )
                else:
                    q = q.filter(RdfsLabelStatement.language.in_(langs))
            for row in q:
                yield row.subject, row.value, row.language
                if allow_none:
                    has_label[row.subject].add(row.language)
            if allow_none:
                for curie in curr_curies:
                    if curie not in has_label:
                        yield curie, None, None

    def curies_by_label(self, label: str) -> List[CURIE]:
        q = self.session.query(RdfsLabelStatement.subject)
        q = q.filter(RdfsLabelStatement.value == label)
        return list(set([row.subject for row in q]))

    def entity_alias_map(self, curie: CURIE) -> ALIAS_MAP:
        m = defaultdict(list)
        m[LABEL_PREDICATE] = [self.label(curie)]
        for row in self.session.query(HasSynonymStatement).filter(
            HasSynonymStatement.subject == curie
        ):
            m[row.predicate].append(row.value)
        return m

    def comments(
        self, curies: Iterable[CURIE], allow_none=True, lang: LANGUAGE_TAG = None
    ) -> Iterable[Tuple[CURIE, str]]:
        for curie_it in chunk(curies, self.max_items_for_in_clause):
            curr_curies = list(curie_it)

            has_comment = set()
            q = self.session.query(Statements).filter(Statements.subject.in_(tuple(curr_curies)))
            q = q.filter(Statements.predicate == RDFS_COMMENT)
            q = self._add_language_filter(q, lang, Statements)
            for row in q:
                yield row.subject, row.value
                has_comment.add(row.subject)
            if lang and lang != self.default_language:
                # fill missing
                curies_with_no_labels = set(curr_curies) - has_comment
                yield from self.labels(curies_with_no_labels, allow_none=allow_none, lang=None)
                allow_none = False
            if allow_none:
                for curie in curr_curies:
                    if curie not in has_comment:
                        yield curie, None

    def definition(self, curie: CURIE, lang: Optional[LANGUAGE_TAG] = None) -> Optional[str]:
        q = self.session.query(HasTextDefinitionStatement.value).filter(
            HasTextDefinitionStatement.subject == curie
        )
        q = self._add_language_filter(q, lang, HasTextDefinitionStatement)
        for (lbl,) in q:
            return lbl
        if lang and lang != self.default_language:
            return self.definition(curie, lang=self.default_language)

    def definitions(
        self,
        curies: Iterable[CURIE],
        include_metadata=False,
        include_missing=False,
        lang: Optional[LANGUAGE_TAG] = None,
    ) -> Iterator[DEFINITION]:
        curies = list(curies)
        has_definition = set()
        metadata_map: Dict[Tuple[str, Optional[str]], METADATA_MAP] = defaultdict(dict)
        if include_metadata:
            # definition metadata may be axiom annotations
            for aa in self._axiom_annotations_multi(curies, predicate=HAS_DEFINITION_CURIE):
                k = (aa.subject, aa.value)
                metadata = metadata_map[k]
                if aa.annotation_predicate not in metadata:
                    metadata[aa.annotation_predicate] = []
                metadata[aa.annotation_predicate].append(aa.annotation_value)
            # or direct annotations
            q = (
                self.session.query(Statements)
                .filter(Statements.predicate.in_(ENTITY_LEVEL_DEFINITION_PREDICATES))
                .filter(Statements.subject.in_(curies))
            )
            for row in q:
                # note that direct annotation is unable to distinguish between multiple definitions;
                # set to match all
                k = (row.subject, None)
                metadata = metadata_map[k]
                if row.predicate not in metadata:
                    metadata[row.predicate] = []
                metadata[row.predicate].append(row.value if row.value else row.object)
        q = self.session.query(HasTextDefinitionStatement).filter(
            HasTextDefinitionStatement.subject.in_(curies)
        )
        q = self._add_language_filter(q, lang, HasTextDefinitionStatement)
        curr_curies = list(curies)
        for row in q:
            reification_metadata = metadata_map.get((row.subject, row.value), {})
            direct_metadata = metadata_map.get((row.subject, None), {})
            yield row.subject, row.value, {**reification_metadata, **direct_metadata}
            has_definition.add(row.subject)
        if lang and lang != self.default_language:
            # fill missing
            curies_with_no_defs = set(curr_curies) - has_definition
            yield from self.definitions(curies_with_no_defs, lang=None)
        if include_missing:
            for curie in curies:
                if curie not in has_definition:
                    yield curie, None, None

    def _definitions_with_metadata(
        self, curies: Iterable[CURIE], include_missing=False
    ) -> Iterator[DEFINITION]:
        curies = list(curies)
        has_definition = set()
        q = self.session.query(HasTextDefinitionStatement)
        q.filter(HasTextDefinitionStatement.subject.in_(curies))
        q.join(HasTextDefinitionStatement.metadata)
        for row in self.session.query(HasTextDefinitionStatement).filter(
            HasTextDefinitionStatement.subject.in_(curies)
        ):
            yield row.subject, row.value, row.source, row.date
            if include_missing:
                has_definition.add(row.subject)
        if include_missing:
            for curie in curies:
                if curie not in has_definition:
                    yield curie, None, None, None

    def entity_metadata_map(self, curie: CURIE, include_all_triples=False) -> METADATA_MAP:
        m = defaultdict(list)
        m["id"] = [curie]
        q = self.session.query(Statements)
        if not include_all_triples:
            subquery = self.session.query(RdfTypeStatement.subject).filter(
                RdfTypeStatement.object == "owl:AnnotationProperty"
            )
            annotation_properties = {row.subject for row in subquery}
            annotation_properties = annotation_properties.union(STANDARD_ANNOTATION_PROPERTIES)
            q = q.filter(Statements.predicate.in_(tuple(annotation_properties)))
        for row in q.filter(Statements.subject == curie):
            if row.value is not None:
                v = _python_value(row.value, row.datatype)
            elif row.object is not None:
                v = row.object
            else:
                v = None
            m[row.predicate].append(v)
        self.add_missing_property_values(curie, m)
        return dict(m)

    def entities_metadata_statements(
        self,
        curies: Iterable[CURIE],
        predicates: Optional[List[PRED_CURIE]] = None,
        include_nested_metadata=False,
        **kwargs,
    ) -> Iterator[METADATA_STATEMENT]:
        q = self.session.query(Statements)
        if not include_nested_metadata:
            subquery = self.session.query(RdfTypeStatement.subject).filter(
                RdfTypeStatement.object == "owl:AnnotationProperty"
            )
            annotation_properties = {row.subject for row in subquery}
            annotation_properties = annotation_properties.union(STANDARD_ANNOTATION_PROPERTIES)
            q = q.filter(Statements.predicate.in_(tuple(annotation_properties)))
        q = q.filter(Statements.subject.in_(curies))
        if predicates is not None:
            q = q.filter(Statements.predicate.in_(predicates))
        for row in q:
            if row.value is not None:
                v = _python_value(row.value, row.datatype)
            elif row.object is not None:
                v = row.object
            else:
                v = None
            yield row.subject, row.predicate, v, row.datatype, {}

    def ontologies(self) -> Iterable[CURIE]:
        for row in self.session.query(OntologyNode):
            yield row.id

    def ontology_versions(self, ontology: CURIE) -> Iterable[str]:
        q = self.session.query(RdfsLabelStatement).filter(RdfsLabelStatement.subject == ontology)
        q = q.filter(RdfsLabelStatement.predicate == OWL_VERSION_IRI)
        for row in q:
            yield row.object

    def ontology_metadata_map(self, ontology: CURIE) -> METADATA_MAP:
        return self.entity_metadata_map(ontology, include_all_triples=True)

    def _get_subset_curie(self, curie: str) -> str:
        if "#" in curie:
            return curie.split("#")[-1]
        else:
            return curie

    def _subset_uri_to_curie_map(self) -> Dict[str, CURIE]:
        m = {}
        for row in self.session.query(Statements.object).filter(Statements.predicate == IN_SUBSET):
            uri = row.object
            m[uri] = self._get_subset_curie(row.object)
        return m

    def _subset_curie_to_uri_map(self) -> Dict[CURIE, str]:
        m = {}
        for row in self.session.query(Statements.object, Statements.value).filter(
            Statements.predicate == IN_SUBSET
        ):
            uri = row.object
            if row.object is None:
                logging.warning(f"Subset may be incorrectly encoded as value for {row.value}")
            else:
                m[self._get_subset_curie(row.object)] = uri
        return m

    def subsets(self) -> Iterable[SUBSET_CURIE]:
        for s in self._subset_curie_to_uri_map().keys():
            yield s

    def subset_members(self, subset: SUBSET_CURIE) -> Iterable[CURIE]:
        sm = self._subset_curie_to_uri_map()
        if subset not in sm:
            raise ValueError(f"Subset {subset} not found in {sm}")
        for row in self.session.query(Statements.subject).filter(
            Statements.predicate == IN_SUBSET, Statements.object == sm[subset]
        ):
            yield self._get_subset_curie(row.subject)

    def terms_subsets(self, curies: Iterable[CURIE]) -> Iterable[Tuple[CURIE, SUBSET_CURIE]]:
        for row in self.session.query(Statements).filter(
            Statements.predicate == IN_SUBSET, Statements.subject.in_(list(curies))
        ):
            yield row.subject, self._get_subset_curie(row.object)

    def terms_categories(self, curies: Iterable[CURIE]) -> Iterable[Tuple[CURIE, CATEGORY_CURIE]]:
        for row in self.session.query(Statements).filter(
            Statements.predicate.in_(IN_CATEGORY_PREDS), Statements.subject.in_(list(curies))
        ):
            yield row.subject, self._get_subset_curie(row.object)

    def _execute(self, stmt):
        self.session.execute(stmt)
        self.session.flush()
        if self.autosave:
            self.save()

    def set_label(self, curie: CURIE, label: str) -> bool:
        existing_label = self.label(curie)
        if existing_label:
            stmt = (
                update(Statements)
                .where(and_(Statements.subject == curie, Statements.predicate == LABEL_PREDICATE))
                .values(value=label)
            )
        else:
            stmt = (
                insert(Statements)
                .values(subject=curie, predicate=LABEL_PREDICATE, value=label)
                .execution_options(autocommit=True)
            )
        self._execute(stmt)

    def basic_search(self, search_term: str, config: SearchConfiguration = None) -> Iterable[CURIE]:
        if config is None:
            config = SEARCH_CONFIG
        if config.force_case_insensitive:
            # in sqlite, LIKEs are case insensitive
            if config.syntax:
                if config.syntax != SearchTermSyntax(SearchTermSyntax.SQL):
                    raise ValueError(
                        f"Cannot force case insensitive search with syntax {config.syntax}"
                    )
            else:
                config.syntax = SearchTermSyntax(SearchTermSyntax.SQL)
        preds = []
        preds.append(omd_slots.label.curie)
        search_all = SearchProperty(SearchProperty.ANYTHING) in config.properties
        if search_all or SearchProperty(SearchProperty.ALIAS) in config.properties:
            preds += SYNONYM_PREDICATES
        if search_all or SearchProperty(SearchProperty.MAPPED_IDENTIFIER) in config.properties:
            preds += ALL_MATCH_PREDICATES
        view = Statements

        def make_query(qcol, preds, scol=view.subject):
            q = self.session.query(scol).filter(view.predicate.in_(tuple(preds)))
            if config.syntax == SearchTermSyntax(SearchTermSyntax.STARTS_WITH):
                q = q.filter(qcol.like(f"{search_term}%"))
            elif config.syntax == SearchTermSyntax(SearchTermSyntax.SQL):
                q = q.filter(qcol.like(search_term))
            elif config.syntax == SearchTermSyntax(SearchTermSyntax.REGULAR_EXPRESSION):
                if self.is_mysql():
                    q = q.filter(qcol.op("regex")(search_term))
                elif self.is_postgres():
                    q = q.filter(qcol.op("~")(search_term))
                else:
                    q = q.filter(qcol.like(regex_to_sql_like(search_term)))
            elif config.is_partial:
                q = q.filter(qcol.like(f"%{search_term}%"))
            else:
                q = q.filter(qcol == search_term)
            return q

        q = make_query(view.value, preds)
        for row in q.distinct():
            if row.subject.startswith("_:"):
                continue
            yield str(row.subject)
        if search_all or SearchProperty(SearchProperty.IDENTIFIER) in config.properties:
            q = make_query(view.subject, preds)
            for row in q.distinct():
                yield str(row.subject)
        if search_all or SearchProperty(SearchProperty.REPLACEMENT_IDENTIFIER) in config.properties:
            q = make_query(view.subject, [TERM_REPLACED_BY], view)
            for row in q.distinct():
                yield str(row.object) if row.object else str(row.value)

    def entailed_outgoing_relationships(
        self, curie: CURIE, predicates: List[PRED_CURIE] = None
    ) -> Iterable[Tuple[PRED_CURIE, CURIE]]:
        return self.outgoing_relationships(curie, predicates, entailed=True)

    def _rebuild_relationship_index(self):
        self._relationships_by_subject_index = None
        self.precompute_lookups()

    def precompute_lookups(self) -> None:
        if self._relationships_by_subject_index is None:
            self._relationships_by_subject_index = {}
        logging.info("Precomputing lookups")

        def add(row):
            if not row.object:
                # https://github.com/INCATools/ontology-access-kit/issues/457
                logging.warning(f"Bad row: {row}")
                return
            s = row.subject
            if s not in self._relationships_by_subject_index:
                self._relationships_by_subject_index[s] = []
            self._relationships_by_subject_index[s].append((s, row.predicate, row.object))

        q = self.session.query(Edge.subject, Edge.predicate, Edge.object)
        for row in q:
            add(row)
        q = self.session.query(Statements.subject, Statements.predicate, Statements.object)
        q = q.filter(Statements.predicate.in_((RDF_TYPE, RDFS_DOMAIN, RDFS_RANGE, INVERSE_OF)))
        for row in q:
            add(row)
        q = self.session.query(Statements.subject, Statements.predicate, Statements.object)
        op_subq = self.session.query(ObjectPropertyNode.id)
        q = q.filter(Statements.predicate.in_(op_subq))
        for row in q:
            add(row)

    def relationships(
        self,
        subjects: List[CURIE] = None,
        predicates: List[PRED_CURIE] = None,
        objects: List[CURIE] = None,
        include_tbox: bool = True,
        include_abox: bool = True,
        include_entailed: bool = False,
        include_dangling: bool = True,
        exclude_blank: bool = True,
        invert: bool = False,
        bypass_index: bool = False,
    ) -> Iterator[RELATIONSHIP]:
        if invert:
            for s, p, o in self.relationships(
                subjects=objects,
                predicates=predicates,
                objects=subjects,
                include_tbox=include_tbox,
                include_abox=include_abox,
                include_entailed=include_entailed,
                exclude_blank=exclude_blank,
            ):
                yield o, p, s
            return
        if subjects is not None:
            # materialize iterators
            subjects = list(subjects)
        if subjects and not objects and self._relationships_by_subject_index:
            # TODO: unify indexes with other implementations
            for s in subjects:
                for _, p, o in self._relationships_by_subject_index.get(s, []):
                    if not o:
                        raise ValueError(f"No object for {s} {p}")
                    if predicates and p not in predicates:
                        continue
                    if not include_abox and p == RDF_TYPE:
                        continue
                    if self.exclude_owl_top_and_bottom and o == OWL_THING:
                        continue
                    if self.exclude_owl_top_and_bottom and s == OWL_NOTHING:
                        continue
                    if exclude_blank and (_is_blank(s) or _is_blank(o)):
                        continue
                    if p == RDF_TYPE and o in OWL_META_CLASSES:
                        continue
                    yield s, p, o
            return
        if subjects is not None and len(subjects) > self.max_items_for_in_clause:
            logging.info(
                f"Chunking {len(subjects)} subjects into subqueries to avoid large IN clauses"
            )
            for subjects_it in chunk(subjects, self.max_items_for_in_clause):
                for r in self.relationships(
                    list(subjects_it),
                    predicates,
                    objects,
                    include_tbox=include_tbox,
                    include_abox=include_abox,
                    include_entailed=include_entailed,
                    include_dangling=include_dangling,
                    exclude_blank=exclude_blank,
                ):
                    yield r
            return
        if objects is not None and len(objects) > self.max_items_for_in_clause:
            logging.info(
                f"Chunking {len(objects)} objects into subqueries to avoid large IN clauses"
            )
            for objects_it in chunk(objects, self.max_items_for_in_clause):
                for r in self.relationships(
                    subjects,
                    predicates,
                    list(objects_it),
                    include_tbox=include_tbox,
                    include_abox=include_abox,
                    include_entailed=include_entailed,
                    include_dangling=include_dangling,
                    exclude_blank=exclude_blank,
                ):
                    yield r
            return
        logging.debug(f"Relationships for s={subjects} p={predicates} o={objects}")
        if include_tbox:
            for s, p, o in self._tbox_relationships(
                subjects, predicates, objects, include_entailed=include_entailed
            ):
                if self.exclude_owl_top_and_bottom and o == OWL_THING:
                    continue
                if self.exclude_owl_top_and_bottom and s == OWL_NOTHING:
                    continue
                if exclude_blank and (_is_blank(s) or _is_blank(o)):
                    continue
                yield s, p, o
            for s, p, o in self._equivalent_class_relationships(subjects, predicates, objects):
                if exclude_blank and (_is_blank(s) or _is_blank(o)):
                    continue
                yield s, p, o
            if subjects or objects:
                for s, p, o in self._equivalent_class_relationships(objects, predicates, subjects):
                    if exclude_blank and (_is_blank(s) or _is_blank(o)):
                        continue
                    yield o, p, s
            for s, p, o in self._rbox_relationships(subjects, predicates, objects):
                if exclude_blank and (_is_blank(s) or _is_blank(o)):
                    continue
                yield s, p, o
        if include_abox:
            for s, p, o in self._rdf_type_relationships(subjects, predicates, objects):
                if exclude_blank and (_is_blank(s) or _is_blank(o)):
                    continue
                yield s, p, o
            for s, p, o in self._subclass_of_has_value_relationships(subjects, predicates, objects):
                if exclude_blank and (_is_blank(s) or _is_blank(o)):
                    continue
                yield s, p, o
            for s, p, o in self._object_property_assertion_relationships(
                subjects, predicates, objects
            ):
                if exclude_blank and (_is_blank(s) or _is_blank(o)):
                    continue
                yield s, p, o

    def _tbox_relationships(
        self,
        subjects: List[CURIE] = None,
        predicates: List[PRED_CURIE] = None,
        objects: List[CURIE] = None,
        include_entailed: bool = False,
        include_dangling: bool = True,
    ) -> Iterator[RELATIONSHIP]:
        if include_entailed:
            tbl = EntailedEdge
        else:
            tbl = Edge
        q = self.session.query(tbl)
        if subjects:
            q = q.filter(tbl.subject.in_(tuple(subjects)))
        if predicates:
            q = q.filter(tbl.predicate.in_(tuple(predicates)))
        if objects:
            q = q.filter(tbl.object.in_(tuple(objects)))
        if not include_dangling:
            subq = self.session.query(Statements.subject)
            q = q.filter(tbl.object.in_(subq))
        logging.debug(f"Tbox query: {q}")
        for row in q:
            yield row.subject, row.predicate, row.object

    def _object_property_assertion_relationships(
        self,
        subjects: List[CURIE] = None,
        predicates: List[PRED_CURIE] = None,
        objects: List[CURIE] = None,
    ) -> Iterator[RELATIONSHIP]:
        q = self.session.query(Statements.subject, Statements.predicate, Statements.object)
        if subjects:
            q = q.filter(Statements.subject.in_(tuple(subjects)))
        if predicates:
            predicates = set(predicates).difference(
                {IS_A, RDF_TYPE, SUBPROPERTY_OF, RDFS_DOMAIN, RDFS_RANGE, INVERSE_OF}
            )
            if not predicates:
                return
            q = q.filter(Statements.predicate.in_(tuple(predicates)))
        else:
            op_subq = self.session.query(ObjectPropertyNode.id)
            q = q.filter(Statements.predicate.in_(op_subq))
        if objects:
            q = q.filter(Statements.object.in_(tuple(objects)))
        logging.debug(f"Abox query: {q}")
        for row in q:
            if not row.object:
                # edge case: see https://github.com/monarch-initiative/phenio/issues/36
                logging.warning(f"Invalid triple for S:{row.subject} P:{row.predicate}")
                continue
            yield row.subject, row.predicate, row.object

    def _subclass_of_has_value_relationships(
        self,
        subjects: List[CURIE] = None,
        predicates: List[PRED_CURIE] = None,
        objects: List[CURIE] = None,
    ) -> Iterator[RELATIONSHIP]:
        sc = aliased(RdfsSubclassOfStatement)
        hv = aliased(OwlHasValue)
        stmt = select(sc.subject, hv.on_property, hv.filler).join(hv, sc.object == hv.id)
        if subjects:
            stmt = stmt.where(sc.subject.in_(tuple(subjects)))
        if predicates:
            predicates = set(predicates).difference(
                {IS_A, RDF_TYPE, SUBPROPERTY_OF, RDFS_DOMAIN, RDFS_RANGE, INVERSE_OF}
            )
            if not predicates:
                return
            stmt = stmt.where(hv.on_property.in_(tuple(predicates)))
        if objects:
            stmt = stmt.where(hv.filler.in_(tuple(objects)))

        logging.debug(f"Abox HasValue query: {stmt}")

        result = self.session.execute(stmt)
        for row in result:
            yield row.subject, row.on_property, row.filler

    def _rdf_type_relationships(
        self,
        subjects: List[CURIE] = None,
        predicates: List[PRED_CURIE] = None,
        objects: List[CURIE] = None,
        include_dangling: bool = True,
    ) -> Iterator[RELATIONSHIP]:
        if predicates and RDF_TYPE not in predicates:
            return
        q = self.session.query(Statements.subject, Statements.object).filter(
            Statements.predicate == RDF_TYPE
        )
        if subjects:
            q = q.filter(Statements.subject.in_(tuple(subjects)))
        if objects:
            q = q.filter(Statements.object.in_(tuple(objects)))
        cls_subq = self.session.query(ClassNode.id)
        q = q.filter(Statements.object.in_(cls_subq))
        logging.info(f"ClassAssertion query: {q}")
        for row in q:
            yield row.subject, RDF_TYPE, row.object

    def _equivalent_class_relationships(
        self,
        subjects: List[CURIE] = None,
        predicates: List[PRED_CURIE] = None,
        objects: List[CURIE] = None,
    ) -> Iterator[RELATIONSHIP]:
        if predicates and EQUIVALENT_CLASS not in predicates:
            return
        q = self.session.query(Statements.subject, Statements.predicate, Statements.object)
        q = q.filter(Statements.predicate == EQUIVALENT_CLASS)
        if subjects:
            q = q.filter(Statements.subject.in_(tuple(subjects)))
        if objects:
            q = q.filter(Statements.object.in_(tuple(objects)))
        cls_subq = self.session.query(ClassNode.id)
        q = q.filter(Statements.object.in_(cls_subq))
        logging.info(f"ECA query: {q}")
        for row in q:
            yield row.subject, row.predicate, row.object

    def _rbox_relationships(
        self,
        subjects: List[CURIE] = None,
        predicates: List[PRED_CURIE] = None,
        objects: List[CURIE] = None,
    ) -> Iterator[RELATIONSHIP]:
        rbox_predicates = {RDFS_DOMAIN, RDFS_RANGE, INVERSE_OF}
        if predicates:
            rbox_predicates = rbox_predicates.intersection(predicates)
            if not predicates:
                return
        q = self.session.query(Statements.subject, Statements.predicate, Statements.object)
        q = q.filter(Statements.predicate.in_(tuple(rbox_predicates)))
        if subjects:
            q = q.filter(Statements.subject.in_(tuple(subjects)))
        if objects:
            q = q.filter(Statements.object.in_(tuple(objects)))
        logging.debug(f"RBOX query: {q}")
        for row in q:
            yield row.subject, row.predicate, row.object

    def relationships_metadata(
        self, relationships: Iterable[RELATIONSHIP], **kwargs
    ) -> Iterator[Tuple[RELATIONSHIP, List[Tuple[PRED_CURIE, Any]]]]:
        """For given relationships, returns axiom annotation metadata"""
        # Fetch all annotations
        anns: List[om.Annotation] = self._axiom_annotations_all()
        # Simplify objects
        anns: List[Dict[str, str]] = [{
            'subject': x.subject,
            'object': x.object,
            'predicate': x.predicate,
            'annotation_predicate': x.annotation_predicate,
            'annotation_value_or_obj': x.annotation_value if x.annotation_value is not None else x.annotation_object,
        } for x in anns]
        # Create lookup for annotations based using edges as keys
        rel_anns: Dict[RELATIONSHIP, List[Tuple[PRED_CURIE, Any]]] = {}
        for x in anns:
            key = (x['subject'], x['predicate'], x['object'])
            if key not in rel_anns:
                rel_anns[key] = []
            rel_anns[key].append((x['annotation_predicate'], x['annotation_value_or_obj']))
        # Group annotations by relationship, and yield
        rel_anns = {rel: rel_anns[rel] if rel in rel_anns else [] for rel in relationships}
        for rel, anns in rel_anns.items():
            yield rel, anns

    def relationships_metadata2(
        self, relationships: Iterable[RELATIONSHIP], **kwargs
    ) -> Iterator[Tuple[RELATIONSHIP, List[Tuple[PRED_CURIE, Any]]]]:
        """For given relationships, returns axiom annotation metadata"""
        rels: List[RELATIONSHIP] = [x for x in relationships]
        # Split based on predicate
        preds: Set[PRED_CURIE] = {x[1] for x in rels}
        rel_anns_all = {}
        for pred in preds:
            # Filter rels
            rels_i = [x for x in rels if x[1] == pred]
            # Fetch annotations for given relationships
            subs, objs = [x[0] for x in rels_i], [x[2] for x in rels_i]
            anns: List[om.Annotation] = [x for x in self._axiom_annotations_multi(subs, pred, objs)]
            anns: List[Dict[str, str]] = [{
                'subject': row.subject,
                'object': row.object,
                'predicate': row.predicate,
                'annotation_predicate': row.annotation_predicate,
                'annotation_value_or_obj': row.annotation_value if row.annotation_value is not None else row.annotation_object,
            } for row in anns]
            rel_anns: Dict[RELATIONSHIP, List[Tuple[PRED_CURIE, Any]]] = {}
            for x in anns:
                key = (x['subject'], x['predicate'], x['object'])
                if key not in rel_anns:
                    rel_anns[key] = []
                rel_anns[key].append((x['annotation_predicate'], x['annotation_value_or_obj']))
            # Group annotations by relationship
            rel_anns = {rel: rel_anns[rel] if rel in rel_anns else [] for rel in rels_i}
            rel_anns_all.update(rel_anns)
        # Yield
        for rel, anns in rel_anns_all.items():
            yield rel, anns

    def node_exists(self, curie: CURIE) -> bool:
        return self.session.query(Statements).filter(Statements.subject == curie).count() > 0

    def check_node_exists(self, curie: CURIE) -> None:
        if not self.node_exists(curie):
            raise ValueError(f"Node {curie} does not exist")

    def simple_mappings_by_curie(
        self, curie: CURIE, bidirectional=False
    ) -> Iterable[Tuple[PRED_CURIE, CURIE]]:
        for mpg in self.sssom_mappings(curies=curie, bidirectional=bidirectional):
            yield mpg.predicate_id, mpg.object_id

    def query(
        self, query: str, syntax: str = None, prefixes: List[str] = None, **kwargs
    ) -> Iterator[Any]:
        q = text(query)
        resultset = self.connection.execute(q)
        m = resultset.mappings()
        for d in m:
            yield dict(d)

    def clone(self, resource: Any) -> None:
        if self.resource.scheme == "sqlite":
            if resource.scheme == "sqlite":
                shutil.copyfile(self.resource.slug, resource.slug)
                new_oi = type(self)(resource)
                return new_oi
        raise NotImplementedError("Can only clone sqlite to sqlite")

    def as_rdflib_graph(self) -> rdflib.Graph:
        g = rdflib.Graph()
        bnodes = {}

        uri_re = re.compile(r"^<(.*)>$")

        def tr(n: str, v: str = None, datatype: str = None):
            if n:
                uri_match = uri_re.match(n)
                if n.startswith("_"):
                    if n not in bnodes:
                        bnodes[n] = rdflib.BNode()
                    return bnodes[n]
                elif uri_match:
                    return rdflib.URIRef(uri_match.group(1))
                else:
                    return rdflib.URIRef(self.curie_to_uri(n))
            else:
                lit = rdflib.Literal(v, datatype=datatype)
                return lit

        for row in self.session.query(Statements):
            s = tr(row.subject)
            p = tr(row.predicate)
            o = tr(row.object, row.value, row.datatype)
            logging.debug(f"Triple {s} {p} {o}")
            g.add((s, p, o))
        return g

    def dump(self, path: str = None, syntax: str = None, **kwargs):
        """
        Implements :ref:`dump`.

        Supported syntaxes:

        - ttl
        - json

        :param path:
        :param syntax:
        :param kwargs:
        :return:
        """
        if syntax is None:
            syntax = "ttl"
        if syntax in ["ttl", "rdfxml", "owl"]:
            if syntax in RDFLIB_SYNTAX_ALIAS_MAP:
                syntax = RDFLIB_SYNTAX_ALIAS_MAP[syntax]
            g = self.as_rdflib_graph()
            logging.info(f"Dumping to {path}")
            g.serialize(path, format=syntax)
        elif syntax in OBOGRAPHS_SYNTAX_ALIAS_MAP.keys():
            syntax = OBOGRAPHS_SYNTAX_ALIAS_MAP[syntax]
            g = self.as_obograph(expand_curies=True)
            gd = obograph.GraphDocument(graphs=[g])
            json_dumper.dump(gd, path)
        elif syntax == "sqlite":
            raise NotImplementedError
        else:
            super().dump(path, syntax)

    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # Implements: AssocationProviderInterface
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    def associations(self, *args, **kwargs) -> Iterator[Association]:
        if not self.can_store_associations:
            logging.info("Using base method")
            yield from super().associations(*args, **kwargs)
            return
        logging.info("Using SQL queries")
        q = self._associations_query(*args, **kwargs)
        for row_it in chunk(q):
            assocs = []
            for row in row_it:
                association = Association(
                    row.subject, row.predicate, row.object, primary_knowledge_source=row.source
                )
                assocs.append(association)
            subjects = {a.subject for a in assocs}
            label_map = {s: name for s, name in self.labels(subjects)}
            for association in assocs:
                if association.subject in label_map:
                    association.subject_label = label_map[association.subject]
            yield from assocs

    def _associations_query(
        self,
        subjects: Iterable[CURIE] = None,
        predicates: Iterable[PRED_CURIE] = None,
        objects: Iterable[CURIE] = None,
        property_filter: Dict[PRED_CURIE, Any] = None,
        subject_closure_predicates: Optional[List[PRED_CURIE]] = None,
        predicate_closure_predicates: Optional[List[PRED_CURIE]] = None,
        object_closure_predicates: Optional[List[PRED_CURIE]] = None,
        include_modified: bool = False,
        query: sqlalchemy.orm.Query = None,
        add_closure_fields: bool = False,
        **kwargs,
    ) -> Any:
        if query:
            q = query
        else:
            q = self.session.query(TermAssociation)
        if property_filter:
            raise NotImplementedError
        if subjects:
            if subject_closure_predicates:
                subquery = self.session.query(EntailedEdge.subject).filter(
                    EntailedEdge.object.in_(objects)
                )
                subquery = subquery.filter(EntailedEdge.predicate.in_(subject_closure_predicates))
                logging.info(f"Object subquery: {q} // {object_closure_predicates}")
                q = q.filter(TermAssociation.subject.in_(subquery))
            else:
                q = q.filter(TermAssociation.subject.in_(tuple(subjects)))
        if predicates:
            if predicate_closure_predicates:
                raise NotImplementedError
            else:
                q = q.filter(TermAssociation.predicate.in_(tuple(predicates)))
        if objects:
            if object_closure_predicates:
                subquery = self.session.query(EntailedEdge.subject).filter(
                    EntailedEdge.object.in_(objects)
                )
                subquery = subquery.filter(EntailedEdge.predicate.in_(object_closure_predicates))
                logging.info(f"Object subquery: {q} // {object_closure_predicates}")
                q = q.filter(TermAssociation.object.in_(subquery))
            else:
                q = q.filter(TermAssociation.object.in_(tuple(objects)))
        logging.info(f"_associations_query: {q}")
        return q

    def add_associations(
        self,
        associations: Iterable[Association],
        normalizers: List[EntityNormalizer] = None,
        **kwargs,
    ) -> bool:
        if not self.can_store_associations:
            logging.info("Using base method to store associations")
            return super().add_associations(associations, normalizers=normalizers)
        for a in associations:
            if normalizers:
                a = a.normalize(normalizers)
            if a.property_values:
                raise NotImplementedError
            stmt = insert(TermAssociation).values(
                subject=a.subject,
                predicate=a.predicate,
                object=a.object,
                source=a.primary_knowledge_source,
            )
            self._execute(stmt)
            if a.subject_label:
                stmt = insert(Statements).values(
                    subject=a.subject,
                    predicate=RDFS_LABEL,
                    value=a.subject_label,
                )
                self._execute(stmt)
        self.session.flush()
        return True

    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # Implements: OboGraphInterface
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    def node(
        self, curie: CURIE, strict=False, include_metadata=False, expand_curies=False
    ) -> obograph.Node:
        logging.debug(f"Node lookup: {curie}")
        meta = obograph.Meta()
        uri = self.curie_to_uri(curie) if expand_curies else curie
        n = obograph.Node(id=uri, meta=meta)
        q = self.session.query(Statements).filter(Statements.subject == curie)
        builtin_preds = [IS_A, DISJOINT_WITH]
        q = q.filter(Statements.predicate.not_in(builtin_preds))
        rows = list(q)

        def _anns_to_xrefs_and_meta(parent_pv: obograph.PropertyValue, anns: List[om.Annotation]):
            parent_pv.xrefs = [ann.object for ann in anns if ann.predicate == HAS_DBXREF]
            if isinstance(parent_pv, obograph.SynonymPropertyValue):
                synonym_types = [ann.object for ann in anns if ann.predicate == HAS_SYNONYM_TYPE]
                if len(synonym_types) > 0:
                    parent_pv.synonymType = synonym_type_code_from_curie(synonym_types[0])
                    if len(synonym_types) > 1:
                        logging.warning(
                            f"Ignoring multiple synonym types: {synonym_types} for {curie}"
                        )
            pvs = [
                obograph.BasicPropertyValue(pred=ann.predicate, val=ann.object)
                for ann in anns
                if ann.predicate != HAS_DBXREF
            ]
            if pvs:
                parent_pv.meta = obograph.Meta(basicPropertyValues=pvs)

        for row in rows:
            if row.value is not None:
                v = row.value
            elif row.object is not None:
                v = row.object
            else:
                continue
            pred = row.predicate
            if pred == omd_slots.label.curie:
                n.lbl = v
            elif pred == RDF_TYPE:
                if v == OWL_CLASS:
                    n.type = "CLASS"
                elif v in [OWL_NAMED_INDIVIDUAL]:
                    n.type = "INDIVIDUAL"
                else:
                    n.type = "PROPERTY"
            else:
                if include_metadata:
                    anns = self._axiom_annotations(curie, pred, row.object, row.value)
                else:
                    anns = []
                if pred == omd_slots.definition.curie:
                    meta.definition = obograph.DefinitionPropertyValue(val=v)
                    _anns_to_xrefs_and_meta(meta.definition, anns)
                elif pred in SYNONYM_PREDICATES:
                    # TODO: handle in a separate util
                    if pred.startswith("oio:"):
                        pred = pred.replace("IAO:", "IAO_")
                        scope_pred = pred.replace("oio:", "")
                    else:
                        scope_pred = "hasExactSynonym"
                    pv = obograph.SynonymPropertyValue(pred=scope_pred, val=v)
                    _anns_to_xrefs_and_meta(pv, anns)
                    meta.synonyms.append(pv)
                elif pred == HAS_DBXREF:
                    pv = obograph.XrefPropertyValue(val=v)
                    _anns_to_xrefs_and_meta(pv, anns)
                    meta.xrefs.append(pv)
                elif pred == IN_SUBSET:
                    meta.subsets.append(v)
                else:
                    pv = obograph.BasicPropertyValue(pred=pred, val=v)
                    _anns_to_xrefs_and_meta(pv, anns)
                    meta.basicPropertyValues.append(pv)
        return n

    def nodes(self, expand_curies=False) -> Iterator[Node]:
        """
        Yields all nodes in all graphs

        :param expand_curies:
        :return:
        """
        for e in self.entities():
            if not e.startswith("<"):
                n = self.node(e, include_metadata=True, expand_curies=expand_curies)
                if n.lbl:
                    yield n

    def synonym_property_values(
        self, subject: Union[CURIE, Iterable[CURIE]]
    ) -> Iterator[Tuple[CURIE, obograph.SynonymPropertyValue]]:
        if isinstance(subject, CURIE):
            subject = [subject]
        q = self.session.query(Statements).filter(Statements.subject.in_(tuple(subject)))
        q = q.filter(Statements.predicate.in_(SYNONYM_PREDICATES))
        for row in q:
            pred = row.predicate.replace("oio:", "")
            spv = obograph.SynonymPropertyValue(pred=pred, val=row.value)
            anns = self._axiom_annotations(row.subject, row.predicate, value=row.value)
            for ann in anns:
                if ann.predicate == HAS_SYNONYM_TYPE:
                    spv.synonymType = synonym_type_code_from_curie(ann.object)
                if ann.predicate == HAS_DBXREF:
                    spv.xrefs.append(ann.object)
            yield row.subject, spv

    def _axiom_annotations(
        self, subject: CURIE, predicate: CURIE, object: CURIE = None, value: Any = None
    ) -> List[om.Annotation]:
        """Get all axiom annotations for a single edge."""
        q = self.session.query(OwlAxiomAnnotation)
        q = q.filter(OwlAxiomAnnotation.subject == subject)
        q = q.filter(OwlAxiomAnnotation.predicate == predicate)
        if object:
            q = q.filter(OwlAxiomAnnotation.object == object)
        if value:
            q = q.filter(OwlAxiomAnnotation.value == value)
        return [
            om.Annotation(
                row.annotation_predicate,
                row.annotation_value if row.annotation_value is not None else row.annotation_object,
            )
            for row in q
        ]

    def _axiom_annotations_multi(
        self,
        subjects: List[CURIE],
        predicate: CURIE,
        objects: List[CURIE] = None,
        values: List[Any] = None,
    ) -> Iterator[OwlAxiomAnnotation]:
        """Get all axiom annotations from multiple edges sharing the same predicate."""
        q = self.session.query(OwlAxiomAnnotation)
        if subjects:
            q = q.filter(OwlAxiomAnnotation.subject.in_(subjects))
        q = q.filter(OwlAxiomAnnotation.predicate == predicate)
        if objects:
            q = q.filter(OwlAxiomAnnotation.object.in_(objects))
        if values:
            q = q.filter(OwlAxiomAnnotation.value.in_(values))
        for row in q:
            yield row

    def _axiom_annotations_all(self) -> List[OwlAxiomAnnotation]:
        """Get all axiom annotations."""
        return self.session.query(OwlAxiomAnnotation).all()

    def ancestors(
        self,
        start_curies: Union[CURIE, List[CURIE]],
        predicates: List[PRED_CURIE] = None,
        reflexive: bool = True,
        method: Optional[GraphTraversalMethod] = None,
    ) -> Iterable[CURIE]:
        if method and method == GraphTraversalMethod.HOP:
            if not isinstance(start_curies, list):
                start_curies = [start_curies]
            ancs = {o for _s, _p, o in walk_up(self, start_curies, predicates, include_abox=False)}
            if reflexive:
                ancs.update(start_curies)
            for o in ancs:
                yield o
            return
        q = self.session.query(EntailedEdge.object).distinct()
        if isinstance(start_curies, list):
            q = q.filter(EntailedEdge.subject.in_(tuple(start_curies)))
        else:
            q = q.filter(EntailedEdge.subject == start_curies)
        if predicates is not None:
            q = q.filter(EntailedEdge.predicate.in_(tuple(predicates)))
        if not reflexive:
            q = q.filter(EntailedEdge.subject != EntailedEdge.object)
        if reflexive:
            if isinstance(start_curies, list):
                yield from start_curies
            else:
                yield start_curies
        logging.debug(f"Ancestors query: {q}")
        for row in q:
            yield row.object

    def _multi_ancestors(
        self, start_curies: Union[CURIE, List[CURIE]], predicates: List[PRED_CURIE] = None
    ) -> Iterable[RELATIONSHIP]:
        q = self.session.query(EntailedEdge)
        if isinstance(start_curies, list):
            q = q.filter(EntailedEdge.subject.in_(tuple(start_curies)))
        else:
            q = q.filter(EntailedEdge.subject == start_curies)
            start_curies = list(start_curies)
        if predicates is not None:
            q = q.filter(EntailedEdge.predicate.in_(tuple(predicates)))
        logging.debug(f"Ancestors query, start from {start_curies}: {q}")
        for row in q:
            yield row.subject, row.predicate, row.object

    def descendants(
        self,
        start_curies: Union[CURIE, List[CURIE]],
        predicates: List[PRED_CURIE] = None,
        reflexive=True,
        method: Optional[GraphTraversalMethod] = None,
    ) -> Iterable[CURIE]:
        if method and method == GraphTraversalMethod.HOP:
            descs = {s for s, _p, _o in walk_down(self, start_curies, predicates)}
            for s in descs:
                yield s
            return
        q = self.session.query(EntailedEdge)
        if isinstance(start_curies, list):
            q = q.filter(EntailedEdge.object.in_(tuple(start_curies)))
        else:
            q = q.filter(EntailedEdge.object == start_curies)
        if predicates is not None:
            q = q.filter(EntailedEdge.predicate.in_(tuple(predicates)))
        if not reflexive:
            q = q.filter(EntailedEdge.subject != EntailedEdge.object)
        for row in q:
            yield row.subject

    def descendant_count(
        self,
        start_curies: Union[CURIE, List[CURIE]],
        predicates: List[PRED_CURIE] = None,
        reflexive=True,
    ) -> int:
        q = self.session.query(EntailedEdge.subject)
        if isinstance(start_curies, list):
            q = q.filter(EntailedEdge.object.in_(tuple(start_curies)))
        else:
            q = q.filter(EntailedEdge.object == start_curies)
        if predicates is not None:
            q = q.filter(EntailedEdge.predicate.in_(tuple(predicates)))
        if not reflexive:
            q = q.filter(EntailedEdge.subject != EntailedEdge.object)
        return q.count()

    def _rdf_list(self, bnode: str) -> Iterable[str]:
        for row in self.session.query(RdfFirstStatement).filter(RdfFirstStatement.subject == bnode):
            yield row.object
        for row in self.session.query(RdfRestStatement.object).filter(
            RdfRestStatement.subject == bnode
        ):
            for x in self._rdf_list(row.object):
                yield x

    def _ixn_definition(self, ixn: str, subject: CURIE) -> Optional[LogicalDefinitionAxiom]:
        ldef = LogicalDefinitionAxiom(definedClassId=subject)
        n = 0
        for ixn_node in self._rdf_list(ixn):
            n += 1
            if _is_blank(ixn_node):
                svfq = self.session.query(OwlSomeValuesFrom).filter(
                    OwlSomeValuesFrom.id == ixn_node
                )
                svfq = list(svfq)
                if svfq:
                    if len(svfq) > 1:
                        raise ValueError(f"Incorrect rdf structure for equiv axioms for {ixn_node}")
                    svf = svfq[0]
                    ldef.restrictions.append(
                        ExistentialRestrictionExpression(
                            propertyId=svf.on_property, fillerId=svf.filler
                        )
                    )
                else:
                    ldef = None
                    break
            else:
                ldef.genusIds.append(ixn_node)
        if n and ldef:
            return ldef

    def logical_definitions(
        self,
        subjects: Optional[Iterable[CURIE]] = None,
        predicates: Iterable[PRED_CURIE] = None,
        objects: Iterable[CURIE] = None,
        **kwargs,
    ) -> Iterable[LogicalDefinitionAxiom]:
        logging.info("Getting logical definitions")
        q = self.session.query(OwlEquivalentClassStatement)
        if predicates is not None:
            predicates = list(predicates)
        if objects is not None:
            objects = list(objects)
        if subjects is None:
            for ldef in self._logical_definitions_from_eq_query(q, predicates, objects):
                yield ldef
            return
        for curie_it in chunk(subjects, self.max_items_for_in_clause):
            logging.info(f"Getting logical definitions for {curie_it} from {subjects}")
            q = q.filter(OwlEquivalentClassStatement.subject.in_(tuple(curie_it)))
            for ldef in self._logical_definitions_from_eq_query(q, predicates, objects):
                yield ldef

    def _logical_definitions_from_eq_query(
        self,
        query,
        predicates: Iterable[PRED_CURIE] = None,
        objects: Iterable[CURIE] = None,
    ) -> Iterable[LogicalDefinitionAxiom]:
        for eq_row in query:
            ixn_q = self.session.query(Statements).filter(
                and_(
                    Statements.subject == eq_row.object,
                    Statements.predicate == "owl:intersectionOf",
                )
            )
            for ixn in ixn_q:
                ldef = self._ixn_definition(ixn.object, eq_row.subject)
                if ldef:
                    if not logical_definition_matches(ldef, predicates=predicates, objects=objects):
                        continue
                    yield ldef

    def _node_to_class_expression(
        self, node: str
    ) -> Optional[Union[CURIE, ExistentialRestrictionExpression]]:
        if not _is_blank(node):
            return node

        svfq = self.session.query(OwlSomeValuesFrom).filter(OwlSomeValuesFrom.id == node)
        svfq = list(svfq)
        if svfq:
            if len(svfq) > 1:
                raise ValueError(f"Incorrect rdf structure for equiv axioms for {node}")
            svf = svfq[0]
            return ExistentialRestrictionExpression(propertyId=svf.on_property, fillerId=svf.filler)
        else:
            return None

    def disjoint_class_expressions_axioms(
        self,
        subjects: Optional[Iterable[CURIE]] = None,
        predicates: Iterable[PRED_CURIE] = None,
        group=False,
        **kwargs,
    ) -> Iterable[DisjointClassExpressionsAxiom]:
        logging.info("Getting disjoint class expression axioms")
        q = self.session.query(OwlDisjointClassStatement)
        if predicates is not None:
            predicates = list(predicates)
        if subjects:
            subjects = list(subjects)
        axs = []
        for da in q:
            # blank
            sx = self._node_to_class_expression(da.subject)
            ox = self._node_to_class_expression(da.object)
            allx = [sx, ox]
            allx_named = [x for x in allx if isinstance(x, str)]
            allx_exprs = [x for x in allx if isinstance(x, ExistentialRestrictionExpression)]
            allx_fillers = [x.fillerId for x in allx_exprs]
            if subjects:
                if not any(x for x in allx_named if x in subjects) and not any(
                    x for x in allx_fillers if x in subjects
                ):
                    continue
            if predicates:
                if not any(x for x in allx_exprs if x.propertyId in predicates):
                    continue
            ax = DisjointClassExpressionsAxiom(
                classIds=allx_named,
                classExpressions=allx_exprs,
            )
            axs.append(ax)
        q = self.session.query(RdfTypeStatement.subject).filter(
            RdfTypeStatement.object == "owl:AllDisjointClasses"
        )
        for adc in q:
            class_ids = []
            class_exprs = []
            for (m,) in self.session.query(Statements.object).filter(
                and_(
                    Statements.subject == adc.subject,
                    Statements.predicate == "owl:members",
                )
            ):
                if isinstance(m, str):
                    class_ids.append(m)
                else:
                    class_exprs.append(self._node_to_class_expression(m))
            axs.append(
                DisjointClassExpressionsAxiom(
                    classIds=class_ids,
                    classExpressions=class_exprs,
                )
            )
        yield from axs

    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # Implements: RelationGraphInterface
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    def entailed_relationships_between(self, subject: CURIE, object: CURIE) -> Iterable[PRED_CURIE]:
        preds = []
        for row in (
            self.session.query(EntailedEdge.predicate)
            .filter(EntailedEdge.subject == subject)
            .filter(EntailedEdge.object == object)
        ):
            p = row.predicate
            if p not in preds:
                yield p
            preds.append(p)

    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # Implements: MappingsInterface
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    def sssom_mappings_by_source(self, subject_or_object_source: str = None) -> Iterable[Mapping]:
        predicates = tuple(ALL_MATCH_PREDICATES)
        base_query = self.session.query(Statements).filter(Statements.predicate.in_(predicates))
        for row in base_query:
            v = row.value if row.value is not None else row.object
            # TODO: this check is slow
            if URIorCURIE.is_valid(v):
                if row.subject.startswith("_:"):
                    continue
                mpg = Mapping(
                    subject_id=row.subject,
                    object_id=v,
                    predicate_id=row.predicate,
                    mapping_justification=SEMAPV.UnspecifiedMatching.value,
                )
                inject_mapping_sources(mpg)
                if subject_or_object_source:
                    # TODO: consider moving to query for efficiency
                    if (
                        mpg.subject_source != subject_or_object_source
                        and mpg.object_source != subject_or_object_source
                    ):
                        continue
                yield mpg
            else:
                if self.strict:
                    raise ValueError(f"not a CURIE: {v}")

    def sssom_mappings(
        self,
        curies: Optional[Union[CURIE, Iterable[CURIE]]] = None,
        source: Optional[str] = None,
        bidirectional=True,
    ) -> Iterator[Mapping]:
        if isinstance(curies, CURIE):
            curies = [curies]
        elif curies is not None:
            curies = list(curies)
        justification = str(SEMAPV.UnspecifiedMatching.value)
        predicates = tuple(ALL_MATCH_PREDICATES)
        base_query = self.session.query(Statements).filter(Statements.predicate.in_(predicates))

        def normalize(curie: Optional[CURIE]) -> Optional[CURIE]:
            if curie is None:
                return None
            if curie.startswith("<"):
                return curie[1:-1]
            return curie

        if curies is None:
            by_subject_query = base_query
        else:
            by_subject_query = base_query.filter(Statements.subject.in_(curies))
        for row in by_subject_query:
            try:
                mpg = Mapping(
                    subject_id=row.subject,
                    object_id=normalize(row.value if row.value is not None else row.object),
                    predicate_id=row.predicate,
                    mapping_justification=justification,
                )
            except ValueError as e:
                logging.error(f"Skipping {row}; ValueError: {e}")
                continue
            inject_mapping_sources(mpg)
            if source and mpg.subject_source != source and mpg.object_source != source:
                continue
            yield mpg
        if curies is None:
            # all mappings have been returned
            return
        if bidirectional:
            # xrefs are stored as literals
            for row in base_query.filter(Statements.value.in_(curies)):
                mpg = Mapping(
                    subject_id=row.subject,
                    object_id=row.value,
                    predicate_id=row.predicate,
                    mapping_justification=justification,
                )
                inject_mapping_sources(mpg)
                if source and mpg.subject_source != source and mpg.object_source != source:
                    continue
                yield mpg
            # skos mappings are stored as objects
            for row in base_query.filter(Statements.object.in_(curies)):
                mpg = Mapping(
                    subject_id=row.subject,
                    object_id=row.object,
                    predicate_id=row.predicate,
                    mapping_justification=justification,
                )
                inject_mapping_sources(mpg)
                if source and mpg.subject_source != source and mpg.object_source != source:
                    continue
                yield mpg

    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # Implements: ValidatorInterface
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    def validate(
        self, configuration: vdm.ValidationConfiguration = None
    ) -> Iterable[vdm.ValidationResult]:
        if configuration and configuration.schema_path:
            sv = SchemaView(configuration.schema_path)
            self._ontology_metadata_model = sv
        else:
            sv = self.ontology_metadata_model
        sv.materialize_patterns()
        for slot_name in sv.all_slots():
            for r in self._check_slot(slot_name):
                yield r
        for r in self._check_for_unknown_slots():
            yield r

    def _missing_value(
        self, predicate_table: Type, type_table: Type = ClassNode
    ) -> Iterable[CURIE]:
        pred_subq = self.session.query(predicate_table.subject)
        obs_subq = self.session.query(DeprecatedNode.id)
        main_q = self.session.query(type_table).join(IriNode, type_table.id == IriNode.id)
        for row in main_q.filter(type_table.id.not_in(pred_subq)).filter(
            type_table.id.not_in(obs_subq)
        ):
            yield row.id

    def term_curies_without_definitions(self) -> Iterable[CURIE]:
        return self._missing_value(HasTextDefinitionStatement)

    def term_curies_without_labels(self) -> Iterable[CURIE]:
        return self._missing_value(RdfsLabelStatement)

    def _check_for_unknown_slots(self) -> Iterable[vdm.ValidationResult]:
        sv = self.ontology_metadata_model
        preds = [sv.get_uri(s, expand=False) for s in sv.all_slots().values()]
        logging.info(f"Known preds: {len(preds)} -- checking for other uses")
        main_q = (
            self.session.query(Statements)
            .filter(Statements.predicate.not_in(preds))
            .join(IriNode, Statements.subject == IriNode.id)
        )
        try:
            for row in main_q:
                result = vdm.ValidationResult(
                    subject=row.subject,
                    predicate=row.predicate,
                    severity=vdm.SeverityOptions.ERROR,
                    type=vdm.ValidationResultType.ClosedConstraintComponent.meaning,
                    info=f"Unknown pred ({row.predicate}) = {row.object} {row.value}",
                )
                yield result
        except ValueError as e:
            logging.error(f"EXCEPTION: {e}")

    def _check_slot(
        self, slot_name: str, class_name: str = "Class"
    ) -> Iterable[vdm.ValidationResult]:
        """
        Validates all data with respect to a specific slot.

        :param slot_name:
        :param class_name:
        :return:
        """
        sv = self.ontology_metadata_model
        # for efficiency we map directly to table/view names rather
        # than querying over rdf:type; this allows for optimization via view materialization
        if class_name == "Class":
            sqla_cls = ClassNode
        elif class_name == "ObjectProperty":
            sqla_cls = ObjectPropertyNode
        elif class_name == "AnnotationProperty":
            sqla_cls = AnnotationPropertyNode
        elif class_name == "NamedIndividual":
            sqla_cls = NamedIndividualNode
        else:
            raise NotImplementedError(f"cannot handle {class_name}")
        slot = sv.induced_slot(slot_name, class_name)
        if slot.designates_type:
            logging.info(f"Ignoring type designator: {slot_name}")
            return
        logging.info(f"Validating: {slot_name}")
        predicate = sv.get_uri(slot, expand=False)
        is_used = (
            self.session.query(Statements.predicate)
            .filter(Statements.predicate == predicate)
            .first()
            is not None
        )
        pred_subq = self.session.query(Statements.subject).filter(Statements.predicate == predicate)
        obs_subq = self.session.query(DeprecatedNode.id)
        if (slot.required or slot.recommended) and not slot.identifier:
            # MinCardinality == 1
            if slot.required:
                severity = vdm.SeverityOptions.ERROR
            else:
                severity = vdm.SeverityOptions.WARNING
            logging.info(f"MinCard check: Leaving off: {slot_name} is {severity.text}")
            # exclude blank nodes
            main_q = self.session.query(sqla_cls).join(IriNode, sqla_cls.id == IriNode.id)
            main_q = main_q.filter(sqla_cls.id.not_in(pred_subq))
            main_q = main_q.filter(sqla_cls.id.not_in(obs_subq))
            for row in main_q:
                result = vdm.ValidationResult(
                    subject=row.id,
                    predicate=predicate,
                    severity=severity,
                    type=vdm.ValidationResultType.MinCountConstraintComponent.meaning,
                    info=f"Missing slot ({slot_name}) for {row.id}",
                )
                yield result
        if not is_used:
            return
        if slot.pattern:
            # check values against regexes
            # NOTE: this may be slow as we have to do this in
            # packages rather than SQL. Some SQL engines have regex support,
            # and we should leverage that when it exists
            re_pattern = re.compile(slot.pattern)
            main_q = self.session.query(Statements).filter(Statements.predicate == predicate)
            for row in main_q:
                val = row.value if row.value is not None else row.object
                if val is not None:
                    if not re_pattern.match(val):
                        result = vdm.ValidationResult(
                            subject=row.subject,
                            predicate=row.predicate,
                            severity=vdm.SeverityOptions.ERROR,
                            type=vdm.ValidationResultType.PatternConstraintComponent.meaning,
                            info=f"Pattern violation: {slot_name} = {val} does not conform to {slot.pattern}",
                        )
                        yield result
        if slot.deprecated:
            main_q = self.session.query(Statements.subject).filter(
                Statements.predicate == predicate
            )
            main_q = main_q.join(sqla_cls, Statements.subject == sqla_cls.id)
            for row in main_q:
                result = vdm.ValidationResult(
                    subject=row.subject,
                    predicate=predicate,
                    severity=vdm.SeverityOptions.WARNING,
                    type=vdm.ValidationResultType.DeprecatedPropertyComponent.meaning,
                    info=f"Deprecated slot ({slot_name}) for {row.subject}",
                )
                yield result
        if not slot.multivalued:
            # MaxCardinality == 1
            # TODO
            is_object_iri = slot.range in sv.all_classes()
            st1 = aliased(Statements)
            st2 = aliased(Statements)
            main_q = self.session.query(st1.subject).join(st2, st1.subject == st2.subject)
            main_q = main_q.filter(st1.predicate == predicate)
            main_q = main_q.filter(st2.predicate == predicate)
            if is_object_iri:
                main_q = main_q.filter(st1.object != st2.object)
            else:
                main_q = main_q.filter(st1.value != st2.value)
            main_q = main_q.join(sqla_cls, st1.subject == sqla_cls.id)
            for row in main_q:
                result = vdm.ValidationResult(
                    subject=row.subject,
                    predicate=predicate,
                    severity=vdm.SeverityOptions.ERROR,
                    type=vdm.ValidationResultType.MaxCountConstraintComponent.meaning,
                    info=f"Too many vals for {slot_name}",
                )
                yield result
        if slot.range:
            rng = slot.range
            rng_elements = sv.slot_applicable_range_elements(slot)
            # for now we don't handle Union or Any
            if len(rng_elements) < 2:
                logging.info(f"Datatype check: {slot_name} range is {rng_elements}")
                is_object_iri = rng in sv.all_classes()
                if is_object_iri:
                    constr = Statements.object.is_(None)
                else:
                    constr = Statements.value.is_(None)
                main_q = self.session.query(Statements.subject)
                main_q = main_q.join(IriNode, Statements.subject == IriNode.id)
                main_q = main_q.join(sqla_cls, Statements.subject == sqla_cls.id)
                main_q = main_q.filter(Statements.predicate == predicate, constr)
                for row in main_q:
                    result = vdm.ValidationResult(
                        subject=row.subject,
                        predicate=predicate,
                        severity=vdm.SeverityOptions.ERROR,
                        type=vdm.ValidationResultType.DatatypeConstraintComponent.meaning,
                        info=f"Incorrect object type for {slot_name} range = {rng} should_be_iri = {is_object_iri}",
                    )
                    yield result
                if rng in sv.all_types():
                    uri = get_range_xsd_type(sv, rng)
                    # uri = rng_type.uri
                    main_q = self.session.query(Statements.subject)
                    main_q = main_q.join(IriNode, Statements.subject == IriNode.id)
                    main_q = main_q.join(sqla_cls, Statements.subject == sqla_cls.id)
                    main_q = main_q.filter(
                        Statements.predicate == predicate, Statements.datatype != uri
                    )
                    for row in main_q:
                        result = vdm.ValidationResult(
                            subject=row.subject,
                            predicate=predicate,
                            severity=vdm.SeverityOptions.ERROR,
                            type=vdm.ValidationResultType.DatatypeConstraintComponent.meaning,
                            info=f"Incorrect datatype for {slot_name} expected: {uri} for {rng}",
                        )
                        yield result

    def gap_fill_relationships(
        self, seed_curies: List[CURIE], predicates: List[PRED_CURIE] = None
    ) -> Iterator[RELATIONSHIP]:
        seed_curies = tuple(seed_curies)
        logging.info(f"Gap fill for {len(seed_curies)} seed terms")
        q = self.session.query(EntailedEdge).filter(EntailedEdge.subject.in_(seed_curies))
        q = q.filter(EntailedEdge.object.in_(seed_curies))
        q = q.filter(EntailedEdge.subject != EntailedEdge.object)
        if predicates:
            q = q.filter(EntailedEdge.predicate.in_(tuple(predicates)))
        for row in q:
            if row.subject != row.object:
                e1 = aliased(EntailedEdge)
                e2 = aliased(EntailedEdge)
                q2 = self.session.query(e1, e2)
                q2 = q2.filter(e1.subject == row.subject)
                q2 = q2.filter(e1.object.in_(seed_curies))
                q2 = q2.filter(e1.object == e2.subject)
                q2 = q2.filter(e2.object == row.object)
                q2 = q2.filter(e1.subject != e1.object)
                q2 = q2.filter(e2.subject != e2.object)
                if predicates:
                    q2 = q2.filter(e1.predicate.in_(tuple(predicates)))
                    q2 = q2.filter(e2.predicate.in_(tuple(predicates)))
                redundant = False
                for e1row, e2row in q2:
                    if predicates is None:
                        redundant = True
                    else:
                        if e1row.predicate in predicates:
                            if e2row.predicate in predicates or e2row.predicate == IS_A:
                                redundant = True
                        elif e2row.predicate in predicates:
                            if e1row.predicate == IS_A:
                                redundant = True
                    if redundant:
                        break
                if not redundant:
                    yield row.subject, row.predicate, row.object

    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # Implements: OwlInterface
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    def disjoint_pairs(self, subjects: Iterable[CURIE] = None) -> Iterable[Tuple[CURIE, CURIE]]:
        q = self.session.query(Statements).filter(Statements.predicate == DISJOINT_WITH)
        if subjects:
            q = q.filter(
                or_(Statements.subject.in_(tuple(subjects)), Statements.object.in_(tuple(subjects)))
            )
        for row in q:
            if not row.subject.startswith("_") and not row.object.startswith("_"):
                yield row.subject, row.object

    def is_disjoint(self, subject: CURIE, object: CURIE, bidirectional=True) -> bool:
        q = self.session.query(Statements).filter(Statements.predicate == DISJOINT_WITH)
        ee1 = aliased(EntailedEdge)
        ee2 = aliased(EntailedEdge)
        q = q.filter(
            ee1.subject == subject, ee1.object == Statements.subject, ee1.predicate == IS_A
        )
        q = q.filter(ee2.subject == object, ee2.object == Statements.object, ee2.predicate == IS_A)
        if q.first():
            return True
        if bidirectional:
            return self.is_disjoint(object, subject, bidirectional=False)
        return False

    def transitive_object_properties(self) -> Iterable[CURIE]:
        for row in self.session.query(TransitivePropertyNode.id):
            yield row[0]

    def simple_subproperty_of_chains(self) -> Iterable[Tuple[CURIE, List[CURIE]]]:
        q = self.session.query(Statements)
        q = q.filter(Statements.predicate == OWL_PROPERTY_CHAIN_AXIOM)
        for row in q:
            chain = list(self._rdf_list(row.object))
            yield row.subject, chain

    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # Implements: SemSim
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    def information_content_scores(
        self,
        curies: Iterable[CURIE],
        predicates: List[PRED_CURIE] = None,
        object_closure_predicates: List[PRED_CURIE] = None,
        use_associations: bool = None,
        **kwargs,
    ) -> Iterator[Tuple[CURIE, float]]:
        curies = list(curies)
        if self.cached_information_content_map:
            yield from super().information_content_scores(curies)
            return
        if use_associations:
            logging.info("Using associations to calculate IC")
            if not self.can_store_associations:
                yield from super().information_content_scores(
                    curies,
                    predicates=predicates,
                    object_closure_predicates=object_closure_predicates,
                    use_associations=use_associations,
                    **kwargs,
                )
                return
                # raise ValueError("Cannot use associations, not stored")
            q = self.session.query(EntailedEdge.object, func.count(TermAssociation.subject))
            q = q.filter(EntailedEdge.subject == TermAssociation.object)
            if curies is not None:
                q = q.filter(EntailedEdge.object.in_(curies))
            if predicates:
                q = q.filter(TermAssociation.predicate.in_(predicates))
            if object_closure_predicates:
                q = q.filter(EntailedEdge.predicate.in_(object_closure_predicates))
            q = q.group_by(EntailedEdge.object)
            logging.info(f"QUERY: {q}")
            num_nodes = (
                self.session.query(TermAssociation).distinct(TermAssociation.subject).count()
            )
        else:
            num_nodes = (
                self.session.query(EntailedEdge.subject).distinct(EntailedEdge.subject).count()
            )
            logging.info(f"Number of nodes in background set={num_nodes}")
            q = self.session.query(EntailedEdge.object, func.count(distinct(EntailedEdge.subject)))
            if predicates:
                raise ValueError(
                    "predicates not valid unless use_associations=True"
                    "did you mean object_closure_predicates?"
                )
            if object_closure_predicates:
                q = q.filter(EntailedEdge.predicate.in_(object_closure_predicates))
            if curies is not None:
                q = q.filter(EntailedEdge.object.in_(curies))
            q = q.group_by(EntailedEdge.object)
        yielded_owl_thing = False
        for row in q:
            curie, freq = row
            yield curie, -math.log(freq / num_nodes) / math.log(2)
            if curie == OWL_THING:
                yielded_owl_thing = True
        # inject owl:Thing, which always has zero information
        if (OWL_THING in curies or not curies) and not yielded_owl_thing:
            yield OWL_THING, 0.0

    def all_by_all_pairwise_similarity(
        self,
        subjects: Iterable[CURIE],
        objects: Iterable[CURIE],
        predicates: List[PRED_CURIE] = None,
        min_jaccard_similarity: Optional[float] = None,
        min_ancestor_information_content: Optional[float] = None,
    ) -> Iterator[TermPairwiseSimilarity]:
        def tuples_to_map(
            entities: List[CURIE], relationships: Iterable[RELATIONSHIP]
        ) -> Dict[CURIE, Set[CURIE]]:
            rmap = defaultdict(set)
            for r in relationships:
                rmap[r[0]].add(r[2])
            for e in entities:
                rmap[e].add(e)
            return rmap

        subjects = list(subjects)
        objects = list(objects)
        subjects_ancs = tuples_to_map(
            subjects, self._multi_ancestors(subjects, predicates=predicates)
        )
        objects_ancs = tuples_to_map(objects, self._multi_ancestors(objects, predicates=predicates))
        logging.info(f"SUBJECT ANCS={len(subjects_ancs)}")
        logging.info(f"OBJECT ANCS={len(objects_ancs)}")
        for s, s_ancs in subjects_ancs.items():
            for o, o_ancs in objects_ancs.items():
                logging.info(f"s={s} o={o}")
                sim = self.pairwise_similarity(
                    s,
                    o,
                    predicates=predicates,
                    subject_ancestors=list(s_ancs),
                    object_ancestors=list(o_ancs),
                    min_jaccard_similarity=min_jaccard_similarity,
                    min_ancestor_information_content=min_ancestor_information_content,
                )
                if sim:
                    yield sim

    def common_descendants(
        self,
        subject: CURIE,
        object: CURIE,
        predicates: List[PRED_CURIE] = None,
        include_owl_nothing: bool = False,
    ) -> Iterable[CURIE]:
        ee1 = aliased(EntailedEdge)
        ee2 = aliased(EntailedEdge)
        q = self.session.query(ee1.subject)
        q = q.filter(ee1.object == subject)
        q = q.filter(ee2.object == object)
        q = q.filter(ee1.subject == ee2.subject)
        if predicates:
            q = q.filter(ee1.predicate.in_(predicates))
            q = q.filter(ee2.predicate.in_(predicates))
        for row in q:
            if include_owl_nothing or row.subject != OWL_NOTHING:
                yield row.subject

    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # Implements: PatcherInterface
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    def migrate_curies(self, curie_map: Dict[CURIE, CURIE]) -> None:
        # TODO: add an operation for this to KGCL
        for k, v in curie_map.items():
            for cls in [Statements, EntailedEdge]:
                cmd = update(cls).where(cls.subject == k).values(subject=v)
                self.session.execute(cmd)
                cmd = update(cls).where(cls.predicate == k).values(predicate=v)
                self.session.execute(cmd)
                cmd = update(cls).where(cls.object == k).values(object=v)
                self.session.execute(cmd)
        if self.autosave:
            self.save()

    def _set_predicate_value(
        self, subject: CURIE, predicate: PRED_CURIE, value: str, datatype: str
    ):
        stmt = delete(Statements).where(
            and_(Statements.subject == subject, Statements.predicate == predicate)
        )
        self._execute(stmt)
        stmt = insert(Statements).values(
            subject=subject, predicate=predicate, value=value, datatype=datatype
        )
        self._execute(stmt)

    def apply_patch(
        self,
        patch: kgcl.Change,
        activity: kgcl.Activity = None,
        metadata: typing.Mapping[PRED_CURIE, Any] = None,
        configuration: kgcl.Configuration = None,
        strict=False,
    ) -> Optional[kgcl.Change]:
        if isinstance(patch, kgcl.NodeChange):
            about = patch.about_node
            if isinstance(patch, kgcl.NodeRename):
                self.set_label(patch.about_node, patch.new_value)
            elif isinstance(patch, kgcl.NewSynonym):
                # TODO: synonym type
                if not patch.predicate:
                    patch.predicate = HAS_EXACT_SYNONYM
                self._execute(
                    insert(Statements).values(
                        subject=about, predicate=patch.predicate, value=patch.new_value
                    )
                )
            elif isinstance(patch, kgcl.RemoveSynonym):
                q = self.session.query(Statements).filter(
                    Statements.subject == about, Statements.value == patch.old_value
                )
                self._execute(
                    delete(Statements).where(
                        and_(
                            Statements.subject == about,
                            Statements.predicate.in_(SYNONYM_PREDICATES),
                            Statements.value == patch.old_value,
                        )
                    )
                )
            elif isinstance(patch, kgcl.AddNodeToSubset):
                # TODO: implement deterministic subset mapping
                pfx = about.split(":")[0].lower()
                subset_curie = f"obo:{pfx}#{patch.in_subset}"
                self._execute(
                    insert(Statements).values(
                        subject=about, predicate=IN_SUBSET, object=subset_curie
                    )
                )
            elif isinstance(patch, kgcl.RemoveNodeFromSubset):
                # TODO: implement deterministic subset mapping
                pfx = about.split(":")[0].lower()
                subset_curie = f"obo:{pfx}#{patch.in_subset}"
                self._execute(
                    delete(Statements).where(
                        and_(
                            Statements.subject == about,
                            Statements.predicate == IN_SUBSET,
                            Statements.object == subset_curie,
                        )
                    )
                )
            elif isinstance(patch, kgcl.NodeObsoletion):
                self.check_node_exists(about)
                self._set_predicate_value(
                    about, DEPRECATED_PREDICATE, value="true", datatype="xsd:string"
                )
                if isinstance(patch, kgcl.NodeObsoletionWithDirectReplacement):
                    # TODO: allow switching between value and object
                    self._set_predicate_value(
                        about,
                        TERM_REPLACED_BY,
                        value=patch.has_direct_replacement,
                        datatype="xsd:string",
                    )
            elif isinstance(patch, kgcl.NodeDeletion):
                self._execute(delete(Statements).where(Statements.subject == about))
            elif isinstance(patch, kgcl.NameBecomesSynonym):
                label = self.label(about)
                self.apply_patch(
                    kgcl.NodeRename(id=f"{patch.id}-1", about_node=about, new_value=patch.new_value)
                )
                self.apply_patch(
                    kgcl.NewSynonym(id=f"{patch.id}-2", about_node=about, new_value=label)
                )
            elif isinstance(patch, kgcl.SynonymReplacement):
                q = self.session.query(Statements).filter(
                    Statements.subject == about, Statements.value == patch.old_value
                )
                predicate = None
                for row in q:
                    if predicate is None:
                        predicate = row.predicate
                    else:
                        if predicate != row.predicate:
                            if predicate in SYNONYM_PREDICATES:
                                predicate = row.predicate
                            else:
                                raise ValueError(
                                    f"Multiple predicates for synonym: {about} "
                                    + f"syn: {patch.old_value} preds={predicate}, {row.predicate}"
                                )
                logging.debug(f"replacing synonym with predicate: {predicate}")
                self._execute(
                    delete(Statements).where(
                        and_(
                            Statements.subject == about,
                            Statements.predicate == predicate,
                            Statements.value == patch.old_value,
                        )
                    )
                )
                self._execute(
                    insert(Statements).values(
                        subject=about, predicate=predicate, value=patch.new_value
                    )
                )
            elif isinstance(patch, kgcl.NewTextDefinition):
                q = self.session.query(Statements).filter(
                    Statements.subject == about, Statements.predicate == HAS_DEFINITION_CURIE
                )
                if q.count() == 0:
                    self._execute(
                        insert(Statements).values(
                            subject=about, predicate=HAS_DEFINITION_CURIE, value=patch.new_value
                        )
                    )
                else:
                    self.apply_patch(
                        kgcl.NodeTextDefinitionChange(subject=about, value=patch.new_value)
                    )
            elif isinstance(patch, kgcl.NodeTextDefinitionChange):
                # TODO: if patch.new_value is provided, check.
                stmt = (
                    update(Statements)
                    .where(
                        and_(
                            Statements.subject == about,
                            Statements.predicate == HAS_DEFINITION_CURIE,
                        )
                    )
                    .values(value=patch.new_value)
                )
                self._execute(stmt)
            else:
                raise NotImplementedError(f"Unknown patch type: {type(patch)}")
        elif isinstance(patch, kgcl.EdgeChange):
            about = patch.about_edge
            if isinstance(patch, kgcl.EdgeCreation):
                self._execute(
                    insert(Statements).values(
                        subject=patch.subject, predicate=patch.predicate, object=patch.object
                    )
                )
                self._rebuild_relationship_index()
                logging.warning("entailed_edge is now stale")
            elif isinstance(patch, kgcl.EdgeDeletion):
                self._execute(
                    delete(Statements).where(
                        and_(
                            Statements.subject == patch.subject,
                            Statements.predicate == patch.predicate,
                            Statements.object == patch.object,
                        )
                    )
                )
                logging.warning("entailed_edge is now stale")
            elif isinstance(patch, kgcl.NodeMove):
                raise NotImplementedError
                # self._execute(delete(Statements).where(and_(Statements.subject==patch.subject,
                #                                            Statements.predicate==patch.predicate,
                #                                            Statements.object==patch.object)))
                logging.warning("entailed_edge is now stale")
            else:
                raise NotImplementedError(f"Cannot handle patches of type {type(patch)}")
        else:
            raise NotImplementedError
        if self.autosave:
            self.save()
        return patch

    def save(
        self,
    ):
        logging.info("Committing and flushing changes")
        self.session.commit()
        self.session.flush()

    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # Implements: MetadataInterface
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    def statements_with_annotations(self, curie: CURIE) -> Iterable[om.Axiom]:
        m = self.entity_metadata_map(curie)
        q = self.session.query(OwlAxiomAnnotation)
        q = q.filter(OwlAxiomAnnotation.subject == curie)
        axiom_by_id = {}
        visited = {}
        for row in q:
            if row.value is not None:
                v = row.value
            elif row.object is not None:
                v = row.object
            else:
                raise ValueError(f"Unexpected null object/value in {row}")
            axiom_id = row.id
            if axiom_id in axiom_by_id:
                ax = axiom_by_id[axiom_id]
            else:
                ax = om.Axiom(
                    annotatedSource=curie, annotatedProperty=row.predicate, annotatedTarget=v
                )
                axiom_by_id[axiom_id] = ax
            v = row.annotation_object
            if v is None:
                v = row.annotation_value
            ax.annotations.append(om.Annotation(predicate=row.annotation_predicate, object=v))
        for ax in axiom_by_id.values():
            visited[(ax.annotatedSource, ax.annotatedProperty, ax.annotatedTarget)] = True
            yield ax
        for k, vs in m.items():
            if not isinstance(vs, list):
                vs = [vs]
            for v in vs:
                if (curie, k, v) not in visited:
                    if not _is_quoted_url(curie):
                        ax = om.Axiom(annotatedSource=curie, annotatedProperty=k, annotatedTarget=v)
                        yield ax

    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # Implements: DifferInterface
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    def compare_term_in_two_ontologies(
        self, other_ontology: BasicOntologyInterface, curie: CURIE, other_curie: CURIE = None
    ) -> Any:
        if other_curie is None:
            other_curie = curie
        logging.info(f"Comparing {curie} with {other_curie}")
        if isinstance(other_ontology, SqlImplementation):

            def nullify_subject(row):
                return f"{row.predicate} {row.object} {row.value} {row.datatype} {row.language}"

            this_rows = [
                nullify_subject(row)
                for row in self.session.query(Statements).filter(Statements.subject == curie)
            ]
            other_rows = [
                nullify_subject(row)
                for row in other_ontology.session.query(Statements).filter(
                    Statements.subject == other_curie
                )
            ]
            this_only = set(this_rows).difference(set(other_rows))
            other_only = set(other_rows).difference(set(this_rows))
            return this_only, other_only
        else:
            raise NotImplementedError(
                f"other ontology {other_ontology} must implement SqlInterface"
            )

    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # Implements: SummaryStatisticsInterface
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    def branch_summary_statistics(
        self,
        branch_name: str = None,
        branch_roots: List[CURIE] = None,
        property_values: Dict[CURIE, Any] = None,
        include_entailed=False,
        parent: GroupedStatistics = None,
        prefixes: List[CURIE] = None,
    ) -> UngroupedStatistics:
        session = self.session
        not_in = False
        if branch_name is None:
            branch_name = "AllOntologies"
        if branch_roots is not None:
            branch_subq = session.query(EntailedEdge.subject)
            branch_subq = branch_subq.filter(EntailedEdge.predicate == IS_A)
            branch_subq = branch_subq.filter(EntailedEdge.object.in_(branch_roots))
        elif property_values is not None:
            logging.info(f"Filtering by {property_values}")
            if len(property_values) > 1:
                raise NotImplementedError("Only one property value is supported at this time")
            k, v = list(property_values.items())[0]
            if k == PREFIX_PREDICATE:
                self._check_has_view(NodeIdentifier, "0.2.5")
                branch_subq = session.query(NodeIdentifier.id)
                branch_subq = branch_subq.filter(NodeIdentifier.prefix == v)
            else:
                branch_subq = session.query(Statements.subject)
                branch_subq = branch_subq.filter(Statements.predicate == k)
                if v is None:
                    not_in = True
                elif isinstance(v, list):
                    branch_subq = branch_subq.filter(Statements.value.in_(v))
                else:
                    branch_subq = branch_subq.filter(Statements.value == v)
        else:
            branch_subq = None
            if prefixes:
                self._check_has_view(NodeIdentifier, "0.2.5")
                branch_subq = session.query(NodeIdentifier.id)
                branch_subq = branch_subq.filter(NodeIdentifier.prefix.in_(tuple(prefixes)))

        def q(x):
            return session.query(x)

        def _filter(select_expr, filter_expr=None):
            if not filter_expr:
                filter_expr = select_expr
            q = session.query(select_expr)
            if branch_subq is not None:
                if not_in:
                    q = q.filter(filter_expr.not_in(branch_subq))
                else:
                    q = q.filter(filter_expr.in_(branch_subq))
            if False and prefixes:
                if len(prefixes) == 1:
                    q = q.filter(filter_expr.startswith(prefixes[0]))
                else:
                    self._check_has_view(NodeIdentifier, "0.2.5")
                    prefix_subq = session.query(NodeIdentifier.id)
                    prefix_subq = prefix_subq.filter(NodeIdentifier.prefix.in_(tuple(prefixes)))
                    q = q.filter(filter_expr.in_(prefix_subq))
            return q

        ssc = UngroupedStatistics(branch_name)
        if not parent:
            self._add_statistics_metadata(ssc)
        obsoletion_subq = q(DeprecatedNode.id)
        text_defn_subq = q(HasTextDefinitionStatement.subject)
        logging.debug(f"Subqueries: {obsoletion_subq} {text_defn_subq}")
        logging.debug("Getting basic counts")
        ssc.class_count = _filter(ClassNode.id).distinct(ClassNode.id).count()
        ssc.named_individual_count = _filter(NamedIndividualNode.id).distinct(ClassNode.id).count()
        depr_class_query = _filter(ClassNode.id).filter(ClassNode.id.in_(obsoletion_subq))
        ssc.deprecated_class_count = depr_class_query.count()
        logging.debug(f"Calculated basic counts. Classes: {ssc.class_count}")
        merged_class_query = (
            _filter(Statements.subject)
            .filter(Statements.predicate == HAS_OBSOLESCENCE_REASON)
            .filter(Statements.object == TERMS_MERGED)
        )
        ssc.merged_class_query = merged_class_query.count()
        logging.debug(f"Calculated merged counts. Classes: {ssc.merged_class_count}")
        ssc.class_count_with_text_definitions = (
            _filter(ClassNode.id).filter(ClassNode.id.in_(text_defn_subq)).count()
        )
        ssc.object_property_count = _filter(ObjectPropertyNode.id).count()
        ssc.annotation_property_count = _filter(AnnotationPropertyNode.id).count()
        ssc.deprecated_property_count = (
            _filter(ObjectPropertyNode.id)
            .filter(ObjectPropertyNode.id.in_(obsoletion_subq))
            .count()
        )
        logging.debug(f"Calculated basic property counts. OPs: {ssc.object_property_count}")
        ssc.rdf_triple_count = _filter(Statements.subject).count()
        ssc.equivalent_classes_axiom_count = _filter(OwlEquivalentClassStatement.subject).count()
        ssc.subclass_of_axiom_count = _filter(RdfsSubclassOfStatement.subject).count()
        logging.debug(f"Calculated basic axiom counts. SCAs: {ssc.subclass_of_axiom_count}")
        subset_query = _filter(Statements.object, Statements.subject).filter(
            Statements.predicate == IN_SUBSET
        )
        ssc.subset_count = subset_query.distinct(Statements.object).count()
        subset_agg_query = session.query(
            Statements.object, func.count(Statements.subject.distinct())
        )
        subset_agg_query = subset_agg_query.filter(Statements.predicate == IN_SUBSET)
        if branch_subq:
            subset_agg_query = subset_agg_query.filter(Statements.subject.in_(branch_subq))
        for row in subset_agg_query.group_by(Statements.object):
            subset = row.object
            if subset is None:
                logging.warning("Skipping subsets modeled as strings")
                continue
            ssc.class_count_by_subset[subset] = FacetedCount(row[0], filtered_count=row[1])
            logging.debug(f"Agg count for subset {subset}: {ssc.class_count_by_subset[subset]}")
        synonym_query = q(Statements.value).filter(Statements.predicate.in_(SYNONYM_PREDICATES))
        if branch_subq:
            synonym_query = synonym_query.filter(Statements.subject.in_(branch_subq))
        ssc.synonym_statement_count = synonym_query.count()
        ssc.distinct_synonym_count = synonym_query.distinct().count()
        logging.debug(f"Calculated basic synonym counts. Statements: {ssc.synonym_statement_count}")
        synonym_agg_query = session.query(
            Statements.predicate, func.count(Statements.value.distinct())
        )
        synonym_agg_query = synonym_agg_query.filter(Statements.predicate.in_(SYNONYM_PREDICATES))
        if branch_subq:
            synonym_agg_query = synonym_agg_query.filter(Statements.subject.in_(branch_subq))
        for row in synonym_agg_query.group_by(Statements.predicate):
            pred = row.predicate
            ssc.synonym_statement_count_by_predicate[pred] = FacetedCount(
                row[0], filtered_count=row[1]
            )
            logging.debug(f"Agg count for synonym {pred}: {row}")
        edge_agg_query = session.query(Edge.predicate, func.count(Edge.subject))
        if branch_subq:
            edge_agg_query = edge_agg_query.filter(Edge.subject.in_(branch_subq))
        for row in edge_agg_query.group_by(Edge.predicate):
            ssc.edge_count_by_predicate[row.predicate] = FacetedCount(row[0], filtered_count=row[1])
        if include_entailed:
            logging.debug("Calculating entailed counts")
            entailed_edge_agg_query = session.query(
                EntailedEdge.predicate, func.count(Edge.subject)
            )
            if branch_subq:
                entailed_edge_agg_query = entailed_edge_agg_query.filter(
                    Edge.subject.in_(branch_subq)
                )
            for row in entailed_edge_agg_query.group_by(EntailedEdge.predicate):
                ssc.entailed_edge_count_by_predicate[row.predicate] = FacetedCount(
                    row[0], filtered_count=row[1]
                )
                logging.debug(f"Agg count for entailed edge {row}")
        match_agg_query = session.query(
            Statements.predicate, func.count(Statements.value.distinct())
        ).filter(Statements.predicate.in_(ALL_MATCH_PREDICATES))
        if branch_subq:
            match_agg_query = match_agg_query.filter(Statements.subject.in_(branch_subq))
        for row in match_agg_query.group_by(Statements.predicate):
            ssc.mapping_statement_count_by_predicate[row.predicate] = FacetedCount(
                row[0], filtered_count=row[1]
            )
            logging.debug(f"Agg count for mapping {row}")
        # non-aggregate query for matches
        match_query = session.query(Statements).filter(
            Statements.predicate.in_(ALL_MATCH_PREDICATES)
        )
        if branch_subq:
            match_query = match_query.filter(Statements.subject.in_(branch_subq))
        subject_ids_by_object_source = defaultdict(list)
        bad_ids = set()
        for row in match_query:
            subject_id = row.subject
            object_id = row.value if row.value else row.object
            if ":" not in object_id:
                if object_id not in bad_ids:
                    logging.warning(f"bad mapping: {object_id}")
                    bad_ids.add(object_id)
            object_source = object_id.split(":")[0]
            subject_ids_by_object_source[object_source].append(subject_id)
        for object_source, subject_ids in subject_ids_by_object_source.items():
            ssc.mapping_statement_count_by_object_source[object_source] = FacetedCount(
                object_source, filtered_count=len(subject_ids)
            )
            ssc.mapping_statement_count_subject_by_object_source[object_source] = FacetedCount(
                object_source, filtered_count=len(set(subject_ids))
            )
        logging.debug("Calculating contributor stats")
        contributor_agg_query = session.query(
            Statements.predicate,
            Statements.object,
            Statements.value,
            func.count(Statements.subject.distinct()),
        )
        contributor_agg_query = contributor_agg_query.filter(
            Statements.predicate.in_(ALL_CONTRIBUTOR_PREDICATES)
        )
        if branch_subq:
            contributor_agg_query = contributor_agg_query.filter(
                Statements.subject.in_(branch_subq)
            )
        for row in contributor_agg_query.group_by(
            Statements.predicate, Statements.object, Statements.value
        ):
            if row.object:
                contributor_id = row.object
                contributor_name = None
            else:
                contributor_id = row.value
                contributor_name = row.value
                if " " in contributor_id or ":" not in contributor_id:
                    logging.debug(
                        f"Ad-hoc repair of literal value for contributor: {contributor_id}"
                    )
                    contributor_id = string_as_base64_curie(contributor_id)
            contributor_id = _remove_uri_quotes(contributor_id)
            if contributor_id not in ssc.contributor_summary:
                ssc.contributor_summary[contributor_id] = ContributorStatistics(
                    contributor_id=contributor_id, contributor_name=contributor_name
                )
            ssc.contributor_summary[contributor_id].role_counts[row.predicate] = FacetedCount(
                row.predicate, row[-1]
            )
            logging.debug(f"Agg count for contributor {row}")
        # TODO: axiom contributor stats
        self._add_derived_statistics(ssc)
        return ssc

    def metadata_property_summary_statistics(self, metadata_property: PRED_CURIE) -> Dict[Any, int]:
        if metadata_property == PREFIX_PREDICATE:
            d = defaultdict(int)
            for e in self.entities(filter_obsoletes=False):
                prefix = e.split(":")[0]
                d[prefix] += 1
            return dict(d)
        session = self.session
        q = session.query(Statements.value, func.count(Statements.value))
        q = q.filter(Statements.predicate == metadata_property)
        q = q.group_by(Statements.value)
        return dict(q)
