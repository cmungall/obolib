"""An in-memory sqlite index for simple associations."""
import logging
import sqlite3
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator

from semsql.sqla.semsql import TermAssociation
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from oaklib.interfaces.association_provider_interface import ASSOCIATION
from oaklib.types import CURIE, PRED_CURIE, ASSOCIATION

COLS = ["id", "subject", "predicate", "object", "evidence_type", "publication", "source"]


@dataclass
class AssociationIndex:

    _connection: sqlite3.Connection = None
    _session: Session = None
    _engine: Engine = None

    def create(self):
        connection_string: str = "sqlite:///:memory:"
        engine = create_engine(connection_string)
        con = engine.raw_connection()
        # con = sqlite3.connect(":memory:")
        con.execute(f"create table term_association({','.join(COLS)})")
        self._connection = con
        session_cls = sessionmaker(engine)
        self._session = session_cls()
        self._engine = engine

    def populate(self, associations: Iterable[ASSOCIATION]):
        associations = [(s, p, o) for s, p, o, _ in associations]
        self._connection.executemany(
            "insert into term_association(subject, predicate, object) values (?,?,?)", associations
        )

    def lookup(
        self,
        subjects: Iterable[CURIE] = None,
        predicates: Iterable[PRED_CURIE] = None,
        objects: Iterable[CURIE] = None,
        property_filter: Dict[PRED_CURIE, Any] = None,
    ) -> Iterator[ASSOCIATION]:
        session = self._session
        q = session.query(TermAssociation)
        if property_filter:
            raise NotImplementedError
        if subjects:
            q = q.filter(TermAssociation.subject.in_(tuple(subjects)))
        if predicates:
            q = q.filter(TermAssociation.predicate.in_(tuple(predicates)))
        if objects:
            q = q.filter(TermAssociation.object.in_(tuple(objects)))
        logging.info(f"Association query: {q}")
        for row in q:
            yield row.subject, row.predicate, row.object, []
