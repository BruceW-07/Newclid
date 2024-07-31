from __future__ import annotations
from typing import TYPE_CHECKING, Any

from geosolver.dependency.symbols import Point
from geosolver.numerical import close_enough
from geosolver.predicates.equal_angles import EqAngle
from geosolver.predicates.predicate import Predicate
from geosolver.reasoning_engines.algebraic_reasoning.tables import Ratio_Chase
from geosolver.tools import reshape
from geosolver.dependency.dependency import Dependency


if TYPE_CHECKING:
    from geosolver.reasoning_engines.algebraic_reasoning.tables import Table
    from geosolver.reasoning_engines.algebraic_reasoning.tables import SumCV
    from geosolver.statement import Statement
    from geosolver.dependency.dependency_graph import DependencyGraph


class EqRatio(Predicate):
    """eqratio AB CD EF GH -

    Represent that AB/CD=EF/GH, as ratios between lengths of segments.
    """

    NAME = "eqratio"

    @classmethod
    def preparse(cls, args: tuple[str, ...]):
        return EqAngle.preparse(args)

    @classmethod
    def parse(cls, args: tuple[str, ...], dep_graph: DependencyGraph):
        return EqAngle.parse(args, dep_graph)

    @classmethod
    def check_numerical(cls, statement: Statement) -> bool:
        ratio = None
        for a, b, c, d in reshape(statement.args, 4):
            a: Point
            b: Point
            c: Point
            d: Point
            _ratio = a.num.distance(b.num) / c.num.distance(d.num)
            if ratio is not None and not close_enough(ratio, _ratio):
                return False
            ratio = _ratio
        return True

    @classmethod
    def _prep_ar(cls, statement: Statement) -> tuple[list[SumCV], Table]:
        points: tuple[Point, ...] = statement.args
        table = statement.dep_graph.ar.rtable
        eqs: list[SumCV] = []
        i = 4
        while i < len(points):
            eqs.append(
                table.get_eq4(
                    table.get_length(points[0], points[1]),
                    table.get_length(points[2], points[3]),
                    table.get_length(points[i], points[i + 1]),
                    table.get_length(points[i + 2], points[i + 3]),
                )
            )
            i += 4
        return eqs, table

    @classmethod
    def add(cls, dep: Dependency) -> None:
        eqs, table = cls._prep_ar(dep.statement)
        for eq in eqs:
            table.add_expr(eq, dep)

    @classmethod
    def why(cls, statement: Statement) -> Dependency:
        eqs, table = cls._prep_ar(statement)
        why: list[Dependency] = []
        for eq in eqs:
            why.extend(table.why(eq))
        if len(why) == 1:
            return why[0].with_new(statement)
        return Dependency.mk(
            statement, Ratio_Chase, tuple(dep.statement for dep in why)
        )

    @classmethod
    def check(cls, statement: Statement) -> bool:
        eqs, table = cls._prep_ar(statement)
        return all(table.expr_delta(eq) for eq in eqs)

    @classmethod
    def to_tokens(cls, args: tuple[Any, ...]) -> tuple[str, ...]:
        return tuple(p.name for p in args)

    @classmethod
    def pretty(cls, statement: Statement) -> str:
        args: tuple[Point, ...] = statement.args
        return " = ".join(
            f"{a.pretty_name}{b.pretty_name}:{c.pretty_name}{d.pretty_name}"
            for a, b, c, d in reshape(args, 4)
        )


class EqRatio3(Predicate):
    """eqratio AB CD MN -

    Represent three eqratios through a list of 6 points (due to parallel lines).
    It can be viewed as in an instance of Thales theorem which has AB // MN // CD.

    It thus represent the corresponding eqratios:
    MA / MC = NB / ND and AM / AC = BN / BD and MC / AC = ND / BD

    ::

          a -- b
         m ---- n
        c ------ d


    """

    NAME = "eqratio3"

    @classmethod
    def preparse(cls, args: tuple[str, ...]):
        a, b, c, d, m, n = args
        if len(set((a, c, m))) < 3 or len(set((b, d, n))) < 3:
            return None
        groups = ((a, b), (c, d), (m, n))
        groups1 = ((b, a), (d, c), (n, m))
        return sum(min(sorted(groups), sorted(groups1)), ())

    @classmethod
    def parse(cls, args: tuple[str, ...], dep_graph: DependencyGraph):
        preparse = cls.preparse(args)
        return (
            tuple(dep_graph.symbols_graph.names2points(preparse)) if preparse else None
        )

    @classmethod
    def check_numerical(cls, statement: Statement) -> bool:
        a, b, c, d, m, n = statement.args
        eqr1 = statement.with_new(EqRatio, (m, a, m, c, n, b, n, d))
        eqr2 = statement.with_new(EqRatio, (m, a, a, c, b, n, b, d))
        eqr3 = statement.with_new(EqRatio, (m, c, a, c, n, d, b, d))
        return (
            eqr1.check_numerical() and eqr2.check_numerical() and eqr3.check_numerical()
        )

    @classmethod
    def check(cls, statement: Statement) -> bool:
        a, b, c, d, m, n = statement.args
        eqr1 = statement.with_new(EqRatio, (m, a, m, c, n, b, n, d))
        eqr2 = statement.with_new(EqRatio, (m, a, a, c, b, n, b, d))
        eqr3 = statement.with_new(EqRatio, (m, c, a, c, n, d, b, d))
        return eqr1.check() and eqr2.check() and eqr3.check()

    @classmethod
    def add(cls, dep: Dependency):
        statement = dep.statement
        a, b, c, d, m, n = statement.args
        eqr1 = statement.with_new(EqRatio, (m, a, m, c, n, b, n, d))
        eqr2 = statement.with_new(EqRatio, (m, a, a, c, b, n, b, d))
        eqr3 = statement.with_new(EqRatio, (m, c, a, c, n, d, b, d))
        dep.with_new(eqr1).add()
        dep.with_new(eqr2).add()
        dep.with_new(eqr3).add()
