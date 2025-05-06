"""Implements theorem matching functions for the Deductive Database (DD)."""

import itertools
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Generator, Optional
import json

from newclid.formulations.clause import translate_sentence
from newclid.dependencies.symbols import Point
from newclid.statement import Statement
from newclid.dependencies.dependency import Dependency

if TYPE_CHECKING:
    import numpy as np
    from newclid.formulations.rule import Rule
    from newclid.dependencies.dependency_graph import DependencyGraph


class Matcher:
    def __init__(
        self,
        dep_graph: "DependencyGraph",
        runtime_cache_path: Optional[Path],
        rng: "np.random.Generator",
    ) -> None:
        self.dep_graph = dep_graph
        self.rng = rng
        self.runtime_cache_path: Optional[Path] = None
        self.update(runtime_cache_path)
        self.cache: dict["Rule", tuple[Dependency, ...]] = {}

    def update(self, runtime_cache_path: Optional[Path] = None):
        self.runtime_cache_path = runtime_cache_path
        if self.runtime_cache_path is not None and not self.runtime_cache_path.exists():
            os.makedirs(os.path.dirname(self.runtime_cache_path), exist_ok=True)
            self.runtime_cache_path.touch()
            with open(self.runtime_cache_path, "w") as f:
                json.dump({}, f)
        self.cache = {}
        
    def apply_theorem(self, theorem: "Rule", mapping: dict[str, str]) -> Optional[set[Dependency]]:
        res: set[Dependency] = set()
        why: list[Statement] = []
        reason = theorem.descrption
        applicable = True
        for premise in theorem.premises:
            s = Statement.from_tokens(
                translate_sentence(mapping, premise), self.dep_graph
            )
            if s is None:
                applicable = False
                break
            if not s.check_numerical():
                applicable = False
                break
            why.append(s)
        if not applicable:
            return None
        for conclusion in theorem.conclusions:
            conclusion_statement = Statement.from_tokens(
                translate_sentence(mapping, conclusion), self.dep_graph
            )
            # assert conclusion_statement.check_numerical()
            if conclusion_statement is None:
                continue
            dep = Dependency.mk(conclusion_statement, reason, tuple(why))
            res.add(dep)
        return res
    
    def rearrange(self, args: list[str]) -> set[tuple[str]]:
        assert len(args) == 8
        permutations = set()
        for i in range(64):
            perm = args.copy()
            for j in range(4):
                if (i >> j) & 1:
                    perm[2 * j], perm[2 * j + 1] = perm[2 * j + 1], perm[2 * j]
            if (i >> 4) & 1:
                perm[0], perm[1], perm[2], perm[3] = perm[2], perm[3], perm[0], perm[1]
                perm[4], perm[5], perm[6], perm[7] = perm[6], perm[7], perm[4], perm[5]
            if (i >> 5) & 1:
                perm[0], perm[1], perm[4], perm[5] = perm[4], perm[5], perm[0], perm[1]
                perm[2], perm[3], perm[6], perm[7] = perm[6], perm[7], perm[2], perm[3]
            permutations.add(tuple(perm))
        return permutations
    
    def cache_theorem(self, theorem: "Rule"):
        file_cache = None
        write = False
        read = False
        mappings: list[dict[str, str]] = []
        eq_premise = None
        statement_list = None
        for premise in theorem.premises:
            if premise[0] == "eqangle":
                eq_premise = premise
                statement_list = self.dep_graph.numerical_checked_eqangle
                break
            if premise[0] == "eqratio":
                eq_premise = premise
                statement_list = self.dep_graph.numerical_checked_eqratio
                break

        if self.runtime_cache_path is not None:
            with open(self.runtime_cache_path) as f:
                file_cache = json.load(f)
            if "matcher" not in file_cache:
                file_cache["matcher"] = {}
            if str(theorem) in file_cache["matcher"]:
                mappings = file_cache["matcher"][str(theorem)]
                read = True
            else:
                file_cache["matcher"][str(theorem)] = mappings
                write = True
        res: set[Dependency] = set()
        self.cache[theorem] = ()
        points = [p.name for p in self.dep_graph.symbols_graph.nodes_of_type(Point)]
        variables = theorem.variables()
        logging.debug(
            f"{theorem} matching cache : before {len(self.cache[theorem])=} {read=} {write=} {len(mappings)=}"
        )
        if read:
            for mapping in mappings:
                new_conclusions = self.apply_theorem(theorem, mapping)
                if new_conclusions:
                    res.update(new_conclusions)
        elif eq_premise:
            variables_in_premise = set()
            for arg in eq_premise[1:]:
                variables_in_premise.add(arg)
            variables_not_in_premise= list(set(variables) - variables_in_premise)
            for statement in statement_list:
                args_permutation = self.rearrange([p.name for p in statement.args])
                for args in args_permutation:
                    mapping = {}
                    flag = True
                    for v, p in zip(eq_premise[1:], args):
                        if v not in mapping:
                            mapping[v] = p
                        elif mapping[v] != p:
                            flag = False
                            break
                    if not flag:
                        continue
                    
                    for extra_mapping in (
                        {v: p for v, p in zip(variables_not_in_premise, point_list)} 
                        for point_list in itertools.product(points, repeat=len(variables_not_in_premise))
                    ):
                        mapping.update(extra_mapping)
                        # logging.info(f"{theorem} mapping {mapping=}")
                        new_conclusions = self.apply_theorem(theorem, mapping)
                        if new_conclusions:
                            res.update(new_conclusions)
                            if write:
                                mappings.append(mapping)
        else:
            for mapping in (
                (
                    {v: p for v, p in zip(variables, point_list)}
                    for point_list in itertools.product(points, repeat=len(variables))
                )
            ):
                new_conclusions = self.apply_theorem(theorem, mapping)
                if new_conclusions:
                    res.update(new_conclusions)
                    if write:
                        mappings.append(mapping)

        self.cache[theorem] = tuple(
            sorted(res, key=lambda x: repr(x))
        )  # to maintain determinism
        if self.runtime_cache_path is not None and write:
            with open(self.runtime_cache_path, "w") as f:
                json.dump(file_cache, f)
        logging.debug(
            f"{theorem} matching cache : now {len(self.cache[theorem])=} {read=} {write=} {len(mappings)=}"
        )

    def match_theorem(self, theorem: "Rule") -> Generator["Dependency", None, None]:
        if theorem not in self.cache:
            self.cache_theorem(theorem)
        for dep in self.cache[theorem]:
            if dep.statement in dep.statement.dep_graph.hyper_graph:
                continue
            applicable = True
            for premise in dep.why:
                if not premise.check():
                    applicable = False
            if applicable:
                yield dep
