from __future__ import annotations

"""Sparse linear MILP builder primitives for the OFDMA slot-indexed model."""

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import LinearConstraint
from scipy.sparse import coo_matrix

from .models import MilpModelSize


@dataclass
class VariableBuilder:
    lower_bounds: list[float] = field(default_factory=list)
    upper_bounds: list[float] = field(default_factory=list)
    integrality: list[int] = field(default_factory=list)
    objective: list[float] = field(default_factory=list)

    def add_variable(self, *, lower: float, upper: float, integer: bool, objective: float = 0.0) -> int:
        variable_id = len(self.objective)
        self.lower_bounds.append(float(lower))
        self.upper_bounds.append(float(upper))
        self.integrality.append(1 if bool(integer) else 0)
        self.objective.append(float(objective))
        return int(variable_id)


@dataclass
class ConstraintBuilder:
    row_indices: list[int] = field(default_factory=list)
    col_indices: list[int] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    lower_bounds: list[float] = field(default_factory=list)
    upper_bounds: list[float] = field(default_factory=list)

    def add_constraint(self, terms: list[tuple[int, float]], *, lower: float, upper: float) -> None:
        row_id = len(self.lower_bounds)
        for variable_id, coefficient in terms:
            if abs(float(coefficient)) <= 1e-15:
                continue
            self.row_indices.append(int(row_id))
            self.col_indices.append(int(variable_id))
            self.values.append(float(coefficient))
        self.lower_bounds.append(float(lower))
        self.upper_bounds.append(float(upper))

    def to_linear_constraint(self, *, variable_count: int) -> LinearConstraint:
        matrix = coo_matrix(
            (self.values, (self.row_indices, self.col_indices)),
            shape=(len(self.lower_bounds), int(variable_count)),
        ).tocsr()
        return LinearConstraint(
            matrix,
            np.asarray(self.lower_bounds, dtype=float),
            np.asarray(self.upper_bounds, dtype=float),
        )


def build_model_size(
    variable_builder: VariableBuilder,
    constraint_builder: ConstraintBuilder,
) -> MilpModelSize:
    binary_count = sum(
        1
        for integer, lower, upper in zip(
            variable_builder.integrality,
            variable_builder.lower_bounds,
            variable_builder.upper_bounds,
        )
        if int(integer) == 1 and float(lower) == 0.0 and float(upper) <= 1.0
    )
    return MilpModelSize(
        variable_count=int(len(variable_builder.objective)),
        binary_variable_count=int(binary_count),
        continuous_variable_count=int(len(variable_builder.objective) - binary_count),
        constraint_count=int(len(constraint_builder.lower_bounds)),
        nonzero_count=int(len(constraint_builder.values)),
    )


__all__ = ["ConstraintBuilder", "VariableBuilder", "build_model_size"]
