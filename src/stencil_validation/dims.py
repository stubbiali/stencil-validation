# -*- coding: utf-8 -*-
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

from __future__ import annotations

import dataclasses
import enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Union

    from stencil_validation.config import Config


class Direction(enum.Enum):
    POSITIVE: int = 1
    NEGATIVE: int = -1


def flip(direction: Direction) -> Direction:
    return Direction.NEGATIVE if direction == Direction.POSITIVE else Direction.POSITIVE


@dataclasses.dataclass(frozen=True)
class Dim:
    name: str
    offset: float = 0
    direction: Direction = Direction.POSITIVE
    static_size: int | None = None

    # the following attributes ensure better inter-operability with IndexedDim
    index: None = None
    squeezed: bool = False

    def __post_init__(self) -> None:
        assert self.offset in (-0.5, 0, 0.5)
        assert self.index is None
        assert not self.squeezed

    def __add__(self, other: float) -> Dim:
        if other == 0:
            return self
        elif other in (-0.5, 0.5):
            return Dim(self.name, other if self.offset == 0 else 0, self.direction)
        else:
            raise ValueError(f"Invalid offset {other}.")

    def __sub__(self, other: float) -> Dim:
        return self + (-other)

    def __neg__(self) -> Dim:
        return Dim(self.name, self.offset, flip(self.direction))

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Dim):
            return (
                self.name == other.name
                and self.offset == other.offset
                and self.direction == other.direction
            )
        elif isinstance(other, IndexedDim):
            return self == other.dim
        else:
            return False

    def __getitem__(self, index: int | slice) -> IndexedDim:
        return IndexedDim(self, index)

    def __hash__(self) -> int:
        return hash((self.name, self.offset, self.direction))

    def __repr__(self) -> str:
        if self.offset > 0:
            prefix, suffix = ("-(", ")") if self.direction == Direction.NEGATIVE else ("", "")
            return f"{prefix}{self.name} + {self.offset}{suffix}"
        elif self.offset < 0:
            prefix, suffix = ("-(", ")") if self.direction == Direction.NEGATIVE else ("", "")
            return f"{prefix}{self.name} - {-self.offset}{suffix}"
        else:
            return f"-{self.name}" if self.direction == Direction.NEGATIVE else f"{self.name}"

    def without_offset(self) -> Dim:
        return Dim(self.name, 0, self.direction, self.static_size)

    def with_size(self, config: Config | None = None) -> SizedDim:
        if config is None:
            assert self.static_size is not None
            return SizedDim(self, self.static_size)
        else:
            return SizedDim.from_config(self, config)


@dataclasses.dataclass(frozen=True)
class IndexedDim:
    dim: Dim
    index: int | slice

    def __post_init__(self) -> None:
        if isinstance(self.index, slice):
            assert self.index.step is None
            if self.index.start is None:
                assert self.index.stop == 1
            elif self.index.stop is None:
                assert self.index.start == -1
            else:
                assert self.index.stop - self.index.start == 1

    @property
    def name(self) -> str:
        return self.dim.name

    @property
    def squeezed(self) -> bool:
        return isinstance(self.index, int)

    def __neg__(self) -> IndexedDim:
        return IndexedDim(-self.dim, self.index)

    def __hash__(self) -> int:
        return hash((self.dim, self.index))

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Dim):
            return self.dim == other
        elif isinstance(other, IndexedDim):
            return self.dim == other.dim and self.index == other.index
        else:
            return False

    def with_size(self, config: Config) -> SizedDim:
        return SizedDim.from_config(self, config)


@dataclasses.dataclass(frozen=True)
class SizedDim:
    dim: Dim | IndexedDim
    size: int

    @classmethod
    def from_config(cls, dim: Dim | IndexedDim, config: Config) -> SizedDim:
        inner_dim = dim.dim if isinstance(dim, IndexedDim) else dim
        size = config.grid_shape.get(
            inner_dim.without_offset(), config.data_shape.get(inner_dim.name, inner_dim.static_size)
        )
        if size is None:
            raise RuntimeError(f"Size not specified for dim `{inner_dim.name}`.")
        if inner_dim.offset in (-0.5, 0.5):
            size += 1
        return cls(dim, size)

    def get_index_slice(self) -> int | slice:
        return slice(0, self.size) if isinstance(self.dim, Dim) else self.dim.index


if TYPE_CHECKING:
    from typing_extensions import TypeAlias

    GenericDim: TypeAlias = Union[Dim, IndexedDim]


ExpandedDim = Dim("ExpandedDim")
I = Dim("I")  # noqa: E741
IJ = Dim("IJ")
J = Dim("J")
K = Dim("K")
