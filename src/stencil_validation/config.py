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
from typing import TYPE_CHECKING

import ifs_physics_common

if TYPE_CHECKING:
    from typing import Literal


@dataclasses.dataclass
class Config:
    grid_shape: dict[str, int] = dataclasses.field(
        default_factory=lambda: {"I": 1, "IJ": 1, "J": 1, "K": 1}
    )
    data_shape: dict[str, int] = dataclasses.field(default_factory=dict)
    gt4py_config: ifs_physics_common.GT4PyConfig = dataclasses.field(
        default_factory=lambda: ifs_physics_common.GT4PyConfig(backend="numpy")
    )
    precision: Literal["double", "single"] = "double"

    @property
    def nx(self) -> int:
        return self.grid_shape["I"]

    @property
    def ny(self) -> int:
        return self.grid_shape["J"]

    @property
    def nz(self) -> int:
        return self.grid_shape["K"]

    def with_grid_shape(self, nx: int, ny: int, nz: int) -> Config:
        self.grid_shape["I"] = self.grid_shape["IJ"] = nx
        self.grid_shape["J"] = ny
        self.grid_shape["K"] = nz
        return self

    def with_data_shape(self, **kwargs: int) -> Config:
        self.data_shape = {**self.data_shape, **kwargs}
        return self

    def with_precision(self, precision: Literal["double", "single"]) -> Config:
        self.precision = precision
        self.gt4py_config.dtypes = self.gt4py_config.dtypes.with_precision(precision)
        return self
