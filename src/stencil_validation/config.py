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
import os
from typing import TYPE_CHECKING

import ifs_physics_common

from stencil_validation.dims import IJ, I, J, K

if TYPE_CHECKING:
    from typing import Literal

    from stencil_validation.dims import Dim


@dataclasses.dataclass
class Config:
    data_shape: dict[str, int]
    grid_shape: dict[Dim, int]
    gt4py_config: ifs_physics_common.GT4PyConfig
    precision: Literal["double", "single"] = "double"
    project_root: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    verbose: bool = False

    @classmethod
    def from_cli(
        cls,
        nx: int = 1,
        ny: int = 1,
        nz: int = 1,
        data_shape: dict[str, int] | None = None,
        precision: Literal["double", "single"] = "double",
        verbose: bool = False,
        gt4py_backend: str = "numpy",
        gt4py_validate_args: bool = True,
    ) -> Config:
        return cls(
            grid_shape={I: nx, IJ: nx, J: ny, K: nz},
            data_shape=data_shape or {},
            precision=precision,
            verbose=verbose,
            gt4py_config=ifs_physics_common.GT4PyConfig(
                backend=gt4py_backend,
                dtypes=ifs_physics_common.DataTypes.from_precision(precision),
                validate_args=gt4py_validate_args,
                verbose=verbose,
            ),
        )

    @property
    def nx(self) -> int:
        return self.grid_shape[I]

    @property
    def ny(self) -> int:
        return self.grid_shape[J]

    @property
    def nz(self) -> int:
        return self.grid_shape[K]
