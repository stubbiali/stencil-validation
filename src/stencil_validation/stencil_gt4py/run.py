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
import click
import importlib
from typing import TYPE_CHECKING

from stencil_validation.config import Config
from stencil_validation.settings import GLOBAL_SETTINGS
from stencil_validation.stencil_gt4py.stencil import get_gt4py_stencil

if TYPE_CHECKING:
    from typing import Literal, Optional


def run(
    name: str,
    version: str,
    grid_shape: tuple[int, int, int],
    data_shape: dict[str, int],
    precision: Literal["double", "single"],
    imports: tuple[str, ...],
    in_file_path: Optional[str],
    overwrite_in_file: bool,
    out_file_path: Optional[str],
    externals: dict,
    backend: str,
    enable_checks: bool,
    verbose: bool,
) -> None:
    GLOBAL_SETTINGS.with_verbosity(verbose)
    config = (
        Config()
        .with_precision(precision)
        .with_grid_shape(*grid_shape)
        .with_data_shape(**data_shape)
    )

    for module in imports:
        importlib.import_module(module)

    config.gt4py_config = config.gt4py_config.with_backend(backend).with_validate_args(
        enable_checks
    )
    get_gt4py_stencil(name, version, config, externals=externals)(
        in_file_path, out_file_path, overwrite_in_file=overwrite_in_file
    )


@click.command()
@click.option("-n", "--name", type=str)
@click.option("--version", type=str)
@click.option("--nx", type=int, default=1)
@click.option("--ny", type=int, default=1)
@click.option("--nz", type=int, default=1)
@click.option("--data-size", "data_shape", type=(str, int), multiple=True)
@click.option("--precision", type=str, default="double")
@click.option("-i", "--import", "imports", type=str, multiple=True)
@click.option("--input-file", type=str)
@click.option("--overwrite-input-file", is_flag=True, default=False)
@click.option("-o", "--output-file", type=str)
@click.option("-e", "--external", "externals", type=(str, str), multiple=True)
@click.option("--backend", type=str, default="numpy")
@click.option("--enable-checks/--disable-checks", is_flag=True, default=False)
@click.option("--verbose", is_flag=True, default=False)
def main(
    name: str,
    version: str,
    nx: int,
    ny: int,
    nz: int,
    data_shape: tuple[tuple[str, int], ...],
    precision: str,
    imports: tuple[str, ...],
    input_file: str,
    overwrite_input_file: bool,
    output_file: str,
    externals: tuple[tuple[str, str], ...],
    backend: str,
    enable_checks: bool,
    verbose: bool,
) -> None:
    run(
        name,
        version,
        grid_shape=(nx, ny, nz),
        data_shape=dict(data_shape),
        precision=precision,
        imports=imports,
        in_file_path=input_file,
        overwrite_in_file=overwrite_input_file,
        out_file_path=output_file,
        externals=dict(externals),
        backend=backend,
        enable_checks=enable_checks,
        verbose=verbose,
    )
