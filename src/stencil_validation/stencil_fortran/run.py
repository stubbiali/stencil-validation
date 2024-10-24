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
from stencil_validation.stencil_fortran.stencil import get_fortran_stencil

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
    data: dict,
    opt_level: int,
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

    get_fortran_stencil(name, version, config)(
        data,
        in_file_path,
        out_file_path,
        overwrite_in_file=overwrite_in_file,
        opt_level=opt_level,
        rebuild=True,
    )


@click.command()
@click.option("-n", "--name", type=str)
@click.option("--version", type=str)
@click.option("--nx", type=int, default=1)
@click.option("--ny", type=int, default=1)
@click.option("--nz", type=int, default=1)
@click.option("--data-size", "data_shape", type=(str, int), multiple=True)
@click.option("--precision", type=str, default="double")
@click.option("--input-file", type=str)
@click.option("--overwrite-input-file", is_flag=True, default=False)
@click.option("-o", "--output-file", type=str)
@click.option("-d", "--data", type=(str, str), multiple=True)
@click.option("--opt-level", type=int, default=3)
@click.option("--verbose", is_flag=True, default=False)
@click.option("-i", "--import", "imports", type=str, multiple=True)
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
    data: tuple[tuple[str, str], ...],
    opt_level: int,
    verbose: bool,
) -> None:
    run(
        name,
        version,
        grid_shape=(nx, ny, nz),
        data_shape=dict(data_shape),
        precision=precision,
        imports=imports,
        input_file=input_file,
        overwrite_input_file=overwrite_input_file,
        output_file=output_file,
        data=dict(data),
        opt_level=opt_level,
        verbose=verbose,
    )
