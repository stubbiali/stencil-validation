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

import importlib
from typing import TYPE_CHECKING

import click

from stencil_validation.config import Config
from stencil_validation.settings import GLOBAL_SETTINGS
from stencil_validation.stencil_gt4py.stencil import get_gt4py_stencil, print_gt4py_stencil_list

if TYPE_CHECKING:
    from typing import Literal


def run(
    name: str,
    version: str,
    grid_shape: tuple[int, int, int],
    data_shape: dict[str, int],
    precision: Literal["double", "single"],
    imports: tuple[str, ...],
    in_file_paths: tuple[str, ...] | None,
    write_in_file_path: str | None,
    out_file_path: str | None,
    externals: dict,
    backend: str,
    enable_checks: bool,
    num_runs: int | None,
    verbose: bool,
    print_stencil_list: bool,
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

    if print_stencil_list:
        print_gt4py_stencil_list()
    else:
        config.gt4py_config = config.gt4py_config.with_backend(backend).with_validate_args(
            enable_checks
        )
        get_gt4py_stencil(name, version, config, externals=externals)(
            in_file_paths, write_in_file_path, out_file_path, num_runs=num_runs
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
@click.option("--input-file", "input_files", type=str, multiple=True)
@click.option("--write-input-file", type=str)
@click.option("-o", "--output-file", type=str)
@click.option("-e", "--external", "externals", type=(str, str), multiple=True)
@click.option("--backend", type=str, default="numpy")
@click.option("--enable-checks/--disable-checks", is_flag=True, default=False)
@click.option("--num-runs", type=int, default=0)
@click.option("--verbose", is_flag=True, default=False)
@click.option("-l", "--list", "print_stencil_list", is_flag=True, default=False)
def main(
    name: str,
    version: str,
    nx: int,
    ny: int,
    nz: int,
    data_shape: tuple[tuple[str, int], ...],
    precision: str,
    imports: tuple[str, ...],
    input_files: tuple[str, ...],
    write_input_file: str,
    output_file: str,
    externals: tuple[tuple[str, str], ...],
    backend: str,
    enable_checks: bool,
    num_runs: int,
    verbose: bool,
    print_stencil_list: bool,
) -> None:
    run(
        name,
        version,
        grid_shape=(nx, ny, nz),
        data_shape=dict(data_shape),
        precision=precision,  # type: ignore[arg-type]
        imports=imports,
        in_file_paths=input_files,
        write_in_file_path=write_input_file,
        out_file_path=output_file,
        externals=dict(externals),
        backend=backend,
        enable_checks=enable_checks,
        num_runs=num_runs,
        verbose=verbose,
        print_stencil_list=print_stencil_list,
    )


if __name__ == "__main__":
    main()
