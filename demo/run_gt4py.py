# -*- coding: utf-8 -*-
from __future__ import annotations
import click
import os
from typing import TYPE_CHECKING

from stencil_validation.config import Config
from stencil_validation.settings import GLOBAL_SETTINGS
from stencil_validation.stencil_gt4py.stencil import get_gt4py_stencil

import _gt4py.math_functions

if TYPE_CHECKING:
    from typing import Literal


this_dir = os.path.dirname(__file__)
INPUT_FILE_PATH = os.path.join(this_dir, "input.h5")
OUTPUT_FILE_PATH = os.path.join(this_dir, "reference.h5")


@click.command()
@click.option("--version", type=str)
@click.option("-n", "--name", type=str)
@click.option("--nlon", type=int, default=1)
@click.option("--nlev", type=int, default=1)
@click.option("--precision", type=str, default="double")
@click.option("-i", "--input-file", type=str, default=INPUT_FILE_PATH)
@click.option("--overwrite-input-file", is_flag=True, default=False)
@click.option("-o", "--output-file", type=str, default=OUTPUT_FILE_PATH)
@click.option("-e", "--external", type=(str, str), multiple=True)
@click.option("--backend", type=str, default="numpy")
@click.option("--enable-checks/--disable-checks", is_flag=True, default=False)
@click.option("--verbose", is_flag=True, default=False)
def main(
    version: str,
    name: str,
    nlon: int,
    nlev: int,
    precision: Literal["double", "single"],
    input_file: str,
    overwrite_input_file: bool,
    output_file: str,
    external: tuple[tuple[str, str], ...],
    backend: str,
    enable_checks: bool,
    verbose: bool,
) -> None:
    GLOBAL_SETTINGS.with_verbosity(verbose)
    config = (
        Config()
        .with_precision(precision)
        .with_grid_shape(nlon, 1, nlev)
        .with_data_shape(D2=2, D4=4, D5=5)
    )
    config.gt4py_config = config.gt4py_config.with_backend(backend).with_validate_args(
        enable_checks
    )
    get_gt4py_stencil(name, version, config, externals=dict(external))(
        input_file, output_file, overwrite_in_file=overwrite_input_file
    )


if __name__ == "__main__":
    main()
