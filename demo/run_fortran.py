# -*- coding: utf-8 -*-
import click
import os

from stencil_validation.config import Config
from stencil_validation.settings import GLOBAL_SETTINGS
from stencil_validation.stencil_fortran.stencil import get_fortran_stencil

import _fortran.math_functions

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
@click.option("-d", "--data", type=(str, str), multiple=True)
@click.option("--opt-level", type=int, default=3)
@click.option("--verbose", is_flag=True, default=False)
def main(
    version: str,
    name: str,
    nlon: int,
    nlev: int,
    precision: str,
    input_file: str,
    overwrite_input_file: bool,
    output_file: str,
    data: tuple[tuple[str, str], ...],
    opt_level: int,
    verbose: bool,
) -> None:
    GLOBAL_SETTINGS.with_verbosity(verbose)
    config = Config().with_precision(precision).with_grid_shape(nlon, 1, nlev)
    data = {
        **dict(data),
        "nlon": config.grid_shape["IJ"],
        "nlev": config.grid_shape["K"],
        "precision": precision,
    }
    get_fortran_stencil(name, version, config)(
        data,
        input_file,
        output_file,
        overwrite_in_file=overwrite_input_file,
        opt_level=opt_level,
        rebuild=True,
    )


if __name__ == "__main__":
    main()
