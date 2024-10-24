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

import click

from stencil_validation.stencil_fortran.run import run


@click.command()
@click.option("-n", "--name", type=str)
@click.option("--version", type=str)
@click.option("--nlon", type=int, default=1)
@click.option("--nlev", type=int, default=1)
@click.option("--precision", type=str, default="double")
@click.option("--input-file", type=str)
@click.option("--overwrite-input-file", is_flag=True, default=False)
@click.option("-o", "--output-file", type=str)
@click.option("-d", "--data", type=(str, str), multiple=True)
@click.option("--opt-level", type=int, default=3)
@click.option("--verbose", is_flag=True, default=False)
def main(
    name: str,
    version: str,
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
    run(
        name,
        version,
        grid_shape=(nlon, 1, nlev),
        data_shape={},
        precision=precision,
        imports=("_fortran.math_functions",),
        in_file_path=input_file,
        overwrite_in_file=overwrite_input_file,
        out_file_path=output_file,
        data={**dict(data), "nlon": nlon, "nlev": nlev, "precision": precision},
        opt_level=opt_level,
        verbose=verbose,
    )


if __name__ == "__main__":
    main()
