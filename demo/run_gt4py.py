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

from stencil_validation.stencil_gt4py.run import run


@click.command()
@click.option("--nlon", type=int, default=1)
@click.option("--nlev", type=int, default=1)
@click.option("--precision", type=str, default="double")
@click.option("--input-file", type=str)
@click.option("--write-input-file", type=str)
@click.option("-o", "--output-file", type=str)
@click.option("--backend", type=str, default="numpy")
@click.option("--enable-checks/--disable-checks", is_flag=True, default=False)
@click.option("--verbose", is_flag=True, default=False)
def main(
    nlon: int,
    nlev: int,
    precision: str,
    input_file: str,
    write_input_file: str,
    output_file: str,
    backend: str,
    enable_checks: bool,
    verbose: bool,
) -> None:
    run(
        name="math_functions",
        version="demo",
        grid_shape=(nlon, 1, nlev),
        data_shape={},
        precision=precision,  # type: ignore[arg-type]
        imports=("_gt4py.math_functions",),
        in_file_path=input_file,
        write_in_file_path=write_input_file,
        out_file_path=output_file,
        externals={},
        backend=backend,
        enable_checks=enable_checks,
        verbose=verbose,
        print_stencil_list=False,
    )


if __name__ == "__main__":
    main()
