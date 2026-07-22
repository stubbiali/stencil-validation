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

import warnings

import click
import numpy as np

from stencil_validation.iox import io_file_operator

ATOL = 1e-12
RTOL = 1e-10


@click.command()
@click.argument("src_file_path", type=str)
@click.argument("trg_file_path", type=str)
@click.option("--atol", type=float, default=None)
@click.option("--rtol", type=float, default=None)
@click.option("--verbose", is_flag=True, default=False)
def compare_io_files(
    src_file_path: str,
    trg_file_path: str,
    index_slices: tuple[slice, ...] | None = None,
    atol: float | None = ATOL,
    rtol: float | None = RTOL,
    verbose: bool = False,
) -> None:
    with io_file_operator(src_file_path, mode="r", verbose=verbose) as src_file_op:
        with io_file_operator(trg_file_path, mode="r", verbose=verbose) as trg_file_op:
            if src_file_op is None:
                print(f"== iodiff: cannot open `{src_file_path}`")
            elif trg_file_op is None:
                print(f"== iodiff: cannot open `{trg_file_path}`")
            else:
                atol = atol or ATOL
                rtol = rtol or RTOL

                print("== iodiff: start\n")
                print(f"   - source file: {src_file_op.f_path}")
                print(f"   - target file: {trg_file_op.f_path}")
                print(f"   - atol: {atol:.1E}")
                print(f"   - rtol: {rtol:.1E}\n")

                common_keys = sorted(
                    (src_keys := {*src_file_op.field_names}).intersection(
                        trg_keys := {*trg_file_op.field_names}
                    )
                )
                if src_keys_only := (src_keys - {*common_keys}):
                    print(f"   - fields found only in source file: {','.join(src_keys_only)}")
                if trg_keys_only := (trg_keys - {*common_keys}):
                    print(f"   - fields found only in target file: {','.join(trg_keys_only)}")
                if src_keys_only or trg_keys_only:
                    print("")

                if len(common_keys) > 0:
                    max_key_len = max(len(key) for key in common_keys)
                    for key in common_keys:
                        src_field = src_file_op.get_field(key)[  # type: ignore[index]
                            index_slices if index_slices is not None else ...
                        ]
                        trg_field = trg_file_op.get_field(key)[  # type: ignore[index]
                            index_slices if index_slices is not None else ...
                        ]
                        assert src_field.shape == trg_field.shape

                        if src_field.dtype.kind == "b":
                            src_field = src_field.astype(float)
                        if trg_field.dtype.kind == "b":
                            trg_field = trg_field.astype(float)

                        # remove nan's and inf's
                        src_field = np.where(np.isnan(src_field), 0, src_field)
                        src_field = np.where(np.isinf(src_field), 0, src_field)
                        trg_field = np.where(np.isnan(trg_field), 0, trg_field)
                        trg_field = np.where(np.isinf(trg_field), 0, trg_field)

                        abs_diff = np.abs(src_field - trg_field)
                        abs_diff_max = abs_diff.max()
                        with warnings.catch_warnings():
                            warnings.filterwarnings("ignore")
                            rel_diff = abs_diff / np.abs(trg_field)
                        rel_diff_max = np.where(trg_field != 0, rel_diff, 0).max()

                        close = np.abs(src_field - trg_field) <= atol + rtol * np.abs(trg_field)
                        allclose = np.all(close)

                        freq_atol = np.sum(abs_diff >= atol) / abs_diff.size * 100
                        freq_rtol = np.sum(rel_diff >= rtol) / abs_diff.size * 100
                        freq_close = np.sum(np.logical_not(close)) / abs_diff.size * 100

                        print(
                            f"   {key:{max_key_len}s}:"
                            f"\033[9{2 if abs_diff_max < atol else 1}m "
                            f"max abs diff = {abs_diff_max:.5E} "
                            f"({f'{freq_atol:.2f}'.zfill(5)} %)\033[00m,"
                            f"\033[9{2 if rel_diff_max < rtol else 1}m "
                            f"max rel diff = {rel_diff_max:.5E} "
                            f"({f'{freq_rtol:.2f}'.zfill(5)} %)\033[00m,"
                            f"\033[9{2 if allclose else 1}m allclose = {allclose} "
                            f"({f'{freq_close:.2f}'.zfill(5)} %)\033[00m"
                        )

                print("\n== iodiff: end")


if __name__ == "__main__":
    compare_io_files()
