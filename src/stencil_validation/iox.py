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

import contextlib
import dataclasses
import os
from typing import TYPE_CHECKING

import h5py as h5
import netCDF4 as nc
import numpy as np

from stencil_validation.dims import Dim, SizedDim
from stencil_validation.units import get_conversion_factor
from stencil_validation.utils import printx

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from typing import Literal

    import numpy.typing as npt


@dataclasses.dataclass
class IOFileOperator:
    f_path: str = ""
    mode: Literal["a", "r", "w", "e"] = "r"
    verbose: bool = False
    error_msg: str = ""

    def __post_init__(self) -> None:
        if self.mode == "e":
            assert self.error_msg

    @property
    def field_names(self) -> tuple[str, ...]:
        return ()

    def get_field(
        self,
        name: str,
        dims: Sequence[SizedDim] | None = None,
        dtype: npt.DTypeLike | None = None,
        units: str | None = None,
    ) -> npt.NDArray | None: ...

    def set_field(
        self,
        data: npt.NDArray,
        name: str,
        dims: Sequence[SizedDim] | None = None,
        dtype: npt.DTypeLike | None = None,
        units: str | None = None,
    ) -> None: ...


DUMMY_IO_FILE_OP = IOFileOperator()


@dataclasses.dataclass
class HDF5Operator(IOFileOperator):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.f = h5.File(self.f_path, mode=self.mode)

    def __del__(self) -> None:
        self.f.close()

    @property
    def field_names(self) -> tuple[str, ...]:
        return tuple(self.f.keys())

    def get_field(
        self,
        name: str,
        dims: Sequence[SizedDim] | None = None,
        dtype: npt.DTypeLike | None = None,
        units: str | None = None,
    ) -> npt.NDArray | None:
        ds = self.f.get(name, None)
        if ds is None:
            return None
        else:
            out = np.asarray(ds[...])
            if dims is not None:
                if out.ndim != len(dims):
                    printx(
                        f"H5 field `{name}` has {out.ndim} dimensions; expected {len(dims)}.",
                        color="grey",
                        verbose=self.verbose,
                    )
                    return None
            if dtype is not None:
                out = out.astype(dtype)
            return out

    def set_field(
        self,
        data: npt.NDArray,
        name: str,
        dims: Sequence[SizedDim] | None = None,
        dtype: npt.DTypeLike | None = None,
        units: str | None = None,
    ) -> None:
        dtype = dtype or data.dtype
        if dims is None:
            if data.size != 1:
                raise RuntimeError(
                    f"If `dims` is `None`, `data` must be 1-item, but has {data.size} elements."
                )
            shape = [1]
            index_slices = [0]
        else:
            shape = [dim.size for dim in dims]
            index_slices = [dim.get_index_slice() for dim in dims]  # type: ignore[misc]

        if name not in self.f:
            self.f.create_dataset(name=name, shape=shape, dtype=dtype)
        self.f[name][tuple(index_slices)] = data


Scalar = Dim("scalar", static_size=1).with_size()


@dataclasses.dataclass
class NetCDFOperator(IOFileOperator):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.ds = nc.Dataset(self.f_path, mode=self.mode)

    def __del__(self) -> None:
        self.ds.close()

    @property
    def field_names(self) -> tuple[str, ...]:
        return tuple(self.ds.variables)

    def get_field(
        self,
        name: str,
        dims: Sequence[SizedDim] | None = None,
        dtype: npt.DTypeLike | None = None,
        units: str | None = None,
    ) -> npt.NDArray | None:
        if name not in self.ds.variables:
            return None
        else:
            if (
                units is not None
                and (ds_units := getattr(self.ds[name], "units", None)) is not None
            ):
                factor = get_conversion_factor(ds_units, units)
            else:
                factor = 1.0

            out = factor * np.asarray(self.ds[name])

            if dims is not None:
                if out.ndim != len(dims):
                    printx(
                        f"NetCDF field `{name}` has {out.ndim} dimensions; expected {len(dims)}.",
                        color="grey",
                        verbose=self.verbose,
                    )
                    return None

            if dtype is not None:
                out = out.astype(dtype)

            return out

    def set_field(
        self,
        data: npt.NDArray,
        name: str,
        dims: Sequence[SizedDim] | None = None,
        dtype: npt.DTypeLike | None = None,
        units: str | None = None,
    ) -> None:
        dtype = dtype or data.dtype
        if dims is None:
            if data.size != 1:
                raise RuntimeError(
                    f"If `dims` is `None`, `data` must be 1-item, but has {data.size} elements."
                )

            if name not in self.ds.variables:
                self.ds.createVariable(name, dtype)  # type: ignore[arg-type]

            self.ds[name][...] = data
        else:
            index_slices = [dim.get_index_slice() for dim in dims]  # type: ignore[misc]

            nc_dims = [str(dim.dim).replace(" ", "") for dim in dims]
            for nc_dim, dim in zip(nc_dims, dims):
                if nc_dim not in self.ds.dimensions:
                    self.ds.createDimension(nc_dim, dim.size)

            if name not in self.ds.variables:
                self.ds.createVariable(name, dtype, nc_dims)  # type: ignore[arg-type]

            self.ds[name][tuple(index_slices)] = data

        if units is not None:
            self.ds[name].units = units


@contextlib.contextmanager
def io_file_operator(
    io_file_path: str | None, mode: Literal["a", "r", "w"], verbose: bool = False
) -> Iterator[IOFileOperator]:
    op = DUMMY_IO_FILE_OP

    if io_file_path is not None:
        f_path = os.path.abspath(io_file_path)

        if mode == "r" and not os.path.exists(f_path):
            printx(f"The file `{f_path}` does not exist.", verbose=verbose)
        else:
            parent_dir, _ = f_path.rsplit("/", maxsplit=1)
            os.makedirs(parent_dir, exist_ok=True)

            f_ext = os.path.splitext(f_path)[1][1:]

            if f_ext == "h5":
                op = HDF5Operator(f_path, mode)
            elif f_ext == "nc":
                op = NetCDFOperator(f_path, mode)
            else:
                printx(f"The file extension `{f_ext}` is not supported.", verbose=verbose)

    try:
        yield op
    finally:
        if op != DUMMY_IO_FILE_OP:
            del op
