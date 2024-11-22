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
from abc import ABC, abstractmethod
from contextlib import contextmanager
import h5py as h5
import netCDF4 as nc
import numpy as np
import os
from typing import TYPE_CHECKING

from stencil_validation.dims import Dim, SizedDim
from stencil_validation.utils import printx

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from numpy.typing import DTypeLike, NDArray
    from typing import Any, Literal, Optional


class IOFileOperator(ABC):
    f_path: str

    def __init__(self, io_file_path: str, *args: Any, **kwargs: Any) -> None:
        self.f_path = io_file_path

    @property
    def field_names(self) -> tuple[str, ...]:
        return ()

    @abstractmethod
    def get_field(
        self,
        name: str,
        dims: Optional[Sequence[SizedDim]] = None,
        dtype: Optional[DTypeLike] = None,
    ) -> Optional[NDArray]:
        pass

    @abstractmethod
    def set_field(
        self,
        data: NDArray,
        name: str,
        dims: Optional[Sequence[SizedDim]] = None,
        dtype: Optional[DTypeLike] = None,
    ) -> None:
        pass


class HDF5Operator(IOFileOperator):
    f: h5.File

    def __init__(self, io_file_path: str, mode: Literal["a", "r", "w"]) -> None:
        super().__init__(io_file_path)
        self.f = h5.File(io_file_path, mode=mode)

    def __del__(self) -> None:
        self.f.close()

    @property
    def field_names(self) -> tuple[str, ...]:
        return tuple(self.f.keys())

    def get_field(
        self,
        name: str,
        dims: Optional[Sequence[SizedDim]] = None,
        dtype: Optional[DTypeLike] = None,
    ) -> Optional[NDArray]:
        ds = self.f.get(name, None)
        if ds is None:
            return None
        else:
            out = np.asarray(ds[...])
            if dims is not None:
                if out.ndim != len(dims):
                    raise RuntimeError(
                        f"H5 field `{name}` has {out.ndim} dimensions instead of {len(dims)}."
                    )
            if dtype is not None:
                out = out.astype(dtype)
            return out

    def set_field(
        self,
        data: NDArray,
        name: str,
        dims: Optional[Sequence[SizedDim]] = None,
        dtype: Optional[DTypeLike] = None,
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


class NetCDFOperator(IOFileOperator):
    ds: nc.Dataset

    def __init__(self, io_file_path: str, mode: Literal["a", "r", "w"]) -> None:
        super().__init__(io_file_path)
        self.ds = nc.Dataset(io_file_path, mode=mode)

    @property
    def field_names(self) -> tuple[str, ...]:
        return tuple(self.ds.variables)

    def get_field(
        self,
        name: str,
        dims: Optional[Sequence[SizedDim]] = None,
        dtype: Optional[DTypeLike] = None,
    ) -> Optional[NDArray]:
        if name not in self.ds.variables:
            return None
        else:
            out = np.asarray(self.ds[name])
            if dims is not None:
                if out.ndim != len(dims):
                    raise RuntimeError(
                        f"H5 field `{name}` has {out.ndim} dimensions instead of {len(dims)}."
                    )
            if dtype is not None:
                out = out.astype(dtype)
            return out

    def set_field(
        self,
        data: NDArray,
        name: str,
        dims: Optional[Sequence[SizedDim]] = None,
        dtype: Optional[DTypeLike] = None,
    ) -> None:
        dtype = dtype or data.dtype
        if dims is None:
            if data.size != 1:
                raise RuntimeError(
                    f"If `dims` is `None`, `data` must be 1-item, but has {data.size} elements."
                )
            index_slices = [0]
            dims = [Scalar]
        else:
            index_slices = [dim.get_index_slice() for dim in dims]  # type: ignore[misc]

        nc_dims = [str(dim.dim).replace(" ", "") for dim in dims]
        for nc_dim, dim in zip(nc_dims, dims):
            if nc_dim not in self.ds.dimensions:
                self.ds.createDimension(nc_dim, dim.size)

        if name not in self.ds.variables:
            self.ds.createVariable(name, dtype, nc_dims)  # type: ignore[arg-type]
        self.ds[name][tuple(index_slices)] = data


@contextmanager
def io_file_operator(
    io_file_path: Optional[str], mode: Literal["a", "r", "w"]
) -> Iterator[Optional[IOFileOperator]]:
    op: Optional[IOFileOperator] = None

    if io_file_path is not None:
        f_path = os.path.abspath(io_file_path)

        if mode == "r" and not os.path.exists(f_path):
            printx(f"The file `{f_path}` does not exist.")
        else:
            parent_dir, f_name = f_path.rsplit("/", maxsplit=1)
            os.makedirs(parent_dir, exist_ok=True)

            f_ext = os.path.splitext(f_path)[1][1:]

            if f_ext == "h5":
                op = HDF5Operator(f_path, mode)
            elif f_ext == "nc":
                op = NetCDFOperator(f_path, mode)
            else:
                printx(f"The file extension `{f_ext}` is not supported.")

    try:
        yield op
    finally:
        if op is not None:
            del op
