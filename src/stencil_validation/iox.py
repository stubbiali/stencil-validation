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

from stencil_validation.utils import printx

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from numpy.typing import DTypeLike, NDArray
    from typing import Literal, Optional

    from stencil_validation.dims import SizedDim


class IOFileOperator(ABC):
    f_path: str

    def __new__(
        cls, io_file_path: str, mode: Literal["a", "r", "w"], *args, **kwargs
    ) -> Optional[IOFileOperator]:
        f_path = os.path.abspath(io_file_path)

        if mode == "r":
            if not os.path.exists(f_path):
                printx(f"The file `{f_path}` does not exist.")
                return None
        else:
            parent_dir, f_name = f_path.rsplit("/", maxsplit=1)
            os.makedirs(parent_dir, exist_ok=True)

        f_ext = os.path.splitext(f_path)[1]

        if f_ext == "h5":
            return HDF5Operator(f_path, mode)
        elif f_ext == "nc":
            return NetCDFOperator(f_path, mode)
        else:
            printx(f"The file extension `{f_ext}` is not supported.")
            return None

    def __init__(self, io_file_path: str, *args, **kwargs) -> None:
        self.f_path = io_file_path

    @abstractmethod
    @property
    def field_names(self) -> tuple[str, ...]:
        pass

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

    def __del__(self):
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
            dtype = dtype or np.float64
            out = np.asarray(ds[...]).astype(dtype)
            if dims is not None:
                if out.ndim != len(dims):
                    raise RuntimeError(
                        f"H5 field {repr(name)} has {out.ndim} dimensions instead of {len(dims)}."
                    )
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
            index_slices = [dim.get_index_slice() for dim in dims]

        if name not in self.f:
            self.f.create_dataset(name=name, shape=shape, dtype=dtype)
        self.f[name][tuple(index_slices)] = data


class NetCDFOperator(IOFileOperator):
    ds: nc.Dataset

    def __init__(self, io_file_path: str, mode: Literal["a", "r", "w"]) -> None:
        super().__init__(io_file_path)
        self.ds = nc.Dataset(io_file_path, mode=mode)

    @property
    def field_names(self) -> tuple[str, ...]:
        return tuple(self.ds.keys())

    def get_field(
        self,
        name: str,
        dims: Optional[Sequence[SizedDim]] = None,
        dtype: Optional[DTypeLike] = None,
    ) -> Optional[NDArray]:
        pass

    def set_field(
        self,
        data: NDArray,
        name: str,
        dims: Optional[Sequence[SizedDim]] = None,
        dtype: Optional[DTypeLike] = None,
    ) -> None:
        pass


@contextmanager
def io_file_operator(
    io_file_path: Optional[str], mode: Literal["a", "r", "w"]
) -> Iterator[Optional[IOFileOperator]]:
    try:
        op = IOFileOperator(io_file_path, mode) if io_file_path is not None else None
        yield op
    finally:
        if op is not None:
            del op
