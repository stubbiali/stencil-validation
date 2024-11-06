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
from abc import abstractmethod
from dataclasses import dataclass, fields
import numpy as np
from typing import TYPE_CHECKING

from ifs_physics_common.utils.numpyx import to_numpy

from stencil_validation.dims import ExpandedDim
from stencil_validation.iox import io_file_operator
from stencil_validation.utils import printx
from stencil_validation.typingx import BoolType, FloatType, IntType

if TYPE_CHECKING:
    from numpy.typing import DTypeLike, NDArray
    from typing import Any, Iterator, Literal, Optional

    from stencil_validation.config import Config
    from stencil_validation.dims import Dim, GenericDim, SizedDim
    from stencil_validation.iox import IOFileOperator


@dataclass
class Descriptor:
    default_io_file_path: Optional[str] = None
    default_value: Optional[Any] = None
    dtype_name: Literal["bool", "float", "int"] = "float"
    io_name: Optional[str] = None
    random_value_range: Optional[tuple[float, float]] = None

    def concretize(
        self, config: Config, io_file_op: Optional[IOFileOperator] = None
    ) -> ConcretizedDescriptor:
        return ConcretizedDescriptor.from_config_and_file(self, config, io_file_op)

    def with_attrs(self, **kwargs: Any) -> Descriptor:
        self_fields = fields(self)
        self_field_names = [f.name for f in self_fields]
        init_kwargs = {name: getattr(self, name) for name in self_field_names}
        for key, value in kwargs.items():
            init_kwargs[key] = value
        return self.__class__(**init_kwargs)

    @abstractmethod
    def get_random_value(self, config: Config) -> Any:
        pass

    @abstractmethod
    def read_value(self, config: Config, io_file_op: IOFileOperator) -> Optional[Any]:
        pass

    @abstractmethod
    def write_value(self, value: Any, config: Config, io_file_op: IOFileOperator) -> None:
        pass

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}(io_name={self.io_name}, default_value={self.default_value})"
        )


@dataclass(frozen=True)
class ConcretizedDescriptor:
    desc: Descriptor
    value: Any

    @classmethod
    def from_config_and_file(
        cls, desc: Descriptor, config: Config, io_file_op: Optional[IOFileOperator] = None
    ):
        printx(f"Concretization of {desc}:")

        value = None

        if desc.io_name is not None:
            if io_file_op is not None:
                value = desc.read_value(config, io_file_op)
                if value is None:
                    printx(f"  * `io_name` not found in `{io_file_op.f_path}`")

            if value is None and desc.default_io_file_path is not None:
                with io_file_operator(desc.default_io_file_path, "r") as io_file_op:
                    if io_file_op is not None:
                        value = desc.read_value(config, io_file_op)
                        if value is None:
                            printx(f"  * `io_name` not found in `{io_file_op.f_path}`")

        if value is not None:
            printx(f"  * `io_name` found in `{io_file_op.f_path}`")
        else:
            if desc.default_value is not None:
                value = desc.default_value
                printx("  * use default value")
            else:
                value = desc.get_random_value(config)
                printx("  * use random value")

        assert value is not None

        return cls(desc, value)

    def to_file(self, config: Config, io_file_op: Optional[IOFileOperator] = None) -> None:
        if io_file_op is not None:
            self.desc.write_value(self.value, config, io_file_op)


@dataclass
class Bool(Descriptor):
    default_value: Optional[BoolType] = None
    dtype_name: Literal["bool"] = "bool"

    def get_random_value(self, config: Config) -> BoolType:
        return config.gt4py_config.dtypes.bool(np.random.rand() < 0.5)

    def read_value(self, config: Config, io_file_op: IOFileOperator) -> Optional[BoolType]:
        value = io_file_op.get_field(self.io_name, dtype=config.gt4py_config.dtypes.bool)
        return value.item() if value is not None else None

    def write_value(self, value: BoolType, config: Config, io_file_op: IOFileOperator) -> None:
        io_file_op.set_field(
            data=np.array([value]), name=self.io_name, dtype=config.gt4py_config.dtypes.bool
        )


class Int(Descriptor):
    default_value: Optional[IntType] = None
    dtype_name: Literal["int"] = "int"

    def get_random_value(self, config) -> IntType:
        low, high = self.random_value_range or (0, 1)
        return np.random.randint(low=low, high=high, dtype=config.gt4py_config.dtypes.int)

    def read_value(self, config: Config, io_file_op: IOFileOperator) -> Optional[IntType]:
        value = io_file_op.get_field(self.io_name, dtype=config.gt4py_config.dtypes.int)
        return value.item() if value is not None else None

    def write_value(self, value: IntType, config: Config, io_file_op: IOFileOperator) -> None:
        io_file_op.set_field(
            data=np.array([value]), name=self.io_name, dtype=config.gt4py_config.dtypes.int
        )


@dataclass
class Float(Descriptor):
    default_value: Optional[FloatType] = None
    dtype_name: Literal["float"] = "float"

    def get_random_value(self, config: Config) -> FloatType:
        low, high = self.random_value_range or (0, 1)
        return config.gt4py_config.dtypes.float(low + (high - low) * np.random.rand())

    def read_value(self, config: Config, io_file_op: IOFileOperator) -> Optional[FloatType]:
        value = io_file_op.get_field(self.io_name, dtype=config.gt4py_config.dtypes.float)
        return value.item() if value is not None else None

    def write_value(self, value: FloatType, config: Config, io_file_op: IOFileOperator) -> None:
        io_file_op.set_field(
            data=np.array([value]), name=self.io_name, dtype=config.gt4py_config.dtypes.float
        )


@dataclass
class Field(Descriptor):
    dims: tuple[Dim, ...] = ()
    io_dims: Optional[tuple[Dim, ...]] = None
    io_dims_map: tuple[GenericDim, ...] = ()
    padding: tuple[int, ...] = ()

    def __post_init__(self):
        # if not otherwise specified, io_dims = dims[::-1]
        self.io_dims = self.io_dims or self.dims[::-1]
        self.io_dims_map = self.io_dims_map or self.io_dims[::-1]

        io_dims_map_filtered = [dim for dim in self.io_dims_map if not dim.squeezed]
        assert len(io_dims_map_filtered) == len(self.dims)

        self.padding = self.padding or (0,) * len(self.dims)

    @property
    def ndim(self) -> int:
        return len(self.dims)

    @property
    def origin(self) -> tuple[int, ...]:
        return tuple(max(-p, 0) for p in self.padding)

    def get_dtype(self, config: Config) -> DTypeLike:
        return getattr(config.gt4py_config.dtypes, self.dtype_name)

    def get_sized_dims(self, config: Config) -> Iterator[SizedDim]:
        return (dim.with_size(config) for dim in self.dims)

    def get_shape(self, config: Config) -> tuple[int, ...]:
        return tuple(dim.size for dim in self.get_sized_dims(config))

    def get_storage_shape(self, config: Config) -> tuple[int, ...]:
        return tuple(s + abs(p) for s, p in zip(self.get_shape(config), self.padding))

    def get_storage_index_slices(self, config: Config) -> tuple[slice, ...]:
        return tuple(slice(o, o + s) for o, s in zip(self.origin, self.get_shape(config)))

    def concretize(
        self, config: Config, io_file_op: Optional[IOFileOperator] = None
    ) -> ConcretizedDescriptor:
        cdesc = super().concretize(config, io_file_op)
        assert cdesc.value.shape == self.get_storage_shape(config)
        return cdesc

    def get_random_value(self, config: Config) -> NDArray:
        low, high = self.random_value_range or (0, 1)
        return np.asarray(
            low + (high - low) * np.random.rand(*self.get_storage_shape(config)),
            dtype=self.get_dtype(config),
        )

    def read_value(self, config: Config, io_file_op: IOFileOperator) -> Optional[NDArray]:
        dtype = self.get_dtype(config)
        value = io_file_op.get_field(
            self.io_name, dims=tuple(dim.with_size(config) for dim in self.io_dims), dtype=dtype
        )
        if value is None:
            return None

        expand_axes = []
        flip_axes = []
        layout_map = []
        index_slices = []
        for i, dim in enumerate(self.io_dims_map):
            if dim == ExpandedDim:
                expand_axes.append(i)
                j = None
            elif dim in self.io_dims:
                j = self.io_dims.index(dim)
            elif -dim in self.io_dims:
                j = self.io_dims.index(-dim)
                flip_axes.append(j)
            else:
                raise ValueError(f"{dim} not found in `io_dims`.")

            if j is not None:
                layout_map.append(j)
                index_slices.append(dim.with_size(config).get_index_slice())

        value = np.flip(value, axis=flip_axes)
        value = np.transpose(value, axes=layout_map)
        value = value[tuple(index_slices)]
        value = np.expand_dims(value, axis=expand_axes)

        assert value.ndim == self.ndim

        target_shape = self.get_shape(config)
        if value.shape != target_shape:
            value = value[tuple(slice(s) for s in target_shape)]
            for i, (target_size, value_size) in enumerate(zip(target_shape, value.shape)):
                reps = tuple(
                    target_size // value_size + 1 if j == i else 1 for j in range(self.ndim)
                )
                value = np.tile(value, reps)
            value = value[tuple(slice(s) for s in target_shape)]

        out = np.zeros(self.get_storage_shape(config), dtype=dtype)
        out_index_slices = tuple(slice(o, o + s) for o, s in zip(self.origin, target_shape))
        out[out_index_slices] = value

        return out

    def write_value(self, value: NDArray, config: Config, io_file_op: IOFileOperator) -> None:
        data = to_numpy(value[self.get_storage_index_slices(config)])

        io_dims_map_filtered = [dim for dim in self.io_dims_map if dim != ExpandedDim]
        squeeze_axes = tuple(i for i, dim in enumerate(self.io_dims_map) if dim == ExpandedDim)
        data = np.squeeze(data, axis=squeeze_axes)

        flip_axes = []
        layout_map = []
        ds_dims = []
        for i, dim in enumerate(self.io_dims):
            if dim in io_dims_map_filtered:
                j = io_dims_map_filtered.index(dim)
            elif -dim in io_dims_map_filtered:
                j = io_dims_map_filtered.index(-dim)
                flip_axes.append(j)
            else:
                raise ValueError(f"{dim} not found in `io_dims_map`.")

            dim_j = io_dims_map_filtered[j]
            ds_dims.append(dim_j.with_size(config))
            if not dim_j.squeezed:
                layout_map.append(j)

        data = np.flip(data, axis=flip_axes)
        data = np.transpose(data, axes=layout_map)

        io_file_op.set_field(data, name=self.io_name, dims=ds_dims, dtype=self.get_dtype(config))


if TYPE_CHECKING:
    DescriptorDict = dict[str, Descriptor]
    ConcretizedDescriptorDict = dict[str, ConcretizedDescriptor]


def inject_io_name(desc_dict: DescriptorDict) -> DescriptorDict:
    for key, desc in desc_dict.items():
        io_name = desc.io_name or key.upper()
        desc_dict[key] = desc.with_attrs(io_name=io_name)
    return desc_dict


def concretize(
    desc_dict: DescriptorDict, config: Config, io_file_op: Optional[IOFileOperator] = None
) -> ConcretizedDescriptorDict:
    cdesc_dict = {}
    for key, desc in desc_dict.items():
        cdesc_dict[key] = desc.concretize(config, io_file_op)
    return cdesc_dict


def to_file(
    cdesc_dict: ConcretizedDescriptorDict, config: Config, io_file_op: IOFileOperator
) -> None:
    for cdesc in cdesc_dict.values():
        cdesc.to_file(config, io_file_op)
