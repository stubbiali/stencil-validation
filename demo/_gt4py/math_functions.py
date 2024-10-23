# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import TYPE_CHECKING

from gt4py.cartesian.gtscript import Field

from stencil_validation.dims import ExpandedDim, I, IJ, J, K
from stencil_validation.stencil_gt4py.descriptors import GT4PyField
from stencil_validation.stencil_gt4py.stencil import GT4PyStencil, MetaGT4PyStencil

if TYPE_CHECKING:
    from types import FunctionType

    from stencil_validation.descriptors import DescriptorDict


IJK_ARGS = lambda dtype_name="float": {
    "dims": (I, J, K),
    "dtype_name": dtype_name,
    "io_dims": (K, IJ),
    "io_dims_map": (IJ, ExpandedDim, K),
    "padding": (0, 0, 1),
}


def math_functions(
    in_a: Field["float"],
    in_b: Field["float"],
    out_c1: Field["float"],
    out_c2: Field["float"],
    out_c3: Field["float"],
    out_c4: Field["float"],
    out_c5: Field["float"],
    out_c6: Field["float"],
    out_c7: Field["float"],
):
    with computation(PARALLEL), interval(...):
        out_c1[0, 0, 0] = exp(in_a[0, 0, 0])
        out_c2[0, 0, 0] = sin(in_a[0, 0, 0])
        out_c3[0, 0, 0] = cos(in_a[0, 0, 0])
        out_c4[0, 0, 0] = in_a[0, 0, 0] ** 0.5
        out_c5[0, 0, 0] = in_a[0, 0, 0] ** 1.5
        out_c6[0, 0, 0] = sqrt(in_a[0, 0, 0])
        out_c7[0, 0, 0] = (in_a[0, 0, 0] * in_b[0, 0, 0]) ** 0.17


class _(GT4PyStencil, metaclass=MetaGT4PyStencil):
    name: str = "math_functions"
    version: str = "demo"
    def_func: FunctionType = math_functions

    @property
    def in_descriptors(self) -> DescriptorDict:
        return {
            "in_a": GT4PyField(io_name="A", random_value_range=(-1000, 1000), **IJK_ARGS()),
            "in_b": GT4PyField(io_name="B", random_value_range=(-1000, 1000), **IJK_ARGS()),
        }

    @property
    def out_descriptors(self) -> DescriptorDict:
        return {
            "out_c1": GT4PyField(io_name="C1", **IJK_ARGS()),
            "out_c2": GT4PyField(io_name="C2", **IJK_ARGS()),
            "out_c3": GT4PyField(io_name="C3", **IJK_ARGS()),
            "out_c4": GT4PyField(io_name="C4", **IJK_ARGS()),
            "out_c5": GT4PyField(io_name="C5", **IJK_ARGS()),
            "out_c6": GT4PyField(io_name="C6", **IJK_ARGS()),
            "out_c7": GT4PyField(io_name="C7", **IJK_ARGS()),
        }

    @property
    def origin(self) -> tuple[int, int, int]:
        return 0, 0, 0

    @property
    def domain(self) -> tuple[int, int, int]:
        return self.config.grid_shape["I"], self.config.grid_shape["J"], self.config.grid_shape["K"]
