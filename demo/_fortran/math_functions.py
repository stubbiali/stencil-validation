# -*- coding: utf-8 -*-
from __future__ import annotations
import os
from typing import TYPE_CHECKING

from stencil_validation.dims import IJ, K
from stencil_validation.stencil_fortran.descriptors import FortranField
from stencil_validation.stencil_fortran.stencil import FortranStencil, MetaFortranStencil

if TYPE_CHECKING:
    from stencil_validation.descriptors import DescriptorDict


IJK_ARGS = lambda dtype_name="float": {
    "dims": (IJ, K),
    "dtype_name": dtype_name,
    "io_dims": (K, IJ),
}


class _(FortranStencil, metaclass=MetaFortranStencil):
    name: str = "math_functions"
    version: str = "demo"
    template_file_path: str = os.path.join(os.path.dirname(__file__), "math_functions.F90.in")
    template_var_info: dict[str, dict] = {
        "nlev": {"type": int},
        "nlon": {"type": int},
        "precision": {"type": str},
    }

    @property
    def in_descriptors(self) -> DescriptorDict:
        return {
            "a": FortranField(io_name="A", random_value_range=(-1000, 1000), **IJK_ARGS()),
            "b": FortranField(io_name="B", random_value_range=(-1000, 1000), **IJK_ARGS()),
        }

    @property
    def out_descriptors(self) -> DescriptorDict:
        return {
            "c1": FortranField(**IJK_ARGS()),
            "c2": FortranField(**IJK_ARGS()),
            "c3": FortranField(**IJK_ARGS()),
            "c4": FortranField(**IJK_ARGS()),
            "c5": FortranField(**IJK_ARGS()),
            "c6": FortranField(**IJK_ARGS()),
            "c7": FortranField(**IJK_ARGS(), io_name_write="c7_write"),
        }
