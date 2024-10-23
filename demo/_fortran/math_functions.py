# -*- coding: utf-8 -*-
from __future__ import annotations
import os
from typing import TYPE_CHECKING

from stencil_validation.dims import IJ, K
from stencil_validation.stencil_fortran.descriptors import FortranField
from stencil_validation.stencil_fortran.stencil import FortranStencil, MetaFortranStencil

if TYPE_CHECKING:
    from stencil_validation.descriptors import DescriptorDict


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
            "a": FortranField(dims=(IJ, K), dtype_name="float", random_value_range=(-1000, 1000)),
            "b": FortranField(dims=(IJ, K), dtype_name="float", random_value_range=(-1000, 1000)),
        }

    @property
    def out_descriptors(self) -> DescriptorDict:
        return {
            "c1": FortranField(dims=(IJ, K), dtype_name="float"),
            "c2": FortranField(dims=(IJ, K), dtype_name="float"),
            "c3": FortranField(dims=(IJ, K), dtype_name="float"),
            "c4": FortranField(dims=(IJ, K), dtype_name="float"),
            "c5": FortranField(dims=(IJ, K), dtype_name="float"),
            "c6": FortranField(dims=(IJ, K), dtype_name="float"),
            "c7": FortranField(dims=(IJ, K), dtype_name="float"),
        }
