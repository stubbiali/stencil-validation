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
import numpy as np
from typing import TYPE_CHECKING

from stencil_validation.descriptors import ConcretizedDescriptor
from stencil_validation.stencil import MetaStencil, Stencil, get_stencil_id
from stencil_validation.stencil_fortran.utils import render_subroutine_template, compile_subroutine

if TYPE_CHECKING:
    from types import FunctionType
    from typing import Any, Literal, Optional

    from stencil_validation.config import Config
    from stencil_validation.descriptors import ConcretizedDescriptorDict


FORTRAN_STENCIL_COLLECTION: dict[str, "MetaFortranStencil"] = {}


class MetaFortranStencil(MetaStencil):
    COLLECTION: dict[str, MetaFortranStencil] = FORTRAN_STENCIL_COLLECTION


class FortranStencil(Stencil, metaclass=MetaFortranStencil):
    template_file_path: str = ""
    template_var_info: dict[str, dict] = {}

    def __call__(
        self,
        template_var_values: Optional[dict[str, Any]] = None,
        in_file_path: Optional[str] = None,
        out_file_path: Optional[str] = None,
        overwrite_in_file: bool = False,
        include_dirs: Optional[list[str]] = None,
        opt_level: Literal[0, 1, 2, 3] = 3,
        rebuild: bool = False,
    ) -> None:
        fn = self.compile(template_var_values or {}, include_dirs, opt_level, rebuild)
        in_cdesc_dict = self.read_args(self.in_descriptors, in_file_path)
        self.write_args(in_cdesc_dict, in_file_path, overwrite_in_file)
        out_cdesc_dict = self.run(fn, in_cdesc_dict)
        self.write_args(out_cdesc_dict, out_file_path, overwrite_file=True)

    def compile(
        self,
        template_var_values: dict[str, Any],
        compiler_args: Optional[list[str]] = None,
        opt_level: Literal[0, 1, 2, 3] = 3,
        rebuild: bool = False,
    ) -> FunctionType:
        src_file_path, cache_id = render_subroutine_template(
            get_stencil_id(self.name, self.version),
            self.template_file_path,
            self.template_var_info,
            template_var_values or {},
        )
        module = compile_subroutine(
            src_file_path,
            cache_id,
            compiler_args=compiler_args,
            opt_level=opt_level,
            rebuild=rebuild,
        )
        fn = getattr(module, self.name, None)
        if fn is None:
            raise RuntimeError(f"Subroutine `{self.name}` not defined in `{src_file_path}`.")
        return fn

    def run(
        self, fn: FunctionType, in_cdesc_dict: ConcretizedDescriptorDict
    ) -> ConcretizedDescriptorDict:
        in_args = {key: cdesc.value for key, cdesc in in_cdesc_dict.items()}
        out_args = fn(**in_args)
        out_args = [out_args] if isinstance(out_args, np.ndarray) else out_args

        if len(out_args) != len(self.out_descriptors):
            raise RuntimeError(
                f"Expecting {len(self.out_descriptors)} outs, but got {len(out_args)}."
            )

        out_desc_dict = self.inject_io_name(self.out_descriptors)
        return {
            key: ConcretizedDescriptor(desc, value)
            for (key, desc), value in zip(out_desc_dict.items(), out_args)
        }


def get_fortran_stencil(name: str, version: str, config: Config) -> FortranStencil:
    return FORTRAN_STENCIL_COLLECTION[get_stencil_id(name, version)](config)
