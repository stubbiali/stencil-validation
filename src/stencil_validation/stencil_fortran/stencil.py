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

from typing import TYPE_CHECKING

import numpy as np

import ifs_physics_common

from stencil_validation.descriptors import ConcretizedDescriptor
from stencil_validation.stencil import MetaStencil, Stencil, get_stencil_id, print_stencil_list
from stencil_validation.stencil_fortran.utils import compile_subroutine, render_subroutine_template

if TYPE_CHECKING:
    from collections.abc import Mapping
    from types import FunctionType
    from typing import Any, ClassVar, Literal

    from stencil_validation.config import Config
    from stencil_validation.descriptors import ConcretizedDescriptorDict


FORTRAN_STENCIL_COLLECTION: dict[str, "MetaFortranStencil"] = {}


class MetaFortranStencil(MetaStencil):
    COLLECTION: Mapping[str, MetaFortranStencil] = FORTRAN_STENCIL_COLLECTION


class FortranStencil(Stencil, metaclass=MetaFortranStencil):
    template_file_path: str = ""
    template_var_info: ClassVar[dict[str, dict]] = {}

    def __call__(
        self,
        template_var_values: dict[str, Any] | None = None,
        in_file_paths: tuple[str, ...] | None = None,
        write_in_file_path: str | None = None,
        out_file_path: str | None = None,
        include_dirs: list[str] | None = None,
        opt_level: Literal[0, 1, 2, 3] = 3,
        rebuild: bool = False,
        num_runs: int | None = None,
    ) -> None:
        fn = self.compile(template_var_values or {}, include_dirs, opt_level, rebuild)
        in_cdesc_dict = self.read_args(self.in_descriptors, in_file_paths)
        self.write_args(in_cdesc_dict, write_in_file_path)
        out_cdesc_dict = self.run(fn, in_cdesc_dict, num_runs)
        self.write_args(out_cdesc_dict, out_file_path)

    def compile(
        self,
        template_var_values: dict[str, Any],
        compiler_args: list[str] | None = None,
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
        return fn  # type: ignore[no-any-return]

    def run(
        self, fn: FunctionType, in_cdesc_dict: ConcretizedDescriptorDict, num_runs: int | None
    ) -> ConcretizedDescriptorDict:
        in_args = {key: cdesc.value for key, cdesc in in_cdesc_dict.items()}
        out_args = fn(**in_args)
        out_args = [out_args] if isinstance(out_args, np.ndarray) else out_args

        if len(out_args) != len(self.out_descriptors):
            raise RuntimeError(
                f"Expecting {len(self.out_descriptors)} outs, but got {len(out_args)}."
            )

        out_desc_dict = self.inject_io_name(self.out_descriptors)
        out_cdesc_dict = {
            key: ConcretizedDescriptor(desc, value)
            for (key, desc), value in zip(out_desc_dict.items(), out_args)
        }

        num_runs = num_runs or 0
        if num_runs > 0:
            with ifs_physics_common.timing(self.name) as timer:
                for _ in range(num_runs):
                    _ = fn(**in_args)
            print(
                f"Average execution time over {num_runs} runs: "
                f"{timer.get_time(self.name, units='ms') / num_runs:.3f} ms."
            )

        return out_cdesc_dict


def get_fortran_stencil(name: str, version: str, config: Config) -> FortranStencil:
    return FORTRAN_STENCIL_COLLECTION[get_stencil_id(name, version)](config)  # type: ignore[no-any-return]


def print_fortran_stencil_list() -> None:
    print_stencil_list(FORTRAN_STENCIL_COLLECTION)
