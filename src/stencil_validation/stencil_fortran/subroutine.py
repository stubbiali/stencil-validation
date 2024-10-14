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
from functools import cached_property
import numpy as np
from typing import TYPE_CHECKING

from stencil_validation.descriptors import (
    ConcretizedDescriptor,
    concretize,
    inject_io_name,
    to_file,
)
from stencil_validation.iox import io_file_operator
from stencil_validation.stencil_fortran.utils import render_subroutine_template, compile_subroutine

if TYPE_CHECKING:
    from types import FunctionType
    from typing import Any, Literal, Optional

    from stencil_validation.config import Config
    from stencil_validation.descriptors import ConcretizedDescriptorDict, DescriptorDict


FORTRAN_SUBROUTINE_COLLECTION: dict[str, "MetaFortranSubroutine"] = {}


def get_subroutine_id(version: str, name: str) -> str:
    return f"{version}_{name}"


class MetaFortranSubroutine(type):
    def __new__(cls, cls_name, bases, dct):
        version = dct.get("version", "")
        name = dct.get("name", "")
        assert not (version == "" and name == "" and len(bases) > 0)
        subroutine_id = get_subroutine_id(version, name)
        if subroutine_id in FORTRAN_SUBROUTINE_COLLECTION:
            raise KeyError(f"Two Fortran subroutines registered under `{subroutine_id}`.")
        out = super().__new__(cls, cls_name, bases, dct)
        if subroutine_id != "_":
            FORTRAN_SUBROUTINE_COLLECTION[subroutine_id] = out
        return out


class FortranSubroutine(metaclass=MetaFortranSubroutine):
    version: str = ""
    name: str = ""
    template_file_path: str = ""
    template_var_info: dict[str, dict[str, Any]] = {}

    @cached_property
    def subroutine_id(self) -> str:
        return self.version + "_" + self.name

    @abstractmethod
    @property
    def input_descriptors(self) -> DescriptorDict: ...

    @abstractmethod
    @property
    def output_descriptors(self) -> DescriptorDict: ...

    def __call__(
        self,
        config: Config,
        template_var_values: Optional[dict[str, Any]] = None,
        input_file_path: Optional[str] = None,
        output_file_path: Optional[str] = None,
        overwrite_input_file: bool = False,
        include_dirs: Optional[list[str]] = None,
        opt_level: Literal[0, 1, 2, 3] = 3,
        rebuild: bool = False,
    ) -> None:
        fn = self.compile(template_var_values or {}, include_dirs, opt_level, rebuild)
        in_cdesc_dict = self.get_in_args(config, input_file_path, overwrite_input_file)
        out_cdesc_dict = self.run(fn, in_cdesc_dict)
        self.write_output_to_file(config, output_file_path, out_cdesc_dict)

    def compile(
        self,
        template_var_values: dict[str, Any],
        include_dirs: Optional[list[str]] = None,
        opt_level: Literal[0, 1, 2, 3] = 3,
        rebuild: bool = False,
    ) -> FunctionType:
        src_file_path, cache_id = render_subroutine_template(
            self.subroutine_id,
            self.template_file_path,
            self.template_var_info,
            template_var_values or {},
        )
        module = compile_subroutine(
            src_file_path, cache_id, include_dirs=include_dirs, opt_level=opt_level, rebuild=rebuild
        )
        fn = getattr(module, self.name, None)
        if fn is None:
            raise RuntimeError(f"Subroutine `{self.name}` not defined in `{src_file_path}`.")
        return fn

    def get_in_args(
        self, config: Config, input_file_path: Optional[str], overwrite_input_file: bool
    ) -> ConcretizedDescriptorDict:
        in_desc_dict = inject_io_name(self.input_descriptors)
        with io_file_operator(input_file_path, mode="r") as in_file_op:
            in_cdesc_dict = concretize(in_desc_dict, config, in_file_op)
            overwrite_input_file = overwrite_input_file or (in_file_op is None)
        if overwrite_input_file:
            with io_file_operator(input_file_path, mode="w") as ow_in_file_op:
                to_file(in_cdesc_dict, config, ow_in_file_op)
        return in_cdesc_dict

    def run(
        self, fn: FunctionType, in_cdesc_dict: ConcretizedDescriptorDict
    ) -> ConcretizedDescriptorDict:
        in_args = {key: cdesc.value for key, cdesc in in_cdesc_dict.items()}
        out_args = fn(**in_args)
        out_args = [out_args] if isinstance(out_args, np.ndarray) else out_args

        if len(out_args) != len(self.output_descriptors):
            raise RuntimeError(
                f"Expecting {len(self.output_descriptors)} outputs, but got {len(out_args)}."
            )

        return {
            key: ConcretizedDescriptor(desc, value)
            for (key, desc), value in zip(self.output_descriptors.items(), out_args)
        }

    def write_output_to_file(
        self,
        config: Config,
        output_file_path: Optional[str],
        out_cdesc_dict: ConcretizedDescriptorDict,
    ) -> None:
        with io_file_operator(output_file_path, mode="w") as out_file_op:
            to_file(out_cdesc_dict, config, out_file_op)


def get_subroutine(version: str, name: str) -> FortranSubroutine:
    return FORTRAN_SUBROUTINE_COLLECTION[get_subroutine_id(version, name)]()
