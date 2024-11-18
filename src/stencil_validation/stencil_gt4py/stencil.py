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
from typing import TYPE_CHECKING

from ifs_physics_common.framework.stencil import compile_stencil, stencil_collection
from stencil_validation.descriptors import concretize

from stencil_validation.stencil import MetaStencil, Stencil, get_stencil_id

if TYPE_CHECKING:
    from types import FunctionType
    from typing import Optional

    from gt4py.cartesian.stencil_object import StencilObject

    from stencil_validation.config import Config
    from stencil_validation.descriptors import ConcretizedDescriptorDict, DescriptorDict


GT4PY_STENCIL_COLLECTION: dict[str, "MetaGT4PyStencil"] = {}


class MetaGT4PyStencil(MetaStencil):
    COLLECTION: dict[str, MetaGT4PyStencil] = GT4PY_STENCIL_COLLECTION  # type: ignore[assignment]


class GT4PyStencil(Stencil, metaclass=MetaGT4PyStencil):
    def_func: FunctionType
    external_info: dict[str, dict] = {}

    stencil_obj: StencilObject

    def __init__(self, config: Config, externals: Optional[dict] = None) -> None:
        super().__init__(config)

        externals = externals or {}
        for ext_name, ext_info in self.external_info.items():
            if "same_as" in ext_info:
                trg_name = ext_info["same_as"]
                if trg_name not in externals:
                    raise RuntimeError(
                        f"Value for the external symbol `{trg_name}` not found to initialize the "
                        f"external symbol `{ext_name}`."
                    )
                externals[ext_name] = externals[trg_name]
            else:
                if "type" not in ext_info:
                    raise RuntimeError(f"No type specified for external symbol `{ext_name}`.")
                ext_type = ext_info["type"]
                if ext_name in externals:
                    externals[ext_name] = ext_type(externals[ext_name])
                elif "default" in ext_info:
                    externals[ext_name] = ext_type(ext_info["default"])
                else:
                    raise RuntimeError(f"No value specified for external symbol `{ext_name}`.")

        stencil_id = get_stencil_id(self.name, self.version)
        stencil_collection(stencil_id)(self.def_func.__func__)
        self.stencil_obj = compile_stencil(stencil_id, self.config.gt4py_config, externals)

    @property
    def inout_descriptors(self) -> DescriptorDict:
        return {}

    @property
    def tmp_descriptors(self) -> DescriptorDict:
        return {}

    @property
    @abstractmethod
    def origin(self) -> tuple[int, int, int]: ...

    @property
    @abstractmethod
    def domain(self) -> tuple[int, int, int]: ...

    def process_cdesc_dicts(
        self,
        in_cdesc_dict: ConcretizedDescriptorDict,
        inout_cdesc_dict: ConcretizedDescriptorDict,
        out_cdesc_dict: ConcretizedDescriptorDict,
        tmp_cdesc_dict: ConcretizedDescriptorDict,
    ) -> None:
        pass

    def __call__(
        self,
        in_file_path: Optional[str] = None,
        write_in_file_path: Optional[str] = None,
        out_file_path: Optional[str] = None,
    ) -> None:
        in_cdesc_dict = self.read_args(self.in_descriptors, in_file_path)
        inout_cdesc_dict = self.read_args(self.inout_descriptors, in_file_path)
        self.write_args({**in_cdesc_dict, **inout_cdesc_dict}, write_in_file_path)
        out_cdesc_dict = concretize(self.out_descriptors, self.config)
        tmp_cdesc_dict = concretize(self.tmp_descriptors, self.config)
        self.process_cdesc_dicts(in_cdesc_dict, inout_cdesc_dict, out_cdesc_dict, tmp_cdesc_dict)

        self.stencil_obj(
            **{key: cdesc.value for key, cdesc in in_cdesc_dict.items()},
            **{key: cdesc.value for key, cdesc in inout_cdesc_dict.items()},
            **{key: cdesc.value for key, cdesc in out_cdesc_dict.items()},
            **{key: cdesc.value for key, cdesc in tmp_cdesc_dict.items()},
            origin=self.origin,
            domain=self.domain,
            exec_info=self.config.gt4py_config.exec_info,
            validate_args=self.config.gt4py_config.validate_args,
        )

        self.write_args({**inout_cdesc_dict, **out_cdesc_dict}, out_file_path)


def get_gt4py_stencil(
    name: str, version: str, config: Config, externals: Optional[dict] = None
) -> GT4PyStencil:
    return GT4PY_STENCIL_COLLECTION[get_stencil_id(name, version)](config, externals)  # type: ignore[no-any-return]
