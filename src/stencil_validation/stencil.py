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

from stencil_validation.descriptors import concretize, to_file
from stencil_validation.iox import io_file_operator

if TYPE_CHECKING:
    from typing import Optional

    from stencil_validation.config import Config
    from stencil_validation.descriptors import ConcretizedDescriptorDict, DescriptorDict


def get_stencil_id(name: str, version: str) -> str:
    return f"{name}_{version}" if name != "" and version != "" else ""


class MetaStencil(type):
    COLLECTION: dict[str, type]

    def __new__(cls, cls_name: str, bases: tuple[type, ...], dct: dict) -> type:
        name = dct.get("name", "")
        version = dct.get("version", "")
        stencil_id = get_stencil_id(name, version)
        if cls.COLLECTION is not None:
            if stencil_id in cls.COLLECTION:
                raise KeyError(f"Two stencils registered under `{stencil_id}`.")
            out = super().__new__(cls, cls_name, bases, dct)
            if stencil_id != "":
                cls.COLLECTION[stencil_id] = out
            return out


class Stencil:
    name: str = ""
    version: str = ""

    config: Config

    def __init__(self, config: Config) -> None:
        self.config = config

    @property
    def in_descriptors(self) -> DescriptorDict:
        return {}

    @property
    def out_descriptors(self) -> DescriptorDict:
        return {}

    @staticmethod
    def inject_io_name(desc_dict: DescriptorDict) -> DescriptorDict:
        for key, desc in desc_dict.items():
            io_name = desc.io_name or key
            desc_dict[key] = desc.with_attrs(io_name=io_name)
        return desc_dict

    def read_args(
        self, desc_dict: DescriptorDict, file_path: Optional[str]
    ) -> ConcretizedDescriptorDict:
        desc_dict = self.inject_io_name(desc_dict)
        with io_file_operator(file_path, mode="r") as file_op:
            cdesc_dict = concretize(desc_dict, self.config, file_op)
        return cdesc_dict

    def write_args(self, cdesc_dict: ConcretizedDescriptorDict, file_path: Optional[str]) -> None:
        with io_file_operator(file_path, mode="w") as file_op:
            to_file(cdesc_dict, self.config, file_op)
