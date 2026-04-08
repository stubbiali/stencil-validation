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

import hashlib
import os
import shutil
from typing import TYPE_CHECKING

import fmodpy
import jinja2

from stencil_validation.settings import GLOBAL_SETTINGS
from stencil_validation.stencil_fortran.settings import (
    FMODPY_BUILD_CACHE,
    FMODPY_CACHE,
    FMODPY_SRC_CACHE,
)

if TYPE_CHECKING:
    from types import ModuleType
    from typing import Any, Literal, Optional


def get_cache_id(subroutine_id: str, src: str) -> str:
    h = hashlib.blake2b(digest_size=8)
    h.update(src.encode())
    hash_id = h.hexdigest()
    return f"{subroutine_id}_{hash_id}"


def render_subroutine_template(
    subroutine_id: str,
    template_file_path: str,
    template_var_info: dict[str, dict[str, Any]],
    template_var_values: dict[str, Any],
) -> tuple[str, str]:
    f_path = os.path.abspath(template_file_path)
    if not os.path.exists(f_path):
        raise RuntimeError(f"The template file `{template_file_path}` does not exist.")

    data = {}
    for var_name, var_info in template_var_info.items():
        if "type" not in var_info:
            raise RuntimeError(f"No type specified for template variable `{var_name}`.")
        var_type = var_info["type"]
        if var_name in template_var_values:
            data[var_name] = var_type(template_var_values[var_name])
        elif "default" in var_info:
            data[var_name] = var_type(var_info["default"])
        else:
            raise RuntimeError(f"No value specified for template variable `{var_name}`.")

    search_path, f_fullname = f_path.rsplit("/", maxsplit=1)
    f_name, f_ext = f_fullname.rsplit(".", maxsplit=1)
    if f_ext != "in":
        raise RuntimeError("The extension of the template file should be `.in`.")

    loader = jinja2.FileSystemLoader(search_path)
    env = jinja2.Environment(loader=loader)
    template = env.get_template(f_fullname)
    src = template.render(**data)

    cache_id = get_cache_id(subroutine_id, src)
    src_dir = os.path.join(FMODPY_SRC_CACHE, cache_id)
    os.makedirs(src_dir, exist_ok=True)
    src_file_path = os.path.join(src_dir, f_name)
    with open(src_file_path, "w") as f:
        f.write(src)

    return src_file_path, cache_id


def compile_subroutine(
    src_file_path: str,
    cache_id: str,
    compiler_args: Optional[list[str]] = None,
    opt_level: Literal[0, 1, 2, 3] = 3,
    rebuild: bool = False,
) -> ModuleType:
    module: ModuleType = fmodpy.fimport(
        src_file_path,
        f_compiler_args=compiler_args or [],
        build_dir=os.path.join(FMODPY_BUILD_CACHE, cache_id),
        output_dir=os.path.join(FMODPY_CACHE, cache_id),
        optimization_level=f"-O{opt_level}",
        rebuild=rebuild,
        verbose=GLOBAL_SETTINGS.verbose,
    )
    shutil.rmtree(FMODPY_BUILD_CACHE, ignore_errors=False)
    return module
