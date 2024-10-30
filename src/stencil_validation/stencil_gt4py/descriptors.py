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

from gt4py.storage import from_array

from stencil_validation.descriptors import Field

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from typing import Optional

    from stencil_validation.config import Config
    from stencil_validation.iox import IOFileOperator


class GT4PyField(Field):
    def get_random_value(self, config: Config) -> NDArray:
        value = super().get_random_value(config)
        return from_array(value, dtype=self.get_dtype(config), backend=config.gt4py_config.backend)

    def read_value(self, config: Config, io_file_op: IOFileOperator) -> Optional[NDArray]:
        value = super().read_value(config, io_file_op)
        if value is not None:
            return from_array(
                value, dtype=self.get_dtype(config), backend=config.gt4py_config.backend
            )
        else:
            return value
