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

from stencil_validation.descriptors import Field

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from typing import Optional

    from stencil_validation.config import Config
    from stencil_validation.iox import IOFileOperator


class FortranField(Field):
    def get_random_value(self, config: Config) -> NDArray:
        value = super().get_random_value(config)
        return np.asfortranarray(value)

    def get_value_from_file(self, config: Config, io_file_op: IOFileOperator) -> Optional[NDArray]:
        value = super().get_value_from_file(config, io_file_op)
        if value is not None:
            return np.asfortranarray(value)
        else:
            return value
