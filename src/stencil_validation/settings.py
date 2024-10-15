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
from dataclasses import dataclass
import os


@dataclass(frozen=False)
class GlobalSettings:
    project_root: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    verbose: bool = False

    def with_verbosity(self, verbose: bool) -> GlobalSettings:
        self.verbose = verbose
        return self


GLOBAL_SETTINGS = GlobalSettings()
