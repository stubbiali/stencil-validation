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
from typing import Optional

from stencil_validation.settings import GLOBAL_SETTINGS


ANSI_ESCAPE_SEQUENCES = {
    "end": "\033[0m",
    "style": {"bold": "\033[1m", "italic": "\033[3m"},
    "colors": {"grey": "\033[90m", "red": "\033[91m", "green": "\033[92m"},
}


def printx(msg: str, end: Optional[str] = None, flush: bool = False, color: str = None) -> None:
    if GLOBAL_SETTINGS.verbose:
        if color in ANSI_ESCAPE_SEQUENCES["colors"]:
            msg = ANSI_ESCAPE_SEQUENCES["colors"][color] + msg + ANSI_ESCAPE_SEQUENCES["end"]
        print(msg, end=end, flush=flush)
