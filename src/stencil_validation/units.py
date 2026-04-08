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

import re

import pint

UNIT_REGISTRY = pint.UnitRegistry()


def parse_units(units: str) -> str:
    """
    In NetCDF files provided by ECMWF, units with exponents are expressed as `m2` or `m-2`.
    This utility converts the above examples into the format `m**(2)` and `m**(-2)` used by pint.
    """

    pattern = r"(?<=[A-Za-z])(-*\d+)"

    def replace_exponent(match: re.Match) -> str:
        return f"**({match.group(1)})"

    return re.sub(pattern, replace_exponent, units)


def get_conversion_factor(src_units: str, trg_units: str) -> float:
    src = UNIT_REGISTRY(parse_units(src_units))
    trg = UNIT_REGISTRY(parse_units(trg_units))
    return src.to(trg).magnitude  # type: ignore[no-any-return]
