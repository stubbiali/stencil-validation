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
import os
import pytest
import tempfile
from typing import TYPE_CHECKING
import uuid

from stencil_validation.config import Config
from stencil_validation.descriptors import concretize, to_file
from stencil_validation.dims import Dim, IJ, K
from stencil_validation.iox import io_file_operator
from stencil_validation.stencil_fortran.descriptors import CompositeFortranField, FortranField
from stencil_validation.stencil_gt4py.descriptors import CompositeGT4PyField, GT4PyField

if TYPE_CHECKING:
    from stencil_validation.descriptors import CompositeField, Field


D2 = Dim("D2")


@pytest.mark.parametrize("composite_field_cls", [CompositeFortranField, CompositeGT4PyField])
def test_invalid_class(composite_field_cls: type[CompositeField]):
    with pytest.raises(AssertionError):
        _ = composite_field_cls(
            dims=(IJ, K, D2),
            fields_map={
                (IJ, K, D2[0]): FortranField(dims=(IJ, K)),
                (IJ, K, D2[1]): GT4PyField(dims=(IJ, K)),
            },
        )


@pytest.mark.parametrize(
    ["field_cls", "composite_field_cls"],
    [[FortranField, CompositeFortranField], [GT4PyField, CompositeGT4PyField]],
)
def test_invalid_dim(field_cls: type[Field], composite_field_cls: type[CompositeField]):
    with pytest.raises(AssertionError):
        _ = composite_field_cls(
            dims=(IJ, K, D2),
            fields_map={(IJ, K, D2[0]): field_cls(dims=(IJ, K)), (IJ, K): field_cls(dims=(IJ, K))},
        )


@pytest.mark.parametrize(
    ["field_cls", "composite_field_cls"],
    [[FortranField, CompositeFortranField], [GT4PyField, CompositeGT4PyField]],
)
def test_invalid_field_dim(field_cls, composite_field_cls):
    with pytest.raises(ValueError):
        config = Config(grid_shape={"IJ": 16, "K": 8}, data_shape={"D2": 2})
        desc = composite_field_cls(
            dims=(IJ, K, D2),
            fields_map={
                (IJ, K, D2[0]): field_cls(dims=(IJ, K)),
                (IJ, K, D2[1]): field_cls(dims=(IJ,)),
            },
        )
        desc.concretize(config)


@pytest.mark.parametrize(
    ["field_cls", "composite_field_cls"],
    [[FortranField, CompositeFortranField], [GT4PyField, CompositeGT4PyField]],
)
def test(field_cls: type[Field], composite_field_cls: type[CompositeField]):
    with tempfile.TemporaryDirectory() as tmpdir:
        config = Config(grid_shape={"IJ": 16, "K": 8}, data_shape={"D2": 2})

        field0 = field_cls(dims=(IJ, K), io_name="A")
        field1 = field_cls(dims=(IJ, K), io_name="B")
        desc_dict = {"field0": field0, "field1": field1}
        cdesc_dict = concretize(desc_dict, config)

        file_path = os.path.join(tmpdir, uuid.uuid4().hex + ".nc")
        with io_file_operator(file_path, mode="w") as io_file_op:
            to_file(cdesc_dict, config, io_file_op)

        composite_field = composite_field_cls(
            dims=(IJ, K, D2),
            fields_map={(IJ, K, D2[0].squeeze()): field0, (IJ, K, D2[1].squeeze()): field1},
        )
        with io_file_operator(file_path, mode="r") as io_file_op:
            composite_value = composite_field.concretize(config, io_file_op).value

    np.testing.assert_allclose(composite_value[..., 0], cdesc_dict["field0"].value)
    np.testing.assert_allclose(composite_value[..., 1], cdesc_dict["field1"].value)


if __name__ == "__main__":
    pytest.main([__file__])
