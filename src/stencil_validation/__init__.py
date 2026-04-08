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

from stencil_validation.config import Config
from stencil_validation.descriptors import (
    BaseField,
    Bool,
    CompositeField,
    ConcretizedDescriptor,
    ConcretizedDescriptorDict,
    Descriptor,
    DescriptorDict,
    Field,
    Float,
    Int,
    concretize,
    to_file,
)
from stencil_validation.dims import Dim, IndexedDim, SizedDim
from stencil_validation.iox import HDF5Operator, IOFileOperator, NetCDFOperator, io_file_operator
from stencil_validation.stencil_fortran.descriptors import CompositeFortranField, FortranField
from stencil_validation.stencil_fortran.stencil import (
    FortranStencil,
    MetaFortranStencil,
    get_fortran_stencil,
    print_fortran_stencil_list,
)
from stencil_validation.stencil_gt4py.descriptors import CompositeGT4PyField, GT4PyField
from stencil_validation.stencil_gt4py.stencil import (
    GT4PyStencil,
    MetaGT4PyStencil,
    get_gt4py_stencil,
    print_gt4py_stencil_list,
)

__version__ = "0.1.0.dev"

__all__ = [
    "BaseField",
    "Bool",
    "CompositeField",
    "CompositeFortranField",
    "CompositeGT4PyField",
    "ConcretizedDescriptor",
    "ConcretizedDescriptorDict",
    "Config",
    "Descriptor",
    "DescriptorDict",
    "Dim",
    "Field",
    "Float",
    "FortranField",
    "FortranStencil",
    "GT4PyField",
    "GT4PyStencil",
    "HDF5Operator",
    "IOFileOperator",
    "IndexedDim",
    "Int",
    "MetaFortranStencil",
    "MetaGT4PyStencil",
    "NetCDFOperator",
    "SizedDim",
    "__version__",
    "concretize",
    "get_fortran_stencil",
    "get_gt4py_stencil",
    "io_file_operator",
    "print_fortran_stencil_list",
    "print_gt4py_stencil_list",
    "to_file",
]
