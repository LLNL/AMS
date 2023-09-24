#!/usr/bin/env python3
# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# AMSLib Project Developers
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import inspect
import sys
from importlib import import_module
from pathlib import Path


def load_class(path_to_module, class_name, requires=list()):
    """
    Load a user defined class from a user defined module located at path_to_module
    Args:
        path_to_module: path to the user-defined module implementing several classes
        class_name: The name of the class to be loaded

    Returns:
        The python-datatype of the class to be returned
    """
    _path = Path(path_to_module).resolve()
    _dir = _path.parent
    sys.path.append(str(_dir))
    _module = import_module(_path.stem)

    for name, objs in inspect.getmembers(_module):
        print(name, objs)
        if inspect.isclass(objs) and name == class_name:
            return getattr(_module, name)

    raise ImportError(f"Could not find class {class_name} in {path_to_module}")
