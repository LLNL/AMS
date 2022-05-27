#!/bin/bash

clang-format --style='{IndentWidth: 4, ColumnLimit: 100}' -i src/mmp.* src/eos_constant_on_host.hpp
