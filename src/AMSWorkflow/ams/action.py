#!/usr/bin/env python3
# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# AMSLib Project Developers
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from abc import ABC, abstractmethod


class UserAction(ABC):
    """
    An abstract class used as an interface from users
    to describe actions to be peformed in the data-pipeline
    of the AMS staging mechanism
    """

    def __init__(self):
        pass

    @abstractmethod
    def data_cb(self, inputs, outputs):
        pass

    @abstractmethod
    def update_model_cb(self, domain, model):
        pass

    @staticmethod
    @abstractmethod
    def add_cli_args(parser):
        """ """
        pass

    @classmethod
    @abstractmethod
    def from_cli(cls, args):
        pass
