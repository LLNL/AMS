# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# AMSLib Project Developers
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import numpy as np
from ams.action import UserAction


class RandomPruneAction(UserAction):
    """
    A class that will be dynamically loaded by the AMS staging mechanism
    The class must be callable.
    """

    def __init__(self, drop_rate):
        super().__init__()
        self.drop_rate = drop_rate

    def data_cb(self, inputs, outputs):
        if len(inputs) == 0:
            return

        randIndexes = np.random.randint(inputs.shape[0], size=int(self.drop_rate * inputs.shape[0]))
        pruned_inputs = inputs[randIndexes]
        pruned_outputs = outputs[randIndexes]
        return pruned_inputs, pruned_outputs

    def update_model_cb(self, domain, model):
        pass

    @staticmethod
    def add_cli_args(arg_parser):
        """
        Must be implemented as it is an 'abstractstaticmethod' on the parent class.

        Args:
            arg_parser: argparge parser to append command line arguments
        """
        arg_parser.add_argument("--fraction", "-f", help="The fraction of elements to drop", required=True, type=float)

    @classmethod
    def from_cli(cls, args):
        """
        Must be impelemented as it is an 'abstractclassmethod' on the parent class

        Args:
            args: The parsed arguments of the argparser.

        Returns: An instance of the cls class initialized with the CLI arguments
        """
        return cls(args.fraction)
