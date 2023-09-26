import numpy as np

from ams.action import UserAction


class RandomPruneAction(UserAction):
    def __init__(self, drop_rate):
        super().__init__()
        self.drop_rate = drop_rate

    def __call__(self, inputs, outputs):
        if len(inputs) == 0:
            return

        # randIndexes = np.random.randint(inputs.shape[0], size=int(self.drop_rate * inputs.shape[0]))
        pruned_inputs = inputs  # [randIndexes
        pruned_outputs = outputs  # [randIndexes]
        return pruned_inputs, pruned_outputs

    @staticmethod
    def add_cli_args(arg_parser):
        arg_parser.add_argument("--fraction", "-f", help="The fraction of elememnts to drop", required=True)

    @classmethod
    def from_cli(cls, args):
        return cls(args.fraction)
