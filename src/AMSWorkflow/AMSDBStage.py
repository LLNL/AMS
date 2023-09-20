# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# AMSLib Project Developers
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
import time

from ams.config import AMSInstance
from ams.stage import DataBlobAction, get_pipeline
from ams.store import create_store_directories


class RandomPruneAction(DataBlobAction):
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


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", "-c", default=None)
    parser.add_argument("--src", "-s", required=True)
    parser.add_argument("--pattern", "-p", required=True)
    parser.add_argument("--mechansism", "-m", choices=["fs", "network"], default="fs")
    # We will need to implement this as a nested subparser
    parser.add_argument(
        "--type",
        "-t",
        choices=["network", "fs"],
    )
    parser.add_argument("--stage-dir", default=None)
    parser.add_argument("--policy", choices=["process", "thread", "sequential"], default="process")
    parser.add_argument("--fraction", "-f", type=float, default=1.0)
    args = parser.parse_args()
    ams_config = AMSInstance()
    create_store_directories(ams_config.db_path)
    # Create AMS Kosh wrapper
    # The AMSPipeline represents a series of actions to be performed
    # from reading the data until storing them
    # into the candidates data base
    pipeline = get_pipeline(args.mechansism)(args.src, args.pattern)

    # Add an action to be performed before pushing to the candidates store
    pipeline.add_data_action(RandomPruneAction(args.fraction))

    # If we want to 'stage' data to local storage before moving them to the PFS we can do so
    if args.stage_dir:
        pipeline.enable_stage(args.stage_dir)

    start = time.time()
    pipeline.execute(args.policy)
    end = time.time()
    print(f"End to End time spend : {end - start}")


if __name__ == "__main__":
    main()
