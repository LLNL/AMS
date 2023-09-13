# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# AMSLib Project Developers
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
from pathlib import Path

from ams.config import AMSInstance
from ams.stage import Stage
from ams.store import AMSDataStore

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--src", "-s",  required=True)
    args = parser.parse_args()
    ams_config = AMSInstance()
    store = AMSDataStore(ams_config.db_path, ams_config.db_store, ams_config.name, False)
    stage = Stage(store, ams_config.db_path,
                  ams_config.do_stage, ams_config.stage_dir,
                  ams_config.stage_policy, ams_config.stage_src, args.src)
    Stage()


if __name__ == "__main__":
    main()
