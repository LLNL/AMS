#!/usr/bin/env python3
# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# AMSLib Project Developers
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
from pprint import pprint

from ams.AMSStore import AMSDataStore
from ams.config import AMSInstance


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--action", "-a", choices=["add", "remove", "print"], required=True)
    parser.add_argument("--entry-type", "-e", choices=["data", "candidate", "model"], required=True)
    parser.add_argument("--version", "-v", help="version to assign to data file", type=int)
    parser.add_argument("filename")
    args = parser.parse_args()
    ams_config = AMSInstance()

    with AMSDataStore(ams_config.db_path, ams_config.db_store, ams_config.name, False) as store:
        if args.action == "add":
            if args.entry_type == "data":
                store.add_data([args.filename], args.version)
            elif args.entry_type == "candidate":
                store.add_candidates([args.filename], args.version)
            elif args.entry_type == "model":
                store.add_model(args.filename, args.version)
        elif args.action == "remove":
            if args.entry_type == "data":
                store.remove_data([args.filename])
            elif args.entry_type == "candidate":
                store.remove_candidates([args.filename])
            elif args.entry_type == "model":
                store.remove_models([args.filename])
        elif args.action == "print":
            pprint(store.get_raw_content())


if __name__ == "__main__":
    main()
