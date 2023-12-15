#!/usr/bin/env python3
# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# AMSLib Project Developers
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse

from ams.store import AMSDataStore, AddToStore, RemoveFromStore, SearchStore, CreateStore

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    subparsers = parser.add_subparsers(dest="command", help="Commands to the AMS database")

    add_parser = subparsers.add_parser("add", help="Add information to the AMS database")
    AddToStore.add_cli_options(add_parser)

    delete_parser = subparsers.add_parser("remove", help="Remove/Delete entry from the AMS database")
    RemoveFromStore.add_cli_options(delete_parser)

    query_parser = subparsers.add_parser("query", help="Query AMS database")
    SearchStore.add_cli_options(query_parser)

    create_parser = subparsers.add_parser("create", help="Create an AMS database")
    CreateStore.add_cli_options(create_parser)

    args = parser.parse_args()

    if args.command == "add":
        action_cls = AddToStore
    elif args.command == "remove":
        action_cls = RemoveFromStore
    elif args.command == "query":
        action_cls = SearchStore
    elif args.command == "create":
        action_cls = CreateStore
    else:
        raise RuntimeError("Unknown action to be performed")

    action = action_cls.from_cli(args)

    with AMSDataStore(action.ams_config.db_path, action.ams_config.db_store, action.ams_config.name, False) as store:
        action(store)


if __name__ == "__main__":
    main()
