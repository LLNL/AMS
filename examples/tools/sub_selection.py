#!/usr/bin/env python

# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# AMSLib Project Developers
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
import random

from ams.config import AMSInstance
from ams.faccessors import get_writer
from ams.store import AMSDataStore
from ams.views import AMSDataView


class BinomialOptionsView(AMSDataView):
    """
    A class providing semantic information
    to the data stored in the kosh-data store
    """

    input_feature_names = ["S", "X", "R", "V", "T"]
    input_feature_dims = [1] * 5
    input_feature_types = ["scalar"] * 5

    output_feature_names = ["CallValue"]
    output_feature_dims = [1]
    output_feature_types = ["scalar"]

    def __init__(self, ams_store, domain_name, entry, versions=None, **options):
        super().__init__(ams_store, domain_name, entry, versions=versions, **options)


def sub_select(num_elements, data, candidates, path_to_model):
    """
    Args:
        num_elements: The number of elements to select and return back to the main application.
        data: A BinomialOptionsView of the data that the models was trained with
        candidates: A BinomialOptionsView of the new data that the model has never seen
        path_to_model: Path to the latest model

    Returns:
        A tuple of input, output pairs that will be stored in the database
    """

    X, Y = candidates.get_data()

    num_items = X.shape[0]
    if num_elements > num_items:
        return X, Y

    rand_indexes = sorted(random.sample(range(num_items), num_elements))
    return X[rand_indexes, ...], Y[rand_indexes, ...]


def main():
    parser = argparse.ArgumentParser(description="Training of AMS model")
    parser.add_argument("--elements", "-e", type=int, default=1000, help="Number of training iterations")
    parser.add_argument("--persistent-db-path", "-db", help="The path of the AMS database", required=True)
    parser.add_argument(
        "--benchmark",
        "-b",
        help="The benchmark to perform sub-selection on",
        choices={"binomial", "idealgas"},
        required=True,
    )

    parser.add_argument("--domain-name", "-dn", help="The domain name to perform sub-selection on", required=True)

    args = parser.parse_args()

    # Get an AMS configuration this class is initialized by the environment
    ams_config = AMSInstance.from_path(args.persistent_db_path)

    # Open the AMS Storage
    with AMSDataStore(ams_config.db_path, ams_config.db_store, ams_config.name, False) as db:

        found = db.search(domain_name=args.domain_name, entry=None)
        candidate_files = db.get_candidate_files(args.domain_name)
        model = db.get_model_files(args.domain_name, versions="latest")
        data_files = db.get_data_files(args.domain_name)

        # When we don't have any candidate data, or a model we push everything to the store
        # and we are done
        if len(data_files) == 0 or len(model) == 0:
            # This is the first time I am executing this thus we copy everything to the data entry
            db.move(args.domain_name, "candidates", "data", candidate_files)
            return

        if len(candidate_files) == 0:
            # No candidates in data base
            return

        # Open Views of data, candidates to start sub-selection process
        with BinomialOptionsView(db, args.domain_name, "data") as data:
            with BinomialOptionsView(db, args.domain_name, "candidates") as candidates:
                print("Here")
                sb_X, sb_Y = sub_select(args.elements, data, candidates, model[0])
                # Pick the correct ams writer to store data
                # Pick a data base suggested file name. This is not mandatory but is good practice.
                ams_writer = get_writer(ams_config.db_type)
                fn = db.suggest_data_file_name(args.domain_name)
                with ams_writer(fn) as fd:
                    fd.store(sb_X, sb_Y)

                # Make the data public to kosh so we can pick them every time we open a view
                db.add_data(args.domain_name, data_files=[fn])

        # remove candidates from store and delete the data-files
        db.remove_candidates(args.domain_name, candidate_files, delete_files=True)


if __name__ == "__main__":
    main()
