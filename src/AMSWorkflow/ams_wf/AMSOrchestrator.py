# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# AMSLib Project Developers
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse

from ams import orchestrator


def main():
    parser = argparse.ArgumentParser(description="AMS Machine Learning Daemon running on Training allocation")
    parser.add_argument(
        "--ml-uri", help="Flux uri of an already existing allocation to schedule ML training jobs", required=True
    )

    parser.add_argument(
        "--ams-rmq-config", "-a", help="AMS configuration file containing the rmq server configuration", required=True
    )

    parser.add_argument(
        "--job-file",
        "-j",
        help="A JSON file containing a list of AMSJob descriptions (to be used for debugging)",
        required=False,
        default=None,
    )

    parser.add_argument(
        "--fake-flux",
        help="Fake the flux job submission by executing subprocesses to be used for debugging",
        action="store_true",
    )

    parser.add_argument(
        "--fake-rmq-update",
        help="Fake the flux job submission by executing subprocesses to be used for debugging",
        action="store_true",
    )

    parser.add_argument(
        "--fake-rmq-publish",
        help="Fake the RabbitMQ publish step",
        action="store_true",
    )

    args = parser.parse_args()
    print("Starting")

    orchestrator.run(
        args.ml_uri,
        args.ams_rmq_config,
        args.job_file,
        args.fake_flux,
        args.fake_rmq_update,
        args.fake_rmq_publish,
    )


if __name__ == "__main__":
    main()
