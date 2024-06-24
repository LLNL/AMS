import argparse
import logging
import sys
import os
import json
from pathlib import Path
from urllib import parse

from ams.deploy_tools import spawn_rmq_broker
from ams.deploy_tools import RootSched
from ams.deploy_tools import start_flux
from ams.rmq_async import broker_status

logger = logging.getLogger(__name__)


def get_rmq_credentials(flux_uri, rmq_creds, rmq_cert):
    if rmq_creds is None:
        # TODO Overhere we need to spawn our own server
        rmq_creds, rmq_cert = spawn_rmq_broker(flux_uri)
    with open(rmq_creds, "r") as fd:
        rmq_creds = json.load(fd)

    return rmq_creds, rmq_cert


def main():
    parser = argparse.ArgumentParser(description="AMS workflow deployment")

    parser.add_argument("--rmq-creds", help="Credentials file (JSON)")
    parser.add_argument("--rmq-cert", help="TLS certificate file")
    parser.add_argument("--flux-uri", help="Flux uri of an already existing allocation")
    parser.add_argument("--nnodes", help="Number of nnodes to use for this AMS Deployment")
    parser.add_argument(
        "--root-scheduler",
        dest="scheduler",
        choices=[e.name for e in RootSched],
        help="The provided scheduler of the cluster",
    )

    args = parser.parse_args()

    """
    Verify System is on a "Valid" Status
    """

    if args.flux_uri is None and args.scheduler is None:
        print("Please provide either a flux URI handle to connect to or provide the base job scheduler")
        sys.exit()

    flux_process = None
    flux_uri = args.flux_uri
    flux_daemon_script = ""
    if flux_uri is None:
        flux_process, flux_uri, flux_daemon_script = start_flux(RootSched[args.scheduler], args.nnodes)

    rmq_creds, rmq_cert = get_rmq_credentials(flux_uri, args.rmq_creds, args.rmq_cert)

    if not broker_status(rmq_creds, rmq_cert):
        # If we created a subprocess in the background to run flux, we should terminate it
        if flux_process is not None:
            flux_process.terminate()
        print("RMQ Broker is not connected, exiting ...")
        sys.exit()

    """
    We Have FLUX URI and here we know that rmq_creds, and rmq_cert are valid and we can start
    scheduling jobs
    """

    # Step 1: Get Resources Available to FLUX

    """
    Terminate Processes We reached the end
    """

    if flux_process is not None:
        print("Terminating")
        flux_process.terminate()
        print("Return code ", flux_process.wait())
        Path(flux_daemon_script).unlink()


if __name__ == "__main__":
    main()
