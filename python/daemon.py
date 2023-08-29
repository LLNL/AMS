#!/usr/bin/env python3
import sys
import os
import flux
import json
from flux.security import SecurityContext
from flux import job
from flux.job import JobspecV1
from rmq import RMQClient


class AMSDaemon:
    """
    Class to manage AMS training
    """

    def __init__(self):
        self.config_rmq_client = RMQClient("rmq/rmq-pds.json", "rmq-pds.crt")
        self.config_rmq_client.connect("test3")
        tmp = self.config_rmq_client.receive("test3", n_msg=1).pop()
        self.config = json.loads(tmp.decode("utf-8"))

        self.mljob_rmq_client = RMQClient("rmq/rmq-pds.json", "rmq-pds.crt")
        self.mljob_rmq_client.connect("ml-start")
        self.flux_handle = flux.Flux()


def main():
    daemon = AMSDaemon()
    jobspec = JobspecV1.from_command(
        command=daemon.config["jobspec"]["command"],
        num_tasks=daemon.config["jobspec"]["num_tasks"],
        num_nodes=daemon.config["jobspec"]["num_nodes"],
        cores_per_task=daemon.config["jobspec"]["cores_per_task"],
        gpus_per_task=daemon.config["jobspec"]["gpus_per_task"],
    )
    ctx = SecurityContext()
    signed_jobspec = ctx.sign_wrap_as(
        daemon.config["uid"], jobspec.dumps(), mech_type="none"
    ).decode("utf-8")
    while True:
        # Get RMQ messages and then submit jobs under
        # defined conditions
        daemon.mljob_rmq_client.receive("ml-start", n_msg=1)
        jobid = flux.job.submit(
            daemon.flux_handle, signed_jobspec, pre_signed=True, wait=True
        )
        job.wait()
        # Send completed message in RMQ


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted")
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
