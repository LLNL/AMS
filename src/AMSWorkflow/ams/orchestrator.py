# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# AMSLib Project Developers
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import flux
import json
from flux.security import SecurityContext
from flux import job
from flux.job import JobspecV1

from .ams_rmq import RMQClient


class Orchestrator:
    def __init__(self, server_config_fn, certificate):
        with open(server_config_fn, "r") as fd:
            server_config = json.load(fd)

        self.host = (server_config["service-host"],)
        self.vhost = (server_config["rabbitmq-vhost"],)
        self.port = (server_config["service-port"],)
        self.user = (server_config["rabbitmq-user"],)
        self.password = server_config["rabbitmq-password"]
        self.certificate = certificate

class AMSDaemon(Orchestrator):
    """
    Class modeling a rmq-client daemon running on a compute
    allocation that will issue flux run commands
    """

    def __init__(self, server_config_fn, certificate):
        super().__init__(server_config_fn, certificate)

    def getMLJobSpec(self, client):
        with client.connect("test3") as channel:
            spec = channel.receive(n_msg=1).pop()
            # TODO: Write some simple wrapper class around the 'dict'
            # to correcly create ML Job Specification
        return spec

    def __run(self, flux_handle, client, jobspec):
        with client.connect("ml-start") as channel:
            while True:
                # Currently we ignore the message
                channel.receive(n_msg=1)
                jobid = flux.job.submit(flux_handle, jobspec, pre_signed=True, wait=True)
                job.wait()
                # TODO: Send completed message in RMQ

    def __call__(self):
        flux_handle = flux.Flux()

        with RMQClient(self.host, self.port, self.vhost, self.user, self.password, self.certificate) as client:
            # We currently assume a single ML job specification
            ml_job_spec = self.getMLJobSpec(client)

            # Create a Flux Job Specification
            jobspec = JobspecV1.from_command(
                command=ml_job_spec["jobspec"]["command"],
                num_tasks=ml_job_spec["jobspec"]["num_tasks"],
                num_nodes=ml_job_spec["jobspec"]["num_nodes"],
                cores_per_task=ml_job_spec["jobspec"]["cores_per_task"],
                gpus_per_task=ml_job_spec["jobspec"]["gpus_per_task"],
            )

            ctx = SecurityContext()
            signed_jobspec = ctx.sign_wrap_as(
                ml_job_spec["uid"], jobspec.dumps(), mech_type="none"
            ).decode("utf-8")
            # This is a 'busy' loop
            self.__run(flux_handle, client, signed_jobspec)

class FluxDaemonWrapper(Orchestrator):
    """
    class to start Daemon through Flux
    """

    def __init__(self, server_config_fn, certificate):
        super().__init__(server_config_fn, certificate)

    def getFluxUri(self, client):
        with client.connect("test3") as channel:
            msg = channel.receive(n_msg=1).pop()
        return msg['ml_uri']

    def __call__(self, application_cmd : list):
        if not isinstance(application_cmd, list):
            raise TypeError('StartDaemon requires application_cmd as a list')

        with RMQClient(self.host, self.port, self.vhost, self.user, self.password, self.certificate) as client:
            self.uri = getFluxUri(client)

        flux_cmd = [
            "flux",
            "proxy",
            "--force",
            f"{self.uri}",
            "flux"]
        cmd = flux_cmd + application_cmd
        subprocess.run(cmd)

