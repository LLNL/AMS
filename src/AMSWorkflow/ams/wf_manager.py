from ams.ams_jobs import AMSDomainJob, AMSNetworkStageJob, AMSMLTrainJob, AMSOrchestratorJob, AMSSubSelectJob
import os
import time
from ams.ams_flux import AMSFluxExecutor
from ams.ams_jobs import AMSJob, AMSJobResources
from ams.rmq import AMSFanOutProducer, AMSRMQConfiguration, AMSSyncProducer
from ams.store import AMSDataStore
import flux
import json
from flux.job.list import get_job

from typing import Tuple, Dict, List, Optional, Set
from dataclasses import dataclass
from pathlib import Path


def get_allocation_resources(uri: str) -> Tuple[int, int, int]:
    """
    @brief Returns the resources of a flux allocation

    :param uri: A flux uri to querry the resources from
    :return: A tuple of (nnodes, cores_per_node, gpus_per_node)
    """
    flux_instance = flux.Flux(uri)
    resources = flux.resource.resource_list(flux_instance).get()["all"]
    cores_per_node = int(resources.ncores / resources.nnodes)
    gpus_per_node = int(resources.ngpus / resources.nnodes)
    return resources.nnodes, cores_per_node, gpus_per_node


@dataclass
class Partition:
    uri: str
    nnodes: int
    cores_per_node: int
    gpus_per_node: int

    @classmethod
    def from_uri(cls, uri):
        res = get_allocation_resources(uri)
        return cls(uri=uri, nnodes=res[0], cores_per_node=res[1], gpus_per_node=res[2])


class JobList(list):
    """
    @brief A list of 'AMSJobs'
    """

    def append(self, job: AMSJob):
        if not isinstance(job, AMSJob):
            raise TypeError("{self.__classs__.__name__} expects an item of a job")

        super().append(job)

    def __getitem__(self, index):
        return super().__getitem__(index)

    def __setitem__(self, index, value):
        if not isinstance(value, AMSJob):
            raise TypeError("{self.__classs__.__name__} expects an item of a job")

        super().__setitem__(index, value)


class AMSWorkflowManager:
    """
    @brief Manages all job submissions of the current execution.
    """

    def __init__(
        self,
        rmq_config: str,
        kosh_path: str,
        store_name: str,
        db_name: str,
        domain_jobs: JobList,
        stage_jobs: JobList,
        sub_select_jobs: JobList,
        train_jobs: JobList,
    ):
        self._rmq_config = rmq_config
        self._kosh_path = kosh_path
        self._store_name = store_name
        self._db_name = db_name
        self._domain_jobs = domain_jobs
        self._stage_jobs = stage_jobs
        self._sub_select_jobs = sub_select_jobs
        self._train_jobs = train_jobs

    @property
    def rmq_config(self):
        return self._rmq_config

    def __str__(self):
        out = ""
        for job in self._domain_jobs:
            out += str(job) + "\n"

        for job in self._stage_jobs:
            out += str(job) + "\n"

        for job in self._sub_select_jobs:
            out += str(job) + "\n"

        for job in self._train_jobs:
            out += str(job) + "\n"

        return out

    def broadcast_train_specs(self, rmq_config):
        with AMSSyncProducer(
            rmq_config.service_host,
            rmq_config.service_port,
            rmq_config.rabbitmq_vhost,
            rmq_config.rabbitmq_user,
            rmq_config.rabbitmq_password,
            rmq_config.rabbitmq_cert,
            rmq_config.rabbitmq_ml_submit_queue,
        ) as rmq_fd:

            for ml_job in self._train_jobs:
                request = json.dumps(
                    [
                        {
                            "domain_name": ml_job.domain,
                            "job_type": "train",
                            "spec": ml_job.to_dict(),
                            "ams_log": True,
                            "request_type": "register_job_spec",
                        }
                    ]
                )
                rmq_fd.send_message(request)

            for subselect_job in self._sub_select_jobs:
                request = json.dumps(
                    [
                        {
                            "domain_name": subselect_job.domain,
                            "job_type": "sub_select",
                            "spec": subselect_job.to_dict(),
                            "ams_log": True,
                            "request_type": "register_job_spec",
                        }
                    ]
                )
                rmq_fd.send_message(request)

    def done_cb(self, future):
        job_id = future.jobid()
        print(f"{job_id} is done")

    def start_domain(self, store, rmq_config, domain_uri):
        print("Start Domain")
        with AMSFluxExecutor(False, threads=1, handle_args=(domain_uri,)) as domain_executor:
            for domain_job in self._domain_jobs:
                domain_job.precede_deploy(store, rmq_config)
                print("Domain command is:", " ".join(domain_job.generate_cli_command()))
                domain_future = domain_executor.submit(domain_job.to_flux_jobspec())
                job_id = domain_future.jobid()
                print(f"Executing Domain with job id {job_id}")
                print(f"Domain with job id {job_id} result: {domain_future.result()}")

    def start_stagers(self, store, rmq_config, domain_uri, stage_uri):
        with AMSFluxExecutor(False, threads=1, handle_args=(stage_uri,)) as stager_executor:
            print("Connected to stager executor", stage_uri)
            for stager in self._stage_jobs:
                print("Stager command is:", " ".join(stager.generate_cli_command()))
                stager_future = stager_executor.submit(stager.to_flux_jobspec())
                stager_future.add_done_callback(self.done_cb)
                job_id = stager_future.jobid()
                print(f"Stager JOB-ID is  {job_id}")
                self.start_domain(store, rmq_config, domain_uri)

    def start(self, domain_uri, stage_uri, ml_uri):
        ams_orchestartor_job = AMSOrchestratorJob(ml_uri, self.rmq_config)
        rmq_config = AMSRMQConfiguration.from_json(self.rmq_config)
        print(f"Starting ..... {ml_uri} ... {stage_uri} ... {domain_uri}")
        with AMSDataStore(self._kosh_path, self._store_name, self._db_name) as store:
            print("Opened the AMS Store")
            with AMSFluxExecutor(False, threads=1, handle_args=(ml_uri,)) as ml_executor:
                print("Connected to ml executor")
                # The AMSFanOutProducer enables us to send control message to all stagers and
                # ml trainers. Currently
                with AMSFanOutProducer(
                    rmq_config.service_host,
                    rmq_config.service_port,
                    rmq_config.rabbitmq_vhost,
                    rmq_config.rabbitmq_user,
                    rmq_config.rabbitmq_password,
                    rmq_config.rabbitmq_cert,
                ) as orchestrator_publisher:
                    ml_future = ml_executor.submit(ams_orchestartor_job.to_flux_jobspec())
                    job_id = ml_future.jobid()
                    print("ML JOB ID is:", job_id)
                    self.start_stagers(store, rmq_config, domain_uri, stage_uri)
                    self.broadcast_train_specs(rmq_config)
                    orchestrator_publisher.broadcast(json.dumps({"request_type": "terminate"}))
                ml_executor.shutdown(wait=True)

    @classmethod
    def from_descr(
        cls,
        json_file: str,
        rmq_config: Optional[str] = None,
    ):

        def collect_domains(jobs: JobList) -> Set[str]:
            return {job.domain for job in jobs}

        def create_domain_list(domains: List[Dict]) -> JobList:
            jobs = JobList()
            for job_descr in domains:
                jobs.append(AMSDomainJob.from_descr(job_descr))
            return jobs

        if not Path(json_file).exists():
            raise RuntimeError(f"Workflow description file {json_file} does not exist")

        with open(json_file, "r") as fd:
            data = json.load(fd)

        if "db" not in data:
            raise KeyError("Workflow decsription file misses 'db' description")

        if not all(key in data["db"] for key in {"kosh-path", "name", "store-name"}):
            raise KeyError("Workflow description files misses entries in 'db'")

        store = AMSDataStore(data["db"]["kosh-path"], data["db"]["store-name"], data["db"]["name"])

        if "domain-jobs" not in data:
            raise KeyError("Workflow description files misses 'domain-jobs' entry")

        if len(data["domain-jobs"]) == 0:
            raise RuntimeError("There are no jobs described in workflow description file")

        domain_jobs = create_domain_list(data["domain-jobs"])

        if "stage-job" not in data:
            raise RuntimeError("There is no description for a stage-job")

        stage_type = data["stage-job"].pop("type", "rmq")
        num_instances = data["stage-job"].pop("instances", 1)

        assert num_instances == 1, "We only support 1 instance at the moment"
        assert stage_type == "rmq", "We only support 'rmq' stagers"

        stage_resources = AMSJobResources(nodes=1, tasks_per_node=1, cores_per_task=6, gpus_per_task=0)
        stage_jobs = JobList()
        stage_job = AMSNetworkStageJob.from_descr(
            data["stage-job"], store.get_candidate_path(), store.root_path, rmq_config, stage_resources
        )
        # NOTE: We need to always copy in our environment. To make sure we find the respective packages
        stage_job.environ = os.environ
        stage_job.stdout = "stager_test.out"
        stage_job.stderr = "stager_test.err"
        stage_jobs.append(stage_job)

        sub_select_jobs = JobList()
        assert "sub-select-jobs" in data, "We are expecting a subselection job"
        for sjob in data["sub-select-jobs"]:
            sub_select_jobs.append(AMSSubSelectJob.from_descr(store, sjob))

        sub_select_domains = collect_domains(sub_select_jobs)

        assert "train-jobs" in data, "We are expecting training jobs"

        train_jobs = JobList()
        for sjob in data["train-jobs"]:
            train_jobs.append(AMSMLTrainJob.from_descr(store, sjob))

        train_domains = collect_domains(train_jobs)
        wf_domain_names = []
        for job in domain_jobs:
            wf_domain_names.append(*job.domain_names)

        wf_domain_names = list(set(wf_domain_names))

        for domain in wf_domain_names:
            assert domain in train_domains, f"Domain {domain} misses a train description"
            assert domain in sub_select_domains, f"Domain {domain} misses a subselection description"

        store = AMSDataStore(data["db"]["kosh-path"], data["db"]["store-name"], data["db"]["name"])
        return cls(
            rmq_config,
            data["db"]["kosh-path"],
            data["db"]["store-name"],
            data["db"]["name"],
            domain_jobs,
            stage_jobs,
            sub_select_jobs,
            train_jobs,
        )
