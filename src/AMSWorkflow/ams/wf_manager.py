from ams.ams_jobs import AMSDomainJob, AMSNetworkStageJob, AMSMLTrainJob, AMSSubSelectJob
from ams.ams_jobs import AMSJob, AMSJobResources
from ams.store import AMSDataStore
import flux
import json

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
        kosh_path: str,
        store_name: str,
        db_name: str,
        domain_jobs: JobList,
        stage_jobs: JobList,
        sub_select_jobs: JobList,
        train_jobs: JobList,
    ):
        self._kosh_path = kosh_path
        self._store_name = store_name
        self._db_name = db_name
        self._domain_jobs = domain_jobs
        self._stage_jobs = stage_jobs
        self._sub_select_jobs = sub_select_jobs
        self._train_jobs = train_jobs

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

    @classmethod
    def from_descr(
        cls,
        json_file: str,
        creds: Optional[str] = None,
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
        stage_jobs.append(
            AMSNetworkStageJob.from_descr(
                data["stage-job"], store.get_candidate_path(), store.root_path, creds, stage_resources
            )
        )

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
            data["db"]["kosh-path"],
            data["db"]["store-name"],
            data["db"]["name"],
            domain_jobs,
            stage_jobs,
            sub_select_jobs,
            train_jobs,
        )
