from ams.ams_jobs import AMSDomainJob, AMSFSStageJob, AMSNetworkStageJob
from ams.store import AMSDataStore
import flux
import json

from typing import Tuple, Dict, List, Optional
from ams_jobs import AMSJob
from dataclasses import dataclass, fields
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


class WorkflowManager:
    """
    @brief Manages all job submissions of the current execution.
    """

    def __init__(self, kosh_path: str, store_name: str, db_name: str, jobs: Dict[str, JobList]):
        self._kosh_path = kosh_path
        self._store_name = store_name
        self._db_name = db_name
        self._jobs = jobs

    @classmethod
    def from_json(
        cls,
        domain_resources: Partition,
        stage_resources: Partition,
        train_resources: Partition,
        json_file: str,
        creds: Optional[str] = None,
    ):

        def create_domain_list(domains: List[Dict]) -> List[JobList]:
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

        stage_job = AMSNetworkStageJob.from_descr(
            data["stage-job"],
            store.get_candidate_path(),
            store.root_path,
            creds,
            stage_resources.nnodes,
            stage_resources.cores_per_node,
            stage_resources.gpus_per_node,
        )
