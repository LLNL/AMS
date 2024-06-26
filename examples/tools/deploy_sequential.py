from pathlib import Path
import os
import argparse
import json

from flux.job import JobspecV1
import flux.job as fjob
import flux

from ams.store import AMSDataStore

from typing import Optional
from dataclasses import dataclass
from ams import util


def constuct_cli_cmd(executable, *args, **kwargs):
    command = [executable]
    for k, v in kwargs.items():
        if len(k) == 1:
            command.append(f"-{k}")
        else:
            command.append(f"--{k}")
        command.append(str(v))

    for a in args:
        command.append(str(a))

    return command


@dataclass(kw_only=True)
class AMSJobResources:
    nodes: int
    tasks_per_node: int
    cores_per_task: int = 1
    exclusive: Optional[bool] = True
    gpus_per_task: Optional[int] = 0


class AMSJob:
    """
    Class Modeling a Job scheduled by AMS.
    """

    def __init__(
        self,
        name,
        executable,
        environ={},
        resources=None,
        stdout=None,
        stderr=None,
        ams_log=False,
        cli_args=[],
        cli_kwargs={},
    ):
        self._name = name
        self._executable = executable
        self._resources = resources
        self.environ = environ
        self._stdout = stdout
        self._stderr = stderr
        self._cli_args = list(cli_args)
        self._cli_kwargs = dict(cli_kwargs)

    def generate_cli_command(self):
        return constuct_cli_cmd(self.executable, *self._cli_args, **self._cli_kwargs)

    def __str__(self):
        data = {}
        data["name"] = self._name
        data["executable"] = self._executable
        data["stdout"] = self._stdout
        data["stderr"] = self._stderr
        data["cli_args"] = self._cli_args
        data["cli_kwargs"] = self._cli_kwargs
        data["resources"] = self._resources
        return f"{self.__class__.__name__}({data})"

    def precede_deploy(self, store):
        pass

    @property
    def resources(self):
        """The resources property."""
        return self._resources

    @resources.setter
    def resources(self, value):
        self._resources = value

    @property
    def executable(self):
        """The executable property."""
        return self._executable

    @executable.setter
    def executable(self, value):
        self._executable = value

    @property
    def environ(self):
        """The environ property."""
        return self._environ

    @environ.setter
    def environ(self, value):
        if isinstance(value, type(os.environ)):
            self._environ = dict(value)
            return
        elif not isinstance(value, dict) and value is not None:
            raise RuntimeError(f"Unknwon type {type(value)} to set job environment")

        self._environ = value

    @property
    def stdout(self):
        """The stdout property."""
        return self._stdout

    @stdout.setter
    def stdout(self, value):
        self._stdout = value

    @property
    def stderr(self):
        """The stderr property."""
        return self._stderr

    @stderr.setter
    def stderr(self, value):
        self._stderr = value

    @property
    def name(self):
        """The name property."""
        return self._name

    @name.setter
    def name(self, value):
        self._name = value


class AMSDomainJob(AMSJob):
    def _generate_ams_object(self, store):
        ams_object = dict()
        if self.stage_dir is None:
            ams_object["db"] = {"fs_path": str(store.get_candidate_path()), "dbType": "hdf5"}
        else:
            ams_object["db"] = {"fs_path": self.stage_dir, "dbType": "hdf5"}

        ams_object["ml_models"] = dict()
        ams_object["domain_models"] = dict()

        for i, name in enumerate(self.domain_names):
            model = store.search(entry="models", version="latest", metadata={"domain": name})
            # This is the case in which we do not have any model
            # Thus we create a data gathering entry
            if len(model) == 0:
                model_entry = {
                    "uq_type": "random",
                    "model_path": "",
                    "uq_aggregate": "mean",
                    "threshold": 1,
                    "db_label": str(util.get_unique_fn()),
                }
            else:
                model_entry = {}
                raise NotImplementedError("Still missing support for passing explicit models")

            ams_object["ml_models"][f"model_{i}"] = model_entry
            ams_object["domain_models"][name] = f"model_{i}"
        return ams_object

    def __init__(self, domain_names, stage_dir, *args, **kwargs):
        self.domain_names = domain_names
        self.stage_dir = stage_dir
        self._ams_object = None
        self._ams_object_fn = None
        super().__init__(*args, **kwargs)

    @classmethod
    def from_descr(cls, stage_dir, kwargs):
        domain_job_resources = AMSJobResources(**kwargs["resources"])
        return cls(
            name=kwargs["name"],
            stage_dir=stage_dir,
            domain_names=kwargs["domain_names"],
            environ=None,
            resources=domain_job_resources,
            **kwargs["cli"],
        )

    def precede_deploy(self, store):
        self._ams_object = self._generate_ams_object(store)
        tmp_path = util.mkdir(store.root_path, "tmp")
        self._ams_object_fn = f"{tmp_path}/{util.get_unique_fn()}.json"
        with open(self._ams_object_fn, "w") as fd:
            json.dump(self._ams_object, fd)
        self.environ["AMS_OBJECTS"] = str(self._ams_object_fn)


class AMSFSStageJob(AMSJob):
    def __init__(
        self,
        store_dir,
        src_dir,
        dest_dir,
        resources,
        environ=None,
        stdout=None,
        stderr=None,
        prune_module_path=None,
        prune_class=None,
        cli_args=[],
        cli_kwargs={},
    ):
        _cli_args = list(cli_args)
        _cli_args.append("--store")
        _cli_kwargs = dict(cli_kwargs)
        _cli_kwargs["dest"] = dest_dir
        _cli_kwargs["src"] = src_dir
        _cli_kwargs["pattern"] = "*.h5"
        _cli_kwargs["db-type"] = "dhdf5"
        _cli_kwargs["mechanism"] = "fs"
        _cli_kwargs["policy"] = "process"
        _cli_kwargs["persistent-db-path"] = store_dir
        _cli_kwargs["src"] = src_dir

        if prune_module_path is not None:
            assert Path(prune_module_path).exists(), "Module path to user pruner does not exist"
            _cli_kwargs["load"] = prune_module_path
            _cli_kwargs["class"] = prune_class

        super().__init__(
            name="AMSStage",
            executable="AMSDBStage",
            environ=environ,
            resources=resources,
            stdout=stdout,
            stderr=stderr,
            cli_args=_cli_args,
            cli_kwargs=_cli_kwargs,
        )

    @staticmethod
    def resources_from_domain_job(domain_job):
        return AMSJobResources(
            nodes=domain_job.resources.nodes,
            tasks_per_node=1,
            cores_per_task=5,
            exclusive=False,
            gpus_per_task=domain_job.resources.gpus_per_task,
        )


def main():
    parser = argparse.ArgumentParser(description="AMS JOB descriptor")

    parser.add_argument("--job-file", help="Jobs to be scheduled by AMS", required=True)

    args = parser.parse_args()

    fn = Path(args.job_file).absolute()
    assert fn.exists(), f"Path {str(fn)} does not exist"

    with open(fn, "r") as fd:
        data = json.load(fd)

    domain_jobs = list()
    for v in data["domain-jobs"]:
        print(v)
        domain_jobs.append(BaseJob(**v))
        print(domain_jobs[-1].generate_cli_command())


class FluxJobStatus:
    """Simple class to get job info from active Flux handle"""

    def __init__(self, handle):
        self.handle = handle

    def get_job(self, jobid):
        """
        Get details for a job
        """
        jobid = fjob.JobID(jobid)
        payload = {"id": jobid, "attrs": ["all"]}
        rpc = fjob.list.JobListIdRPC(self.handle, "job-list.list-id", payload)
        try:
            jobinfo = rpc.get()

        # The job does not exist!
        except FileNotFoundError:
            return None

        jobinfo = jobinfo["job"]

        # User friendly string from integer
        state = jobinfo["state"]
        jobinfo["state"] = fjob.info.statetostr(state)

        # Get job info to add to result
        info = rpc.get_jobinfo()
        jobinfo["nnodes"] = info._nnodes
        jobinfo["result"] = info.result
        jobinfo["returncode"] = info.returncode
        jobinfo["runtime"] = info.runtime
        jobinfo["priority"] = info._priority
        jobinfo["waitstatus"] = info._waitstatus
        jobinfo["nodelist"] = info._nodelist
        jobinfo["nodelist"] = info._nodelist
        jobinfo["exception"] = info._exception.__dict__

        # Only appears after finished?
        if "duration" not in jobinfo:
            jobinfo["duration"] = ""
        return jobinfo


def submit_ams_job(
    flux_handle,
    command,
    resources,
    environ,
    stdout=None,
    stderr=None,
):
    jobspec = JobspecV1.from_command(
        command=command,
        num_tasks=resources.tasks_per_node * resources.nodes,
        num_nodes=resources.nodes,
        cores_per_task=resources.cores_per_task,
        gpus_per_task=resources.gpus_per_task,
        exclusive=resources.exclusive,
    )

    jobspec.setattr_shell_option("mpi", "spectrum")
    jobspec.setattr_shell_option("gpu-affinity", "per-task")
    jobspec.stdout = "ams_test.out"
    jobspec.stderr = "ams_test.err"

    if environ is not None:
        jobspec.environment = environ

    if stderr is not None:
        jobspec.stderr = stderr
    if stdout is not None:
        jobspec.stdout = stdout

    return jobspec, fjob.submit_async(flux_handle, jobspec)


def submit_cb(fut, flux_handle, store, jobs):
    jobid = fut.get_id()
    tmp = FluxJobStatus(flux_handle)
    result_fut = flux.job.result_async(flux_handle, jobid)
    result_fut.then(result_cb, flux_handle, store, jobs)


def result_cb(fut, flux_handle, store, jobs):
    job = fut.get_info()
    result = job.result.lower()
    tmp = FluxJobStatus(flux_handle)
    current_job = jobs.pop(0)
    print(f"{job.id}: {current_job.name } finished with {result} and returned {job.returncode}")
    print(current_job)
    push_next_job(flux_handle, store, jobs)


def push_next_job(flux_handle, store, jobs):
    if len(jobs) == 0:
        return
    job = jobs[0]
    job.precede_deploy(store)
    spec, submit_future = submit_ams_job(
        flux_handle,
        job.generate_cli_command(),
        job.resources,
        job.environ,
        stdout=job.stdout,
        stderr=job.stdout,
    )
    submit_future.then(submit_cb, flux_handle, store, jobs)


def main():
    parser = argparse.ArgumentParser(description="AMS workflow deployment")
    parser.add_argument("--flux-uri", help="Flux uri of an already existing allocation")
    parser.add_argument("--nnodes", help="Number of nnodes to use for this AMS Deployment", required=True)
    parser.add_argument("--job-file", help="Jobs to be scheduled by AMS", required=True)
    parser.add_argument("--stage-dir", help="Stage files to temporary directory", required=False, default=None)

    args = parser.parse_args()

    if args.stage_dir is not None:
        assert Path(args.stage_dir).exists(), "Path should exist"

    fn = Path(args.job_file).absolute()
    assert fn.exists(), f"Path {str(fn)} does not exist"

    with open(fn, "r") as fd:
        data = json.load(fd)

    # Get all the jobs to be scheduled
    domain_jobs = data["domain-jobs"]
    stage_job_descr = data["stage-job"]

    # We create/open the store
    with AMSDataStore(data["db"]["kosh_path"], data["db"]["store-name"], data["db"]["name"]) as ams_store:
        # Create a termporal directory to store "ams_objects"
        # with FluxBootStrap(int(args.nnodes), scheduler, args.flux_uri) as flux_bootstrap:
        flux_handle = flux.Flux(args.flux_uri)
        # We pick the first job in our descriptor and we prepare to schedule it.
        # The sequential scheduling is "easy". We push the first 'domain-job' to the queue
        # and then every 'complete call back pushes the next 'Task'. Tasks are scheduled
        # in round-robin-fashion which means. Domain-job, staging, training and next
        scheduled_jobs = []
        for job_descr in domain_jobs:
            d_job = AMSDomainJob.from_descr(args.stage_dir, job_descr)
            d_job.environ = os.environ
            stage_resources = AMSFSStageJob.resources_from_domain_job(d_job)
            stage_job = AMSFSStageJob(
                data["db"]["kosh_path"],
                args.stage_dir,
                str(ams_store.get_candidate_path()),
                stage_resources,
                environ=os.environ,
                **stage_job_descr,
            )
            scheduled_jobs.append(d_job)
            scheduled_jobs.append(stage_job)
        push_next_job(flux_handle, ams_store, scheduled_jobs)
        flux_handle.reactor_run()

    return


main()
