from pathlib import Path
import os
import argparse
import json

from flux.job import JobspecV1
import flux.job as fjob
import flux

from ams.store import AMSDataStore


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
    print("Submitting ", command)

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
    print(type(fut))
    tmp = FluxJobStatus(flux_handle)
    result_fut = flux.job.result_async(flux_handle, jobid)
    result_fut.then(result_cb, flux_handle, store, jobs)


def result_cb(fut, flux_handle, store, jobs):
    print(type(fut))
    job = fut.get_info()
    result = job.result.lower()
    tmp = FluxJobStatus(flux_handle)
    current_job = jobs.pop(0)
    print(f"{job.id}: {current_job.name } finished with {result} and returned {job.returncode}")
    print(current_job)
    push_next_job(flux_handle, submit_futurtore, jobs)


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
    # NOTE: I am contemplating whether we should have a single "sub-select" job or having
    # multiple ones. The multiple ones sounds more generic, but more complex.
    sub_select_job_descr = data["sub-select-jobs"][0]
    train_job_descr = data["train-jobs"][0]

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
            sub_select_job = AMSSubSelectJob.from_descr(ams_store, sub_select_job_descr)
            sub_select_job.environ = os.environ

            train_job = AMSMLTrainJob.from_descr(ams_store, train_job_descr)
            train_job.environ = os.environ

            scheduled_jobs.append(d_job)
            scheduled_jobs.append(stage_job)
            scheduled_jobs.append(sub_select_job)
            scheduled_jobs.append(train_job)
        push_next_job(flux_handle, ams_store, scheduled_jobs)
        flux_handle.reactor_run()

    return


main()
