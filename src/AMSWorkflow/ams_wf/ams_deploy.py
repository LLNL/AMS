import os
from pathlib import Path
import sys
import argparse
from enum import Enum
import logging
import shutil
import subprocess as sp
import json
import flux
import time
from flux.job import JobspecV1
from flux.kvs import KVSDir
import flux.job as fjob
import flux.uri as uri
import signal
from ams.store import CreateStore, AMSDataStore

logger = logging.getLogger(__name__)


class RootSched(Enum):
    SLURM = 1
    LSF = 2


class JobSpec:
    def __init__(self, name, job_descr, exclusive=True):
        self._name = name
        self._executable = job_descr["executable"]
        self._arguments = " ".join([v for v in job_descr["arguments"]])
        self._command = [job_descr["executable"]] + job_descr["arguments"]
        self._environ = os.environ.copy()
        if "env_variables" in job_descr:
            for k, v in job_descr["env_variables"].items():
                self._environ[k] = str(v)
                print(f"{k} {self._environ[k]}")

        logger.info(f"Creating Job Description: {self._executable} {self._arguments}")

        self._resources = job_descr["resources"]

        jobspec = JobspecV1.from_command(
            command=self._command,
            num_tasks=job_descr["resources"]["num_tasks"],
            num_nodes=job_descr["resources"]["num_nodes"],
            cores_per_task=job_descr["resources"]["cores_per_task"],
            gpus_per_task=job_descr["resources"]["gpus_per_task"],
            exclusive=exclusive,
        )

        stdout = job_descr.get("stdout", None)
        stderr = job_descr.get("stderr", None)
        if stdout is None:
            stdout = Path(f"{self._name}.out")
        if stderr is None:
            stderr = Path(f"{self._name}.err")

        jobspec.environment = self._environ
        jobspec.stdout = stdout
        jobspec.stderr = stderr
        self._spec = jobspec

    def start(self, flux_handle):
        logger.info(f"Submitting Job {self._name}")
        job_id = fjob.submit(flux_handle, self._spec, pre_signed=False, waitable=True)
        return job_id

    @property
    def stdout(self):
        return self._spec.stdout

    @property
    def stderr(self):
        return self._spec.stderr

    @property
    def name(self):
        return self._name


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


class AMSJobScheduler:
    def __init__(self, stager_job_generator, config):
        self._flux_handle = flux.Flux()
        logger.debug("Preparing user app job specification")
        self._user_app = JobSpec("user_app", config["user_app"], exclusive=True)
        self._ml_train = JobSpec("ml_training", config["ml_training"], exclusive=True)
        self._ml_pruner = JobSpec("ml_pruner", config["ml_pruner"], exclusive=True)
        self._stager = JobSpec("ams_stager", stager_job_generator(config), exclusive=True)


class AMSConcurrentJobScheduler(AMSJobScheduler):
    def __init__(self, config):
        def create_rmq_stager_job_descr(user_descr):
            config = dict()

            # TODO: This is SUPER ugly and not to mention
            # potenitally buggy. We will need to clean this up
            # once we have all pieces in places (including AMSlib json initialization)
            with open("rmq_config.json", "w") as fd:
                json.dump(user_descr["stager"]["rmq"], fd, indent=6)

            rmq_config_path = Path("rmq_config.json").resolve()

            config["executable"] = sys.executable
            config["arguments"] = [
                "-m",
                "ams_wf.AMSDBStage",
                "-db",
                user_descr["db"]["path"],
                "--policy",
                "process",
                "--dest",
                str(Path(user_descr["db"]["path"]) / Path("candidates")),
                "--db-type",
                "dhdf5",
                "--store",
                "--mechanism",
                "network",
                "--class",
                user_descr["stager"]["pruner_class"],
                "--cert",
                user_descr["stager"]["rmq"]["rabbitmq-cert"],
                "--creds",
                str(rmq_config_path),
                "--queue",
                user_descr["stager"]["rmq"]["rabbitmq-outbound-queue"],
                "--load",
                user_descr["stager"]["pruner_path"],
            ] + user_descr["stager"]["pruner_args"]

            config["resources"] = {
                "num_nodes": 1,
                "num_processes_per_node": 1,
                "num_tasks": 1,
                "cores_per_task": 32,
                "gpus_per_task": 0,
            }

            return config

        super().__init__(create_rmq_stager_job_descr, config)

    def execute(self):
        def execute_and_wait(job_descr, handle):
            jid = job_descr.start(handle)
            if not result.success:
                logger.critical(f"Unsuccessfull Job Execution: {job_descr.name}")
                logger.debug(f"Error code of failed job {result.jobid} is {result.errstr}")
                logger.debug(f"stdout is redirected to: {job_descr.stdout}")
                logger.debug(f"stderr is redirected to: {job_descr.stderr}")
                return False
            return True

        # We create a KVS Namespace
        kvs_dir = KVSDir(self._flux_handle)
        print("KVS (Deploy) RESOURCE DIR IS", kvs_dir["resource.R"])
        # We start stager first
        logger.debug("Start stager")
        stager_id = self._stager.start(self._flux_handle)
        logger.debug(f"Stager job id is {stager_id}")

        stager_connected = False
        # Here we actively wait for stagers to subscribe into
        # our KVS. By doing so, we know the
        for tries in range(0, 6):
            if kvs_dir.exists("AMSStager"):
                stager_connected = True
                break
            logger.debug(f"Stager KVS Value does not exist")
            time.sleep(25)

        if not stager_connected:
            logger.critical(f"Cannot connect to stager")
            stager_status = FluxJobStatus(self._flux_handle)
            logger.debug(f"{json.dumps(stager_status.get_job(stager_id), indent=4)}")
            logger.critical(f"Killing pending jobs")
            kill_status = fjob.kill(self._flux_handle, jobid=stager_id, signum=signal.SIGINT)
            fjob.wait(self._flux_handle, jobid=stager_id)
            return False

        logger.debug("Start user app")
        user_app_id = self._user_app.start(self._flux_handle)
        logger.debug(f"User App job id is {user_app_id}")

        # We are actively waiting for main application to terminate
        logger.debug("Wait for user application")
        result = fjob.wait(self._flux_handle, jobid=user_app_id)

        # stager handles SIGTERM, kill it
        logger.debug(f"{json.dumps(stager_status.get_job(stager_id), indent=4)}")
        kill_status = fjob.kill(self._flux_handle, jobid=stager_id, signum=signal.SIGINT)
        logger.debug("Waiting for job to be killed")
        fjob.wait(self._flux_handle, jobid=stager_id)
        logger.debug(f"{json.dumps(stager_status.get_job(stager_id), indent=4)}")

        return True


class AMSSequentialJobScheduler(AMSJobScheduler):
    def __init__(self, config):
        def create_fs_stager_job_descr(user_descr):
            config = dict()
            config["executable"] = sys.executable
            # TODO : Handle the case of sequential case that users RMQ
            config["arguments"] = [
                "-m",
                "ams_wf.AMSDBStage",
                "-db",
                user_descr["db"]["path"],
                "--policy",
                "process",
                "--dest",
                str(Path(user_descr["db"]["path"]) / Path("candidates")),
                "--db-type",
                "dhdf5",
                "--store",
                "-m",
                "fs",
                "--class",
                user_descr["stager"]["pruner_class"],
                "--load",
                user_descr["stager"]["pruner_path"],
            ] + user_descr["stager"]["pruner_args"]

            config["resources"] = {
                "num_nodes": 1,
                "num_processes_per_node": 1,
                "num_tasks": 1,
                "cores_per_task": 5,
                "gpus_per_task": 0,
            }

            return config

        super().__init__(create_fs_stager_job_descr, config)

    def execute(self):
        def execute_and_wait(job_descr, handle):
            jid = job_descr.start(handle)
            result = fjob.wait(handle, jobid=jid)
            if not result.success:
                logger.critical(f"Unsuccessfull Job Execution: {job_descr.name}")
                logger.debug(f"Error code of failed job {result.jobid} is {result.errstr}")
                logger.debug(f"stdout is redirected to: {job_descr.stdout}")
                logger.debug(f"stderr is redirected to: {job_descr.stderr}")
                return False
            return True

        for step in [self._user_app, self._stager, self._ml_pruner, self._ml_train]:
            if not execute_and_wait(step, self._flux_handle):
                return False

        return True


def deploy(config):
    # Before starting we need to make sure configbase files exist and
    # kosh store is up and running
    st = CreateStore(config["db"]["path"], config["db"]["store_name"], config["db"]["name"])
    logger.info(f"Generating AMS Store at {st.ams_config.db_path}")
    logger.info(f"Flux URI is {os.environ.get('FLUX_URI')}")
    with AMSDataStore(st.ams_config.db_path, st.ams_config.db_store, st.ams_config.name, False) as store:
        st(store)
    # TODO: In case of RMQ configbase we need to make sure
    # the server is up and running
    logger.info(f"")
    if config["execution_mode"] == "concurrent":
        executor = AMSConcurrentJobScheduler(config)
    elif config["execution_mode"] == "sequential":
        executor = AMSSequentialJobScheduler(config)
    return executor.execute()


def bootstrap(cmd, scheduler, flux_log):
    def slurm_bootstrap(cmd, flux_log_file):
        nnodes = os.environ.get("SLURM_NNODES", None)
        if nnodes == None:
            logger.critical("Environemnt variable 'SLURM_NNODES' is not set, cannot deduce flux number of nodes")
            sys.exit()

        bootstrap_cmd = f"srun -N {nnodes} -n {nnodes} --pty --mpi=none --mpibind=off flux start"

        if flux_log_file is not None:
            bootstrap_cmd = f"{bootstrap_cmd} -o,S,log-filename=${flux_log_file}"
        bootstrap_cmd = f"{bootstrap_cmd} {cmd}"
        logger.debug(f"Executing command {bootstrap_cmd}")
        logging.shutdown()
        result = sp.run(bootstrap_cmd, shell=True)
        return result.returncode
        # NOTE: From this point on we should definetely not use the logger mechanism. We manually shut it donw
        # to allo the bootstrapped script to use the same logger (this is important in the case of logger into a file)

    logger.info(f"Bootstrapping using {scheduler.name}")

    if scheduler == RootSched.SLURM:
        slurm_bootstrap(cmd, flux_log)
    else:
        logger.critical("Unknown scheduler, cannot bootstrap")
        sys.exit()
    return 0


class AMSConfig:
    @staticmethod
    def validate(config):
        def validate_keys(level, config, mandatory_fields):
            if not all(field in config.keys() for field in mandatory_fields):
                missing_fields = " ".join([v for v in mandatory_fields if v not in config.keys()])
                logger.critical(f"The following fields are missing : {missing_fields} from entry {level}")
                return False
            return True

        def validate_step_field(level, config):
            if not validate_keys(level, config, ["executable", "resources"]):
                logger.critical(f"Mising fields in {level}")
                return False

            if not validate_keys(level, config["resources"], ["num_nodes", "num_processes_per_node"]):
                logger.critical(f"Missing fields in resources of {level}")
                return False

            return True

        if not validate_keys(
            "root", config, ["user_app", "ml_training", "ml_pruning", "execution_mode", "db", "stager"]
        ):
            return False

        if not validate_step_field("user_app", config["user_app"]):
            return False

        if not validate_step_field("ml_training", config["ml_training"]):
            return False

        if not validate_step_field("ml_pruning", config["ml_pruning"]):
            return False

        if not validate_keys(
            "ml_training|resources", config["ml_training"]["resources"], ["num_nodes", "num_processes_per_node"]
        ):
            return False

        if not validate_keys(
            "ml_pruning|resources", config["ml_training"]["resources"], ["num_nodes", "num_processes_per_node"]
        ):
            return False

        if config["execution_mode"] not in ["sequential", "concurrent"]:
            logger.critical("Unknown 'execution_mode', please select from 'sequential', 'concurrent'")
            return False

        if config["execution_mode"] == "concurrent":
            if config["stager"]["mode"] == "filesystem":
                logger.critical("Database is concurrent but the stager polls data from filesystem")
                return False

        if config["stager"]["mode"] == "rmq":
            rmq_config = config["stager"]["rmq"]
            if not isinstance(rmq_config["service-port"], int):
                logger.critical(
                    "The RMQ service-port must be an integer type {0}".format(type(rmq_config["service-port"]))
                )
                return False
            if not Path(rmq_config["rabbitmq-cert"]).exists():
                logger.critical("The RMQ certificate file does not exist (or is not not accessible)")
                return False

            rmq_keys = AMSConfig.to_descr()["rmq"].keys()

            if not validate_keys("rmq", rmq_config, rmq_keys):
                return False

        return True

    @staticmethod
    def to_descr():
        return {
            "user_app": {
                "executable": "path to executable",
                "arguments": ["one", "two", "three"],
                "env_variables": {"VARNAME": "VALUE"},
                "resources": {"num_nodes": "XX", "num_processes_per_node": "YY", "num_gpus_per_node": "ZZ"},
            },
            "ml_training": {
                "executable": "path to executable",
                "arguments": ["one", "two", "three"],
                "env_variables": {"VARNAME": "VALUE"},
                "resources": {"num_nodes": "XX", "num_processes_per_node": "YY", "num_gpus_per_node": "ZZ"},
            },
            "ml_pruning": {
                "executable": "path to executable",
                "arguments": ["one", "two", "three"],
                "env_variables": {"VARNAME": "VALUE"},
                "resources": {"num_nodes": "XX", "num_processes_per_node": "YY", "num_gpus_per_node": "ZZ"},
            },
            "execution_mode": "sequential",
            "db": {"path": "path/to/db", "name": "Application name of this store", "store_name": "ams_store.sql"},
            "stager": {"mode": "filesystem", "num_clients": "number of rmq clients (mandatory only when mode is rmq)"},
            "rmq": {
                "service-port": "Port",
                "service-host": "server address",
                "rabbitmq-erlang-cookie": "magic cookie",
                "rabbitmq-name": "rmq server name",
                "rabbitmq-password": "password",
                "rabbitmq-user": "user",
                "rabbitmq-vhost": "virtual host",
                "rabbitmq-cert": "path to certificate to establish connection",
                "rabbitmq-inbound-queue": "Queue name to send data from outside in the simulation",
                "rabbitmq-outbound-queue": "Queue name to send data from the simulation to outside",
            },
        }


def generate_cli(parser):
    generate_parser = parser.add_parser("generate", help="Generate an AMS workflow configuration file")
    generate_parser.add_argument(
        "--config", "-c", dest="config", required=True, help="Path to the AMS file to be generated"
    )
    generate_parser.set_defaults(func=generate_config)


def generate_config(args):
    logger.info(f"Generating configuration file {args.config}")
    with open(args.config, "w") as fd:
        json.dump(AMSConfig.to_descr(), fd, indent=6)
    editor = os.environ.get("EDITOR", None)
    if editor is None:
        logger.critical(f"Environemnt variable EDITOR is not set, example configuration is stored in {args.config}")
        sys.exit()
    cmd = f"{editor} {args.config}"
    result = sp.run(cmd, shell=True)
    if result.returncode != 0:
        logger.warning(f"{editor} {args.config} returned non zero code")

    with open(args.config, "r") as fd:
        data = json.load(fd)

    if not AMSConfig.validate(data):
        logger.critical("Generated configuration file is not valid")


def validate_cli(parser):
    validate_parser = parser.add_parser("validate", help="Validate an AMS configuration file")
    validate_parser.add_argument("--config", "-c", dest="config", required=True, help="Path to configuration file")
    validate_parser.set_defaults(func=validate_config)


def validate_config(args):
    with open(args.config, "r") as fd:
        data = json.load(fd)

    if not AMSConfig.validate(data):
        logger.info("Generated configuration file is NOT valid")
        return False

    logger.info("Generated configuration file IS valid")
    return True


def start_cli(parser):
    start_parser = parser.add_parser(
        "start", help="Deploy an AMS run. The command assumes we already are inside a job allocation"
    )
    start_parser.add_argument(
        "--config", "-c", dest="config", required=True, help="AMS configuration file listing job requirements"
    )
    start_parser.add_argument(
        "--root-scheduler",
        "-s",
        dest="scheduler",
        required=True,
        choices=[e.name for e in RootSched],
        help="The provided scheduler of the cluster",
    )
    start_parser.add_argument("--flux-log-file", "-f", dest="flux_log", help="log file to be used by flux")
    start_parser.set_defaults(func=start_execute)


def start_execute(args):
    def is_bootstrapped():
        return os.environ.get("FLUX_URI") is not None

    def get_cmd():
        cmd = "python " + " ".join(sys.argv)
        return cmd

    with open(args.config, "r") as fd:
        data = json.load(fd)

    if not AMSConfig.validate(data):
        logger.info("Configuration file is not valid, exiting early...")
        return False

    if is_bootstrapped():
        logger.info("Execution is bootstrapped")
        return deploy(data)

    logger.info("Execution is NOT bootstrapped")
    cmd = get_cmd()
    return bootstrap(cmd, RootSched[args.scheduler], args.flux_log) == 0


def main():
    parser = argparse.ArgumentParser(description="AMS workflow deployment")
    parser.add_argument(
        "-v",
        "--verbose",
        dest="verbose",
        help="Set verbosity level",
        choices=[k for k in logging._nameToLevel.keys()],
        default="INFO",
    )
    parser.add_argument(
        "-l", "--log-file", dest="log_file", help="Path to file to store logs (when unspecified stdout/err is used)"
    )
    sub_parsers = parser.add_subparsers(dest="command", help="Commands supported by ams deployment tool")
    sub_parsers.required = True
    start_cli(sub_parsers)
    generate_cli(sub_parsers)
    validate_cli(sub_parsers)

    args = parser.parse_args()
    logger_fmt = logging.Formatter(
        "%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d:%H:%M:%S",
    )
    if args.log_file is not None:
        log_handler = logging.FileHandler(
            args.log_file,
            mode="a",
        )
    else:
        log_handler = logging.StreamHandler(sys.stdout)

    log_handler.setFormatter(logger_fmt)
    logger.addHandler(log_handler)
    logger.setLevel(logging._nameToLevel[args.verbose])
    logger.propagate = False

    ret = not args.func(args)
    return ret


if __name__ == "__main__":
    sys.exit(main())
