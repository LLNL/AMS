from flux.job import JobspecV1
import os

from typing import Optional
from dataclasses import dataclass, fields
from ams import util


def constuct_cli_cmd(executable, *args, **kwargs):
    command = [executable]
    for k, v in kwargs.items():
        command.append(str(k))
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

    def to_dict(self):
        return {field.name: getattr(self, field.name) for field in fields(self)}


class AMSJob:
    """
    Class Modeling a Job scheduled by AMS.
    """

    @classmethod
    def generate_formatting(self, store):
        return {"AMS_STORE_PATH": store.root_path}

    def __init__(
        self,
        name,
        executable,
        environ={},
        resources=None,
        stdout=None,
        stderr=None,
        ams_log=False,
        is_mpi=False,
        cli_args=[],
        cli_kwargs={},
    ):
        self._name = name
        self._executable = executable
        self._resources = resources
        if isinstance(self._resources, dict):
            self._resources = AMSJobResources(**resources)

        self.environ = environ
        self._stdout = stdout
        self._stderr = stderr
        self._cli_args = []
        self._cli_kwargs = {}
        self._is_mpi = is_mpi
        if cli_args is not None:
            self._cli_args = list(cli_args)
        if cli_kwargs is not None:
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

    @classmethod
    def from_dict(cls, _dict):
        return cls(**_dict)

    def to_dict(self):
        data = {}
        data["name"] = self._name
        data["executable"] = self._executable
        data["stdout"] = self._stdout
        data["stderr"] = self._stderr
        data["cli_args"] = self._cli_args
        data["cli_kwargs"] = self._cli_kwargs
        data["resources"] = self._resources.to_dict()
        return data

    def to_flux_jobspec(self):
        jobspec = JobspecV1.from_command(
            command=self.generate_cli_command(),
            num_tasks=self.resources.tasks_per_node * self.resources.nodes,
            num_nodes=self.resources.nodes,
            cores_per_task=self.resources.cores_per_task,
            gpus_per_task=self.resources.gpus_per_task,
            exclusive=self.resources.exclusive,
        )

        if self._is_mpi:
            jobspec.setattr_shell_option("mpi", "spectrum")
            jobspec.setattr_shell_option("gpu-affinity", "per-task")
        if self._stdout is not None:
            jobspec.stdout = "ams_test.out"
        if self._stderr is not None:
            jobspec.stderr = "ams_test.err"

        jobspec = self.environ

        return jobspec


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
            models = store.search(domain_name=name, entry="models", version="latest")
            print(json.dumps(models, indent=6))
            # This is the case in which we do not have any model
            # Thus we create a data gathering entry
            if len(models) == 0:
                model_entry = {
                    "uq_type": "random",
                    "model_path": "",
                    "uq_aggregate": "mean",
                    "threshold": 1,
                    "db_label": name,
                }
            else:
                model = models[0]
                model_entry = {
                    "uq_type": model["uq_type"],
                    "model_path": model["file"],
                    "uq_aggregate": "mean",
                    "threshold": model["threshold"],
                    "db_label": name,
                }

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
    def from_descr(cls, stage_dir, descr):
        domain_job_resources = AMSJobResources(**descr["resources"])
        return cls(
            name=descr["name"],
            stage_dir=stage_dir,
            domain_names=descr["domain_names"],
            environ=None,
            resources=domain_job_resources,
            **descr["cli"],
        )

    def precede_deploy(self, store):
        self._ams_object = self._generate_ams_object(store)
        tmp_path = util.mkdir(store.root_path, "tmp")
        self._ams_object_fn = f"{tmp_path}/{util.get_unique_fn()}.json"
        with open(self._ams_object_fn, "w") as fd:
            json.dump(self._ams_object, fd)
        self.environ["AMS_OBJECTS"] = str(self._ams_object_fn)


class AMSMLJob(AMSJob):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def from_descr(cls, store, descr):
        formatting = AMSJob.generate_formatting(store)
        resources = AMSJobResources(**descr["resources"])
        cli_kwargs = descr["cli"].get("cli_kwargs", None)
        if cli_kwargs is not None:
            for k, v in cli_kwargs.items():
                if isinstance(v, str):
                    cli_kwargs[k] = v.format(**formatting)
        cli_args = descr["cli"].get("cli_args", None)
        if cli_args is not None:
            for i, v in enumerate(cli_args):
                cli_args[i] = v.format(**formatting)

        return cls(
            name=descr["name"],
            environ=None,
            stdout=descr["cli"].get("stdout", None),
            stderr=descr["cli"].get("stderr", None),
            executable=descr["cli"]["executable"],
            resources=resources,
            cli_kwargs=cli_kwargs,
            cli_args=cli_args,
        )


class AMSMLTrainJob(AMSMLJob):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class AMSSubSelectJob(AMSMLJob):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


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
        _cli_kwargs["--dest"] = dest_dir
        _cli_kwargs["--src"] = src_dir
        _cli_kwargs["--pattern"] = "*.h5"
        _cli_kwargs["--db-type"] = "dhdf5"
        _cli_kwargs["--mechanism"] = "fs"
        _cli_kwargs["--policy"] = "process"
        _cli_kwargs["--persistent-db-path"] = store_dir
        _cli_kwargs["--src"] = src_dir

        if prune_module_path is not None:
            assert Path(prune_module_path).exists(), "Module path to user pruner does not exist"
            _cli_kwargs["--load"] = prune_module_path
            _cli_kwargs["--class"] = prune_class

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


def nested_instance_job_descr(num_nodes, cores_per_node, gpus_per_node, time="inf", stdout=None, stderr=None):
    jobspec = JobspecV1.from_nest_command(
        command=["sleep", time],
        num_slots=num_nodes,
        num_nodes=num_nodes,
        cores_per_slot=cores_per_node,
        gpus_per_slot=gpus_per_node,
        # NOTE: This is set to true, cause we do not want the parent partion to
        # schedule other jobs to the same resources and allow the "partion" to
        # have exclusive ownership of the resources. We should rethink this,
        # as it may make the system harder to debug
        exclusive=True,
    )

    if stdout is not None:
        jobspec.stdout = stdout
    if stderr is not None:
        jobspec.stderr = stderr
    jobspec.cwd = os.getcwd()
    return jobspec


def get_echo_job(message):
    jobspec = JobspecV1.from_command(
        command=["pwd"],
        num_tasks=1,
        num_nodes=1,
        cores_per_task=1,
        gpus_per_task=0,
        exclusive=True,
    )
    return jobspec
    return jobspec
    return jobspec
