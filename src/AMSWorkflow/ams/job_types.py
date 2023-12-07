from dataclasses import dataclass
from pathlib import Path
import os
import sys
import shutil
from warnings import warn
from typing import List, Dict, Optional, ClassVar
from flux.job import JobspecV1
import flux.job as fjob

from ams.loader import load_class


@dataclass(kw_only=True)
class BaseJob:
    """
    Class Modeling a Job scheduled by AMS. There can be five types of JOBs (Physics, Stagers, Training, RMQServer and TrainingDispatcher)
    """

    name: str
    executable: str
    nodes: int
    tasks_per_node: int
    args: List[str] = list()
    exclusive: bool = True
    cores_per_task: int = 1
    environ: Dict[str, str] = dict()
    orderId: ClassVar[int] = 0
    gpus_per_task: Optional[int] = None
    stdout: Optional[str] = None
    stderr: Optional[str] = None

    def _construct_command(self):
        command = [self.executable] + self.args
        return command

    def _construct_environ(self, forward_environ):
        environ = self.environ
        if forward_environ is not None:
            if not isinstance(forward_environ, type(os.environ)) and not isinstance(forward_environ, dict):
                raise TypeError(f"Unsupported forward_environ type ({type(forward_environ)})")
            for k, v in forward_environ:
                if k in environ:
                    warn(f"Key {k} already exists in environment ({environ[k]}), prioritizing existing one ({v})")
                else:
                    environ[k] = forward_environ[k]
        return environ

    def _construct_redirect_paths(self, redirectDir):
        stdDir = Path.cwd()
        if redirectDir is not None:
            stdDir = Path(redirectDir)

        if self.stdout is None:
            stdout = f"{stdDir}/{self.name}_{BaseJob.orderId}.out"
        else:
            stdout = f"{stdDir}/{self.stdout}_{BaseJob.orderId}.out"

        if self.stderr is None:
            stderr = f"{stdDir}/{self.name}_{BaseJob.orderId}.err"
        else:
            stderr = f"{stdDir}/{self.stderr}_{BaseJob.orderId}.err"

        BaseJob.orderId += 1

        return stdout, stderr

    def schedule(self, flux_handle, forward_environ=None, redirectDir=None, pre_signed=False, waitable=True):
        jobspec = JobspecV1.from_command(
            command=self._construct_command(),
            num_tasks=self.tasks_per_node * self.nodes,
            num_nodes=self.nodes,
            cores_per_task=self.cores_per_task,
            gpus_per_task=self.gpus_per_task,
            exclusive=self.exclusive,
        )

        stdout, stderr = self._construct_redirect_paths(redirectDir)
        environ = self._construct_environ(forward_environ)
        jobspec.environment = environ
        jobspec.stdout = stdout
        jobspec.stderr = stderr

        return jobspec, fjob.submit(flux_handle, jobspec, pre_signed=pre_signed, waitable=waitable)


@dataclass(kw_only=True)
class PhysicsJob(BaseJob):
    def _verify(self):
        is_executable = shutil.which(self.executable) is not None
        is_path = Path(self.executable).is_file()
        return is_executable or is_path

    def __post_init__(self):
        if not self._verify():
            raise RuntimeError(
                f"[PhysicsJob] executable is neither a executable nor a system command {self.executable}"
            )


@dataclass(kw_only=True, init=False)
class Stager(BaseJob):
    def _get_stager_default_cores(self):
        """
        We need the following cores:
            1 RMQ Client to receive messages
            1 Process to store to filesystem
            1 Process to make public to kosh
        """
        return 3

    def _verify(self, pruner_path, pruner_cls):
        assert Path(pruner_path).is_file(), "Path to Pruner class should exist"
        user_class = load_class(pruner_path, pruner_cls)
        print(f"Loaded Pruner Class {user_class.__name__}")

    def __init__(
        self,
        name: str,
        num_cores: int,
        db_path: str,
        pruner_cls: str,
        pruner_path: str,
        pruner_args: List[str],
        num_gpus: Optional[int],
        **kwargs,
    ):
        executable = sys.executable

        self._verify(pruner_path, pruner_cls)

        # TODO: Here we are accessing both the stager arguments and the pruner_arguments. Is is an oppotunity to emit
        # an early error message. But, this would require extending argparse or something else. Noting for future reference
        cli_arguments = [
            "-m",
            "ams_wf.AMSDBStage",
            "-db",
            db_path,
            "--policy",
            "process",
            "--dest",
            str(Path(db_path) / Path("candidates")),
            "--db-type",
            "dhdf5",
            "--store",
            "-m",
            "fs",
            "--class",
            pruner_cls,
        ]
        cli_arguments += pruner_args

        num_cores = self._get_stager_default_cores() + num_cores
        super().__init__(
            name=name,
            executable=executable,
            nodes=1,
            tasks_per_node=1,
            cores_per_task=num_cores,
            args=cli_arguments,
            gpus_per_task=num_gpus,
            **kwargs,
        )
