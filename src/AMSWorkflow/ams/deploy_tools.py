import sys
import os
import select
import subprocess as sp
from enum import Enum


class RootSched(Enum):
    SLURM = 1
    LSF = 2


def _run_daemon(cmd, shell=False):
    print(f"Going to run {cmd}")
    proc = sp.Popen(cmd, shell=shell, stdout=sp.PIPE, stderr=sp.PIPE, bufsize=1, text=True, universal_newlines=True)
    return proc


def _read_flux_uri(proc, timeout=5):
    """
    Reads the first line from the flux start command's stdout and puts it into a queue.
    :param timeout: The maximum of time we wait for writting to stdout
    :param proc: The process from which to read stdout.
    """

    # Time to wait for I/O plus the time already waited
    total_wait_time = 0
    poll_interval = 0.5  # Poll interval in seconds

    while total_wait_time < timeout:
        # Check if there is data to read from stdout
        ready_to_read = select.select([proc.stdout], [], [], poll_interval)[0]
        if ready_to_read:
            first_line = proc.stdout.readline()
            print("Frist line is", first_line)
            if "ssh" in first_line:
                return first_line
        total_wait_time += poll_interval
    print(f"Waited for {total_wait_time}")
    return None


def spawn_rmq_broker(flux_uri):
    # TODO We need to implement this, my current specification is limited
    # We probably need to access to flux, to spawn a daemon inside the flux allocation
    raise NotImplementedError("spawn_rmq_broker is not implemented, spawn it manually and provide the credentials")
    return None, None


def start_flux(scheduler, nnodes=None):
    def bootstrap_with_slurm(nnodes):
        if nnodes is None:
            nnodes = os.environ.get("SLURM_NNODES", None)

        bootstrap_cmd = f"srun -N {nnodes} -n {nnodes} --pty --mpi=none --mpibind=off flux start"
        flux_get_uri_cmd = "flux uri --remote \\$FLUX_URI; sleep inf"

        daemon = _run_daemon(f'{bootstrap_cmd} "{flux_get_uri_cmd}"', shell=True)
        flux_uri = _read_flux_uri(daemon, timeout=10)
        print("Got flux uri: ", flux_uri)
        if flux_uri is None:
            print("Fatal Error, Cannot read flux")
            daemon.terminate()
            raise RuntimeError("Cannot Get FLUX URI")

        return daemon, flux_uri

    if scheduler == RootSched.SLURM:
        return bootstrap_with_slurm(nnodes)

    raise NotImplementedError("We are only supporting bootstrap through SLURM")
