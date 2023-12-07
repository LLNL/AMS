import os
from pathlib import Path
import sys
import argparse
from enum import Enum
import logging
import subprocess as sp
import json


class RootSched(Enum):
    SLURM = 1
    LSF = 2


def bootstrap(cmd, scheduler, flux_log):
    def slurm_bootstrap(cmd, flux_log_file):
        nnodes = os.environ.get("SLURM_NNODES", None)
        if nnodes == None:
            logging.critical("Environemnt variable 'SLURM_NNODES' is not set, cannot deduce flux number of nodes")
            sys.exit()

        bootstrap_cmd = f"srun -N {nnodes} -n {nnodes} --pty --mpi=none --mpibind=off flux start"

        if flux_log_file is not None:
            bootstrap_cmd = f"{bootstrap_cmd} -o,S,log-filename=${flux_log_file}"
        bootstrap_cmd = f"{bootstrap_cmd} {cmd}"
        logging.debug(f"Executing command {bootstrap_cmd}")
        logging.shutdown()
        sp.run(bootstrap_cmd, shell=True)
        # NOTE: From this point on we should definetely not use the logging mechanism. We manually shut it donw
        # to allo the bootstrapped script to use the same logger (this is important in the case of logging into a file)

    logging.info(f"Bootstrapping using {scheduler.name}")

    if scheduler == RootSched.SLURM:
        slurm_bootstrap(cmd, flux_log)
    else:
        logging.critical("Unknown scheduler, cannot bootstrap")
        sys.exit()


class AMSConfig:
    @staticmethod
    def validate(config):
        def validate_keys(level, config, mandatory_fields):
            if not all(field in config.keys() for field in mandatory_fields):
                missing_fields = " ".join([v for v in mandatory_fields if v not in config.keys()])
                logging.critical(f"The following fields are missing : {missing_fields} from entry {level}")
                return False
            return True

        if not validate_keys("root", config, ["user_app", "ml_training", "execution_mode", "db", "stager"]):
            return False

        if not validate_keys("user_app", config["user_app"], ["executable", "resources"]):
            return False

        exec_path = Path(config["user_app"]["executable"])
        if not exec_path.exists():
            logging.critical("Executable {exec_path} does not exist")
            return False

        if not validate_keys(
            "user_app|resources", config["user_app"]["resources"], ["num_nodes", "num_processes_per_node"]
        ):
            return False

        if not validate_keys("ml_training", config["ml_training"], ["num_nodes", "num_processes_per_node"]):
            return False

        if config["execution_mode"] not in ["sequential", "concurrent"]:
            logging.critical("Unknown 'execution_mode', please select from 'sequential', 'concurrent'")
            return False

        if config["execution_mode"] == "concurrent":
            if config["stager"]["mode"] == "filesystem":
                logging.critical("Database is concurrent but the stager polls data from filesystem")
                return False
            elif config["stager"]["mode"] == "rmq":
                if "num_clients" not in config["stager"]:
                    logging.critical("When stager set in mode 'rmq' you need to define the number of rmq clients")
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
            "ml_training": {"num_nodes": "XX", "num_processes_per_node": "YY", "num_gpus_per_node": "ZZ"},
            "execution_mode": "sequential",
            "db": {"path": "path/to/db"},
            "stager": {"mode": "filesystem", "num_clients": "number of rmq clients (mandatory only when mode is rmq)"},
        }


def generate_cli(parser):
    generate_parser = parser.add_parser("generate", help="Generate an AMS workflow configuration file")
    generate_parser.add_argument(
        "--config", "-c", dest="config", required=True, help="Path to the AMS file to be generated"
    )
    generate_parser.set_defaults(func=generate_config)


def generate_config(args):
    logging.info(f"Generating configuration file {args.config}")
    with open(args.config, "w") as fd:
        json.dump(AMSConfig.to_descr(), fd, indent=6)
    editor = os.environ.get("EDITOR", None)
    if editor is None:
        logging.critical(f"Environemnt variable EDITOR is not set, example configuration is stored in {args.config}")
        sys.exit()
    cmd = f"{editor} {args.config}"
    sp.run(cmd, shell=True)
    with open(args.config, "r") as fd:
        data = json.load(fd)

    if not AMSConfig.validate(data):
        logging.critical("Generated configuration file is not valid")


def validate_cli(parser):
    validate_parser = parser.add_parser("validate", help="Validate an AMS configuration file")
    validate_parser.add_argument("--config", "-c", dest="config", required=True, help="Path to configuration file")
    validate_parser.set_defaults(func=validate_config)


def validate_config(args):
    with open(args.config, "r") as fd:
        data = json.load(fd)

    if not AMSConfig.validate(data):
        logging.critical("Generated configuration file is not valid")


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

    if is_bootstrapped():
        logging.info("Execution is bootstrapped")
        return

    logging.info("Execution is NOT bootstrapped")
    cmd = get_cmd()
    bootstrap(cmd, RootSched[args.scheduler], args.flux_log)
    return


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
    sub_parsers = parser.add_subparsers(help="Commands supported by ams deployment tool")
    start_cli(sub_parsers)
    generate_cli(sub_parsers)

    args = parser.parse_args()
    if args.log_file is not None:
        logging.basicConfig(
            format="%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
            datefmt="%Y-%m-%d:%H:%M:%S",
            level=args.verbose,
            filename=args.log_file,
            filemode="a",
        )
    else:
        logging.basicConfig(
            format="%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
            datefmt="%Y-%m-%d:%H:%M:%S",
            level=args.verbose,
        )

    args.func(args)


if __name__ == "__main__":
    main()
