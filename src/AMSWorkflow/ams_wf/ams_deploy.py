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
        result = sp.run(bootstrap_cmd, shell=True)
        return result.returncode
        # NOTE: From this point on we should definetely not use the logging mechanism. We manually shut it donw
        # to allo the bootstrapped script to use the same logger (this is important in the case of logging into a file)

    logging.info(f"Bootstrapping using {scheduler.name}")

    if scheduler == RootSched.SLURM:
        slurm_bootstrap(cmd, flux_log)
    else:
        logging.critical("Unknown scheduler, cannot bootstrap")
        sys.exit()
    return 0


class AMSConfig:
    @staticmethod
    def validate(config):
        def validate_keys(level, config, mandatory_fields):
            if not all(field in config.keys() for field in mandatory_fields):
                missing_fields = " ".join([v for v in mandatory_fields if v not in config.keys()])
                logging.critical(f"The following fields are missing : {missing_fields} from entry {level}")
                return False
            return True

        def validate_step_field(level, config):
            if not validate_keys(level, config, ["executable", "resources"]):
                logging.critical(f"Mising fields in {level}")
                return False

            exec_path = Path(config["executable"])
            if not exec_path.exists():
                logging.critical("Executable {exec_path} does not exist")
                return False

            if not validate_keys(level, config["resources"], ["num_nodes", "num_processes_per_node"]):
                logging.critical(f"Missing fields in resources of {level}")
                return False

            return True

        if not validate_keys("root", config, ["user_app", "ml_training", "ml_pruning", "execution_mode", "db", "stager"]):
            return False

        if not validate_step_field("user_app", config["user_app"]):
            return False

        if not validate_step_field("ml_training", config["ml_training"]):
            return False

        if not validate_step_field("ml_pruning", config["ml_pruning"]):
            return False

        if not validate_keys("ml_training|resources", config["ml_training"]["resources"], ["num_nodes", "num_processes_per_node"]):
            return False

        if not validate_keys("ml_pruning|resources", config["ml_training"]["resources"], ["num_nodes", "num_processes_per_node"]):
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
        if config["stager"]["mode"]:
            rmq_config = config["rmq"]
            if not isinstance(rmq_config["service-port"], int):
                print(isinstance(int, type(rmq_config["service-port"])))
                print(rmq_config["service-port"], type(rmq_config["service-port"]), int)
                logging.critical("The RMQ service-port must be an integer type {0}".format(type(rmq_config["service-port"])))
                return False
            if not Path(rmq_config["rabbitmq-cert"]).exists():
                logging.critical("The RMQ certificate file does not exist (or is not not accessible)")
                return False

            rmq_keys = AMSConfig.to_descr()["rmq"].keys()

            if not validate_keys("rmq", config["rmq"], rmq_keys):
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
            "db": {"path": "path/to/db"},
            "stager": {"mode": "filesystem", "num_clients": "number of rmq clients (mandatory only when mode is rmq)"},
            "rmq" : {
                "service-port": "Port",
                "service-host": "server address",
                "rabbitmq-erlang-cookie": "magic cookie",
                "rabbitmq-name": "rmq server name",
                "rabbitmq-password": "password",
                "rabbitmq-user": "user",
                "rabbitmq-vhost": "virtual host",
                "rabbitmq-cert": "path to certificate to establish connection",
                "rabbitmq-inbound-queue": "Queue name to send data from outside in the simulation",
                "rabbitmq-outbound-queue": "Queue name to send data from the simulation to outside"
            }
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
    result = sp.run(cmd, shell=True)
    if result.returncode != 0:
        logging.warning(f"{editor} {args.config} returned non zero code")

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
        logging.info("Generated configuration file is NOT valid")
        return False
    logging.info("Generated configuration file IS valid")
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

    if is_bootstrapped():
        logging.info("Execution is bootstrapped")
        return False

    with open(args.config, "r") as fd:
        data = json.load(fd)

    if not validate_config(data):
        logging.info("Configuration file is not valid, exiting early...")
        return False

    logging.info("Execution is NOT bootstrapped")
    cmd = get_cmd()
    return (bootstrap(cmd, RootSched[args.scheduler], args.flux_log) == 0)


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

    return not args.func(args)


if __name__ == "__main__":
    sys.exit(main())
