import sys
import json
from pathlib import Path
import os

from ams.rmq import BlockingClient, default_ams_callback

def verify(
    use_device,
    num_inputs,
    num_outputs,
    data_type,
    num_iterations,
    num_elements,
    rmq_json,
    timeout = None,
    domain_test = "rmq_db_no_model" # defined in ams_rmq_env.cpp
):
    host = rmq_json["service-host"]
    vhost = rmq_json["rabbitmq-vhost"]
    port = rmq_json["service-port"]
    user = rmq_json["rabbitmq-user"]
    password = rmq_json["rabbitmq-password"]
    queue = rmq_json["rabbitmq-outbound-queue"]
    cert = None
    if "rabbitmq-cert" in rmq_json:
        cert = rmq_json["rabbitmq-cert"]
        cert = None if cert == "" else cert

    dtype = 4
    if data_type == "double":
        dtype = 8

    with BlockingClient(host, port, vhost, user, password, cert, default_ams_callback) as client:
        with client.connect(queue) as channel:
            msgs = channel.receive(n_msg = num_iterations, timeout = timeout)

    assert len(msgs) == num_iterations, f"Received incorrect number of messsages ({len(msgs)}): expected #msgs ({num_iterations})"

    for i, msg in enumerate(msgs):
        domain, _, _ = msg.decode()
        assert msg.num_elements == num_elements, f"Message #{i}: incorrect #elements ({msg.num_element}) vs. expected #elem {num_elements})"
        assert msg.input_dim == num_inputs, f"Message #{i}: incorrect #inputs ({msg.input_dim}) vs. expected #inputs {num_inputs})"
        assert msg.output_dim == num_outputs, f"Message #{i}: incorrect #outputs ({msg.output_dim}) vs. expected #outputs {num_outputs})"
        assert msg.dtype_byte == dtype, f"Message #{i}: incorrect datatype ({msg.dtype_byte} bytes) vs. expected type {dtype} bytes)"
        assert domain == domain_test, f"Message #{i}: incorrect domain name (got {domain}) expected rmq_db_no_model)"

    return 0

def from_json(argv):
    print(argv)
    use_device = int(argv[0])
    num_inputs = int(argv[1])
    num_outputs = int(argv[2])
    data_type = argv[3]
    num_iterations = int(argv[4])
    num_elements = int(argv[5])

    env_file = Path(os.environ["AMS_OBJECTS"])
    if not env_file.exists():
        print("Environment file does not exist")
        return -1

    with open(env_file, "r") as fd:
        rmq_json = json.load(fd)

    res = verify(
        use_device,
        num_inputs,
        num_outputs,
        data_type,
        num_iterations,
        num_elements,
        rmq_json["db"]["rmq_config"],
        timeout = 60 # in seconds
    )
    if res != 0:
        return res
    print("[Success] rmq test received")
    return 0

if __name__ == "__main__":
    if "AMS_OBJECTS" in os.environ:
        sys.exit(from_json(sys.argv[1:]))
    sys.exit(1)
