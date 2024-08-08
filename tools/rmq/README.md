# Tools to interact with RabbitMQ

This folder contains several scripts to send and receive messages using
RabbitMQ. They are useful to test and interact with AMSlib. Each script
is completetly standalone and does not require the AMS Python package,
however they require `pika` and `numpy`.

## Generate TLS certificate

To use most of the tools related to RabbitMQ you might need to provide TLS certificates.
To generate such certificate you can use OpenSSL, for example:

```bash
    openssl s_client -connect $REMOTE_HOST:$REMOTE_PORT -showcerts < /dev/null 2>/dev/null | sed -ne '/-BEGIN CERTIFICATE-/,/-END CERTIFICATE-/p' > rmq-tls.crt
```
where `REMOTE_HOST` is the hostname of the RabbitMQ server and `REMOTE_PORT` is the port.

## Consume messages from AMSlib

To receive, or consume, messages emitted by AMSlib you can use `recv_binary.py`:

```bash
python3 recv_binary.py -c rmq-credentials.json -t rmq-tls.crt -q test3
```

If the credentials match, every messages sent by a simulation integrated
with AMS will be received by `recv_binary.py`.

## Send a message to AMSlib

### Send string messages
To send a simple text message to AMSlib, for example to force AMS to update its
surrogate model, you can do it using `send.py`:

```bash
python3 send.py -c rmq-credentials.json -t rmq-tls.crt -e ams-fanout -r training -n 1 -m
"UPDATE:ConstantOneModel_cpu.pt"
```

where `rmq-pds.json` contains the RabbitMQ credentials and `rmq-pds.crt` the
TLS certificate. See `send.py -h` for more options.

The RabbitMQ credentials file must follow this template:
```json
{
    "rabbitmq-erlang-cookie": "",
    "rabbitmq-name": "",
    "rabbitmq-password": "",
    "rabbitmq-user": "",
    "rabbitmq-vhost": "",
    "service-port": 0,
    "service-host": "",
}
```

> Note that you can use `send.py` to send any type of string messages to any RabbitMQ
> server.

### Send binary-compatible AMSlib messages

To send a message that mimics what each MPI rank in AMSlib would send to
the AMS Python module, one can use `send_ams.py`. For example,

```bash
python3 send_ams.py -c rmq-credentials.json -t rmq-tls.crt -r test3 -n 10
```

In another terminal, to receive the message just sent, you can run:

```bash
python3 recv_binary.py -c rmq-credentials.json -t rmq-tls.crt -q test3
```

This tool is useful to test the AMS python workflow without
actually running a simulation.
