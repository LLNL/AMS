# Tools to interact with RabbitMQ

This folder contains several scripts to send and receive messages using
RabbitMQ. They are useful to test and interact with AMSlib.

## Send a message to AMSlib
If you want to send a message to AMSlib, for example to force AMS to update its
surrogate, you can do it using `send.py`:

```bash
./send.py -c rmq-pds.json -t rmq-pds.crt -e ams-fanout -r training -n 1 -m
"UPDATE:ConstantOneModel_cpu.pt"
```

where `rmq-pds.json` contains the RabbitMQ credentials and `rmq-pds.crt` the
TLS certificate. 

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

> Note that you can use `send.py` to send any type of message to any RabbitMQ
> server.

## Consume messages from AMSlib

To receive, or consume, messages emitted by AMS you can use `recv_binary.py`:

```bash
./recv_binary.py -c rmq-pds.json -t rmq-pds.crt -q test3
```


