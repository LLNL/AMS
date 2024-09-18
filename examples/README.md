# AMSLib Example

The example code can be used as a mini-application to understand the main AMS functionalities.
To enable compilation of the example code please add the `-DWITH_EXAMPLE=On` flag on your cmake command
or enable the `+examples` spack variant.

## Example workflow steps of AMS approach

1. Execute the example code without any model, with the database support enabled (`-DWITH_DB=On`, '+hdf5' for spack).
  ```
   ./examples/ams_example -db <PATH-TO-EXISTING-DIRECTORY> -dt hdf5
  ```
2. Use py-torch to train a model with the data under the '<PATH-TO-EXISTING-DIRECTORY>' and store the model to [torch jit format](https://pytorch.org/tutorials/advanced/cpp_export.html) to some file '<MODEL-FILE>'

3. Run the example code with the model as a parameter:
  ```
   ./examples/ams_example -db <PATH-TO-EXISTING-DIRECTORY> -dt hdf5 -S '<MODEL-FILE>'
  ```

## The AMS Library Database

AMS supports multiple database back-ends and formats. We currently use mainly `hdf5` however there exist 
experimental `Redis` and a RabbitMQ back-end.

### HDF5 backend

To enable the HDF5 database backend please pass the following flags to the cmake command:

```bash
-DWITH_HDF5=On \
-DHDF5_Dir=$AMS_HDF5_PATH \
```


### Redis backend

If you want to use Redis as database back-end you will have to perform
additional steps.

1. CMake flags -- you will have to indicate where _Hiredis_ and _Redis-plus-plus_
are located with adding to the `cmake` command line:
```bash
-DWITH_DB=On -DWITH_REDIS=On
```

> Note that `hiredis` and `redis-plus-plus` should already be installed if you use the shared Spack installation.

2. Generate OpenSSL certificates -- To access Redis DBs located in PDS, you will need to use TLS and have a valid certificate:

```bash
openssl s_client -connect $REMOTE_HOST:$REMOTE_PORT -showcerts < /dev/null 2>/dev/null | sed -ne '/-BEGIN CERTIFICATE-/,/-END CERTIFICATE-/p' > redis_certificate.crt
```
where `$REMOTE_HOST` is the host hosting the DB and `$REMOTE_PORT` is the port number. For example, `REMOTE_HOST=some.url.somewhere` and `REMOTE_PORT=32236`.

3. Provide the right information to the `miniapp`. The following JSON
   information (password, port and hostname) can be obtained on https://launchit.llnl.gov/.
```json
{
    "database-password": "PASSWORD",
    "service-port": 32245,
    "host": "Host Address",
    "cert": "redis-certificate.crt"
}
```
> You will have to provide the path of the JSON configuration as follows `--dbconfig <PATH-TO-CONFIG>`.

### RabbitMQ

If you want to use RabbitMQ as database back-end you will have to perform additional steps.

- It requires [AMQP-CPP](https://github.com/CopernicaMarketingSoftware/AMQP-CPP), [libevent](https://libevent.org/) and OpenSSL.
- To compiles with that option you will need to add `-DWITH_DB=On -DWITH_RMQ=On -Damqpcpp_DIR=$AMS_AMQPCPP_PATH` to your existing CMake command line
- It requires a RabbitMQ server service running somewhere (credentials must be in JSON and given to the application via `-dt rmq -db creds.json`. The credentials `creds.json` should be looking like that:
```json
{
    "rabbitmq-erlang-cookie": "",
    "rabbitmq-name": "",
    "rabbitmq-password": "",
    "rabbitmq-user": "",
    "rabbitmq-vhost": "/",
    "service-port": 1234,
    "service-host": "",
    "rabbitmq-cert": "creds.pem",
    "rabbitmq-queue-physics": "ams-data",
    "rabbitmq-exchange-training": "ams-training",
    "rabbitmq-key-training": "training"
}
```
`rabbitmq-cert` is where the TLS certificate is (absolute path ideally), `rabbitmq-queue-physics` is the name of the queue used by AMS.
You can use for testing the credentials I have pre-generated  to access a RabbitMQ server located in PDS here : `/usr/workspace/AMS/pds/rabbitmq/`.

Known issues (that I am working on):
- [ ] Slow when sending data, I am working on linearizing everything so we only perform one or two sends (one for inputs, one for outputs or just one for everything per cycle).

