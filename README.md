# marbl material proporties mini-miniapp

## Setup and Build using shared `spack`

All dependencies needed for the `miniapp` are installed through `spack` on a shared
space `/usr/workspace/AMS`.

Steps to build the code.

1. Clone the code repository.
```bash
$ git clone ssh://git@czgitlab.llnl.gov:7999/autonomous-multiscale-project/marbl-matprops-miniapp.git
$ cd marbl-matprops-miniapp
```

2. Setup the environment -- the following script will load all dependencies and
setup some relevant paths.
```bash
$ source setup/setup_env.sh
```

3. Build the mini-app. Please note the optional features that can be turned off. We currently support 3 types of databases, hdf5 is the stable one, redis and csv are still not tested.

```bash
$ mkdir build; cd build
$ cmake \
-DWITH_HDF5=On \
-DHDF5_Dir=$AMS_HDF5_PATH \
-DCMAKE_PREFIX_PATH=$INSTALL_DIR \
-DWITH_DB=On \
-DCMAKE_INSTALL_PREFIX=./install \
-DCMAKE_BUILD_TYPE=Debug \
-DWITH_EXAMPLES=On \
-DWITH_CUDA=On \
-DUMPIRE_DIR=$AMS_UMPIRE_PATH \
-DWITH_MPI=On \
-DMFEM_DIR=$AMS_MFEM_PATH \
-DWITH_FAISS=On \
-DWITH_CALIPER=On \
-DWITH_MPI=On \
-DFAISS_DIR=$AMS_FAISS_PATH \
-DWITH_TORCH=On \
-DTorch_DIR=$AMS_TORCH_PATH \
-DAMS_CUDA_ARCH=${AMS_CUDA_ARCH} \
../

$ make -j6
```
Most of the compile time options are optional. You can turn-off them to decrease the build time significantly.

### Issue with MPI on specturm-mpi and direct GPU

Currently the software stack (0.19) uses cuda@11.6 however,
the latest cuda that supports GPU-Direct for specturm-mpi is cuda@11.2. Thus, currently
we may have issues running Direct GPU on systems.

### Run Mini-app

To run the mini-app use any of these options.
```bash
Usage: ./build/examples/ams_example [options] ...
Options:
   -h, --help
	Print this help message and exit.
   -d <string>, --device <string>, current value: cpu
	Device config string
   -S <string>, --surrogate <string>, current value:
	Path to surrogate model
   -H <string>, --hdcache <string>, current value:
	Path to hdcache index
   -z <string>, --eos <string>, current value: ideal_gas
	EOS model type
   -c <int>, --stop-cycle <int>, current value: 1
	Stop cycle
   -m <int>, --num-mats <int>, current value: 5
	Number of materials
   -e <int>, --num-elems <int>, current value: 10000
	Number of elements
   -q <int>, --num-qpts <int>, current value: 64
	Number of quadrature points per element
   -r <double>, --empty-element-ratio <double>, current value: -1
	Fraction of elements that are empty for each material. If -1 use a random value for each.
   -s <int>, --seed <int>, current value: 0
	Seed for rand
   -p, --pack-sparse, -np, --do-not-pack-sparse, current option: --pack-sparse
	pack sparse material data before evals (cpu only)
   -i, --with-imbalance, -ni, --without-imbalance, current option: --without-imbalance
	Create artificial load imbalance across ranks
   -avg <double>, --average <double>, current value: 0.5
	Average value of random number generator of imbalance threshold
   -std <double>, --stdev <double>, current value: 0.2
	Standard deviation of random number generator of imbalance threshold
   -t <double>, --threshold <double>, current value: 0.5
	Threshold value used to control selection of surrogate vs physics execution
   -db <string>, --dbconfig <string>, (required)
	Path to directory where applications will store their data
   -v, --verbose, -qu, --quiet, current option: --quiet
	Print extra stuff

  Print extra stuff
```

```bash
$ ./build/examples/ams_example -S /usr/workspace/AMS/miniapp_resources/trained_models/debug_model.pt
```
  **TODO:** add instructions on command line options!

## I/O backends

### Files

If you want to use plain files you just have to use the flag `-DWITH_DB=On`
when running `cmake`.


### HDF5 backend

To enable the HDF5 database backend please pass the following flags to the cmake command:

```bash
-DWITH_HDF5=On \
-DHDF5_Dir=$AMS_HDF5_PATH \
```


### Redis backend

If you want to use Redis as database back-end you will have to perform
additional steps. In this case, we consider that the Redis DB is located within
LC in PDS.

1. CMake flags -- you will have to indicate where _Hiredis_ and _Redis-plus-plus_
are located with adding to the `cmake` command line:
```bash
-DWITH_DB=On -DWITH_REDIS=On
```

> Note that `hiredis` and `redis-plus-plus` should already be installed if you use the shared Spack installation.

2. Generate OpenSSL certificates -- To access Redis DBs located in PDS, you will need to use TLS and have a valid certificate.
If you run on LC system, you can use the information located at `/usr/workspace/AMS/miniapp_resources/redis`.

If you do not have access to this folder or, if you want to generate your own certificate:
```bash
openssl s_client -connect $REMOTE_HOST:$REMOTE_PORT -showcerts < /dev/null 2>/dev/null | sed -ne '/-BEGIN CERTIFICATE-/,/-END CERTIFICATE-/p' > redis_certificate.crt
```
where `$REMOTE_HOST` is the host hosting the DB and `$REMOTE_PORT` is the port number. For example, `REMOTE_HOST=cz-pottier1-testredis.apps.czapps.llnl.gov` and `REMOTE_PORT=32236`.
If you rely on PDS at LC, you can get the password, the hostname and the port number of your Redis instance by visiting https://launchit.llnl.gov/.

3. Provide the right information to the `miniapp`. The following JSON
   information (password, port and hostname) can be obtained on https://launchit.llnl.gov/.
```json
{
    "database-password": "PASSWORD",
    "service-port": 32245,
    "host": "cz-pottier1-redis1.apps.czapps.llnl.gov",
    "cert": "redis-certificate.crt"
}
```
> You will have to provide the path of the JSON configuration as follows `--dbconfig /usr/workspace/AMS/miniapp_resources/redis/redis-miniapp.json`.
