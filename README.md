# marbl material proporties mini-miniapp


## Setup and Build using shared `spack`

All dependencies needed for the `miniapp` are installed through `spack` on a shared
space `/usr/workspace/AMS`.

Steps to build the code.

1. Clone the code repository.
```
$ git clone ssh://git@czgitlab.llnl.gov:7999/autonomous-multiscale-project/marbl-matprops-miniapp.git
$ cd marbl-matprops-miniapp
```

2. Setup the environment -- the following script will load all dependencies and
setup some relevant paths.
```
$ source setup/setup_env.sh
```

3. Build the mini-app. Please note the optional features that can be turned off.
```
$ mkdir build; cd build
$ cmake \
-DWITH_REDIS=On \
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

### Issue with MPI on specturm-mpi systems (lassen)

Enabling MPI and Torch at build time conflicts with the current
installation of LSF. Torch depends on nccl which in turn depends
on rdma. We have not add an external in spack for rdma as it seems it is
not installed somewhere idependently. Thus py-torch installs their own
rdma version which conflicts with the current one required from LSF. We are
working for a fix.

### Run Mini-app

To run the mini-app use any of these options.
```
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
   -v, --verbose, -qu, --quiet, current option: --quiet
  Print extra stuff
```

```
$ ./build/examples/ams_example -S /usr/workspace/AMS/miniapp_resources/trained_models/debug_model.pt
```
  **TODO:** add instructions on command line options!

## System Setup Using Spack (Deprecated)

** These are the instructions to create a new spack environment. not needed if
you can simply use the shared installations from above.**

To install all dependencies of the proxy-app we rely on spack@0.18.0 (Earlier versions should work as
well) and on spack environments. We provide instruction on how to install on an IBM-NVIDIA V100 system.
We assume that the system already has gcc@8.3.1 and cuda@10.11.1.

Change directory to the root of the repo. Then:
1. Create a new spack environment.
```bash
mkdir spack_env
spack env create -d spack_env
```
2. Open and edit the spack environement configuration file (spack_env/spack.yaml). In the end the configuration file should look like the following:

```bash
# This is a Spack Environment file.
#
# It describes a set of packages to be installed, along with
# configuration settings.
spack:
  # add package specs to the `specs` list
  specs: []
  concretization: together
  config:
    url_fetch_method: curl
    install_tree:
      root: $PATH_TO_ROOT_DIR/spack_env/
    packages:
      cuda:
        externals:
        - spec: cuda@11.1.0 arch=linux-rhel7-power9le
          prefix: PREFIX_TO_CUDA@11.1.0
        - spec: cuda@11.2.0 arch=linux-rhel7-power9le
          prefix: PREFIX_TO_CUDA@11.2.0
        - spec: cuda@11.3.0 arch=linux-rhel7-power9le
          prefix: PREFIX_TO_CUDA@11.3.0
        - spec: cuda@11.4.1 arch=linux-rhel7-power9le
          prefix: PREFIX_TO_CUDA@11.4.1
        - spec: cuda@11.5.0 arch=linux-rhel7-power9le
          prefix: PREFIX_TO_CUDA@11.5.0
        buildable: false
    spectrum-mpi:
      externals:
      - spec: spectrum-mpi@10.3.1 arch=linux-rhel7-power9le
        prefix: ${PATH_TO_SPECTRUM_MPI}
      buildable: false
```

The following configuration instructs spack to store all binaries under the directory "${PATH_TO_ROOT_DIR}/spack_env". Any full path will do. You need to remember this path though. Next we inform spack of our cuda and MPI installations. We create two external packages and fill the "prefix" field to point to the respective installation directory. Finally, for both packages we enforce spack to use the specific installation by setting the field "buildable" to false.

3.  Activate the environment.

```bash
spack env activate spack_env
```

4. Install python:
```bash
spack add python@3.8.6%gcc@8.3.1
spack concretize
spack install
```

5. Load new python through spack and install some python packages.

```bash
spack load python@3.8.6
python -m ensurepip
python -m pip install cmake ninja
```

6. Add next dependencies to the spack environment.
```bash
spack add mfem%gcc@8.3.1+cuda~mpi+shared cuda_arch=70 ^cuda@11.1.0 ^spectrum-mpi
spack add caliper@2.7.0%gcc@8.3.1~adiak+cuda cuda_arch=70 ^cuda@11.1.0 ^spectrum-mpi
spack add python@3.8.6%gcc@8.3.1
spack add py-torch@1.8.1%gcc@8.3.1+cuda~distributed~mkldnn~xnnpack cuda_arch=70
spack concretize
spack install
```

The previous commands instruct spack to install mfem, py-torch and caliper. It enforces some constraints into the spack solver. Namely, we require to use spectrum-mpi and gcc@8.3.1. The "concretize" command solves the constrains and the install command will install all these dependencies in your system.

## Build Proxy Application using the Spack environment
1. Activate the installed environment and load all required packages:
```bash
spack env activate spack_env
spack load mfem
spack load caliper
spack load python
spack load py-torch
```

2. Build the Proxy Application
```bash
mkdir build/
cd build
cmake -DWITH_EXAMPLES=On -DTorch_DIR=$(spack location -i py-torch)/lib/python3.8/site-packages/torch/share/cmake/Torch -DWITH_CALIPER=On -DWITH_CUDA=On -DWITH_TORCH=On  -DMFEM_DIR=$(spack location -i mfem) ../
make
```

## Build Proxy Application Container on LC Catalyst
Container building on LC clusters requires using podman. Podman supports container builds using a Dockerfile, but needs some specific changes to the build environment to build the miniapp successfully. Currently the podman build works on Catalyst, although LC hopes to enable it on clusters like Quartz as well.
1. Set up the build environment based on [LC instructions](https://lc.llnl.gov/cloud/services/containers/Building_containers/#3-building-from-a-dockerfile-with-podman):
Get an allocation on a compute node. The build will take > 2 hours:
```bash
salloc -N 1 -t 240 --userns
```
2. Verify that /var/tmp/<your LC username> is empty
3. Verify that the output of `ps aux | grep podman` is empty (besides the grep command)
4. Configure overlay FS and set ulimit:
```bash
. /collab/usr/gapps/lcweg/containers/scripts/enable-podman.sh overlay
sed -i "s/var/overlay/" ~/.config/containers/storage.conf
ulimit -Ss 8192
```
5. Build the image with podman, tagging it with the GitLab CZ registry to enable subsequent image push:
```bash
cd marbl-matprops-miniapp/
podman build -t czregistry.llnl.gov:5050/autonomous-multiscale-project/marbl-matprops-miniapp:<tagname> -f docker/Dockerfile .
```
6. Create a personal access token in the CZ GitLab with "api", "read_registry," and "write_registry" scopes.
7. Log in to the GitLab CZ registry with you LC username and the personal access toaken generated in step 6 as the password. Then push the image to the registry:
```bash
podman login czregistry.llnl.gov:5050
podman push czregistry.llnl.gov:5050/autonomous-multiscale-project/marbl-matprops-miniapp:<tagname>
```


## running
To run the proxy application please issue the following command inside the build directory:
```bash
./build/examples/ams_example -S /usr/workspace/AMS/miniapp_resources/trained_models/debug_model.pt
```

By default the evals will be on the cpu, if you want to run on the gpu add the option: ` -d cuda`

## questions

- The indicators are constant but they change in a real sim with ale, should they change here?
- The initial eos inputs are random and unchanging, should they be real data and/or change
  each "cycle"?
