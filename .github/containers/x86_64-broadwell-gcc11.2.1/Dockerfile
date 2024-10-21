FROM almalinux:8 AS base
MAINTAINER Giorgis Georgakoudis <georgakoudis1@llnl.gov>
RUN \
    yum install -y dnf &&\
    dnf group install -y "Development Tools" &&\
    dnf install -y git gcc-toolset-11 environment-modules libevent-devel openssl-devel &&\
    dnf upgrade -y
COPY repo repo
RUN \
    mkdir -p ams-spack-env
COPY spack.yaml ams-spack-env/spack.yaml

FROM base AS setup-spack-env
RUN \
    source /etc/profile &&\
    mkdir -p /usr/share/Modules/modulefiles/gcc &&\
    /usr/share/Modules/bin/createmodule.sh /opt/rh/gcc-toolset-11/enable > /usr/share/Modules/modulefiles/gcc/11.2.1 &&\
    module load gcc/11.2.1 &&\
    git clone --depth 1 --branch releases/v0.20 https://github.com/spack/spack.git &&\
    source spack/share/spack/setup-env.sh &&\
    spack compiler find &&\
    spack compiler rm gcc@8.5.0 &&\
    sed -i "s/modules.*/modules: [gcc\/11.2.1]/"  ~/.spack/linux/compilers.yaml

FROM setup-spack-env AS install-spack-env
RUN \
    source /etc/profile &&\
    module load gcc/11.2.1 &&\
    source spack/share/spack/setup-env.sh &&\
    spack env activate -p ams-spack-env &&\
    spack install --fail-fast &&\
    spack clean --all &&\
    dnf clean all

