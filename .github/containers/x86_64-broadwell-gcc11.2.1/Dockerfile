FROM centos:7 AS base
MAINTAINER Giorgis Georgakoudis <georgakoudis1@llnl.gov>
RUN \
    yum install -y dnf &&\
    dnf install -y epel-release &&\
    dnf group install -y "Development Tools" &&\
    dnf install -y curl findutils gcc-gfortran gnupg2 hostname iproute redhat-lsb-core python3 python3-pip python3-setuptools unzip python-boto3 centos-release-scl-rh &&\
    dnf install -y devtoolset-11 environment-modules &&\
    dnf upgrade -y
COPY repo repo
RUN \
    mkdir -p ams-spack-env
COPY spack.yaml ams-spack-env/spack.yaml


FROM base AS setup-spack-env
RUN \
    source /etc/profile &&\
    mkdir -p /usr/share/Modules/modulefiles/gcc &&\
    /usr/share/Modules/bin/createmodule.sh /opt/rh/devtoolset-11/enable > /usr/share/Modules/modulefiles/gcc/11.2.1 &&\
    module load gcc/11.2.1 &&\
    git clone --depth 1 --branch releases/v0.20 https://github.com/spack/spack.git &&\
    source spack/share/spack/setup-env.sh &&\
    spack compiler find &&\
    spack compiler rm gcc@4.8.5 &&\
    sed -i "s/modules.*/modules: [gcc\/11.2.1]/"  ~/.spack/linux/compilers.yaml

FROM setup-spack-env AS install-spack-env
RUN \
    source spack/share/spack/setup-env.sh &&\
    spack env activate -p ams-spack-env &&\
    spack install &&\
    spack clean --all

