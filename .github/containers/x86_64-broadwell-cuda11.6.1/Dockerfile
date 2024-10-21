FROM nvidia/cuda:11.6.1-devel-ubi8 AS base
MAINTAINER Giorgis Georgakoudis <georgakoudis1@llnl.gov>
RUN \
    yum install -y dnf &&\
    dnf install -y git xz autoconf automake unzip patch gcc-gfortran bzip2 file libevent-devel openssl-devel &&\
    dnf upgrade -y &&\
    dnf clean all
COPY repo repo
RUN \
    mkdir -p ams-spack-env
COPY spack.yaml ams-spack-env/spack.yaml

FROM base AS setup-spack-env
RUN \
    git clone --depth 1 --branch releases/v0.20 https://github.com/spack/spack.git &&\
    source spack/share/spack/setup-env.sh &&\
    spack env activate -p ams-spack-env &&\
    spack external find --all --not-buildable --exclude openssl --exclude openblas --exclude bzip2 &&\
    spack install --fail-fast &&\
    spack clean --all
