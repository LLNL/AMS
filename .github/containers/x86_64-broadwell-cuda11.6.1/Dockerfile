FROM nvidia/cuda:11.6.1-devel-ubi8 AS base
MAINTAINER Giorgis Georgakoudis <georgakoudis1@llnl.gov>
RUN \
    yum install -y dnf &&\
    dnf install -y git xz autoconf automake unzip patch gcc-gfortran bzip2 file &&\
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
    spack external find --all --not-buildable --exclude openssl --exclude openblas --exclude bzip2

FROM setup-spack-env AS install-spack-env
RUN \
    source spack/share/spack/setup-env.sh &&\
    spack env activate -p ams-spack-env &&\
    spack install --fail-fast

FROM install-spack-env AS clean-spack
RUN \
    spack clean --all
