FROM docker.io/nvidia/cuda:11.4.3-cudnn8-devel-ubuntu20.04 as builder

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

ARG spack_cpu_arch=power9le
ARG build_jobs=16
ARG cuda_arch=70

RUN touch /etc/apt/apt.conf.d/99verify-peer.conf \
    && echo >>/etc/apt/apt.conf.d/99verify-peer.conf "Acquire { https::Verify-Peer false }"

RUN apt-get update
RUN DEBIAN_FRONTEND="noninteractive" apt-get -y install \
    build-essential \
    gfortran \
    libnuma-dev \
    autotools-dev \
    autoconf \
    automake \
    cmake \
    git \
    python3 \
    python3-dev \
    python3-distutils \
    python3-pip \
    python3-venv \
    file \
    curl \
    unzip \
    valgrind \
    openssh-server \
    openssh-client \
    apt-utils \
    && apt-get clean \
    && apt-get autoremove \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# What we want to install and how we want to install it
# is specified in a manifest file (spack.yaml)
RUN mkdir /opt/spack-environment \
    &&  (echo "spack:" \
    &&   echo "  specs:" \
    &&   echo "  - python@3.8.6" \
    &&   echo "  - openmpi@4.1.3 +cuda ^cuda@11.4.152" \
    &&   echo "  - mfem@4.4.0 +umpire+cuda~mpi+shared cuda_arch=${cuda_arch} ^cuda@11.4.152 ^openmpi@4.1.3" \
    &&   echo "  - caliper@2.7.0 ~adiak+cuda+mpi cuda_arch=${cuda_arch} ^cuda@11.4.152 ^openmpi@4.1.3" \
    &&   echo "  - umpire@6.0.0 +cuda cuda_arch=${cuda_arch} ^cuda@11.4.152" \
    &&   echo "  - faiss@1.7.2 +python +cuda cuda_arch=${cuda_arch} ^python@3.8.6 ^cuda@11.4.152" \
    &&   echo "  concretizer:" \
    &&   echo "    unify: true" \
    &&   echo "  config:" \
    &&   echo "    install_tree: /opt/software" \
    &&   echo "    url_fetch_method: curl" \
    &&   echo "    packages:" \
    &&   echo "      nccl:" \
    &&   echo "        externals:" \
    &&   echo "        - spec: nccl@2.11.4 arch=linux-ubuntu20.04" \
    &&   echo "          prefix: /usr/lib/x86_64-linux-gnu/" \
    &&   echo "      buildable: False" \
    &&   echo "      cudnn:" \
    &&   echo "        - spec: cudnn@8.2.4 arch=linux-ubuntu20.04" \
    &&   echo "          prefix: /usr/lib/x86_64-linux-gnu/" \
    &&   echo "      buildable: False" \
    &&   echo "  view: /opt/view") > /opt/spack-environment/spack.yaml

# Install the software, remove unnecessary deps
RUN cd /opt/spack-environment \
    && git clone --single-branch --branch v0.18.1 https://github.com/spack/spack.git

# Need patched Spack for Faiss 1.7.2
COPY docker/package.py /opt/spack-environment/spack/var/spack/repos/builtin/packages/faiss/
COPY docker/fixes-in-v1.7.2.patch /opt/spack-environment/spack/var/spack/repos/builtin/packages/faiss/

RUN cd /opt/spack-environment \
    && . spack/share/spack/setup-env.sh \
    && spack env activate . \
    && spack external find openssh cuda cudnn nccl valgrind \
    && spack install --reuse --fail-fast

RUN cd /opt/spack-environment \
    && . spack/share/spack/setup-env.sh \
    && spack env activate . \
    && spack load python@3.8.6 \
    && python -m ensurepip \
    && python -m pip install cmake ninja 

# Get the Torch lib with --no-check-certificate due to LLNL cerfitificate man-in-the-middle
RUN cd /opt/view \
    && wget --no-check-certificate https://download.pytorch.org/libtorch/cu113/libtorch-cxx11-abi-shared-with-deps-1.12.1%2Bcu113.zip \
    && unzip libtorch-cxx11-abi-shared-with-deps-1.12.1+cu113.zip \
    && rm libtorch-cxx11-abi-shared-with-deps-1.12.1+cu113.zip

# Modifications to the environment that are necessary to run
RUN cd /opt/spack-environment \
    && . spack/share/spack/setup-env.sh \
    && spack env activate --sh -d . >> /etc/profile.d/z10_spack_environment.sh

RUN cd /root/
WORKDIR /root/
RUN mkdir -p /root/marbl-matprops-miniapp/build && ls -al /root/marbl-matprops-miniapp/
COPY . /root/marbl-matprops-miniapp/
RUN cd /root/marbl-matprops-miniapp/build
WORKDIR /root/marbl-matprops-miniapp/build
RUN . /etc/profile \
    && cmake -DTorch_DIR=/opt/view/libtorch/share/cmake/Torch -DWITH_TORCH=On -DWITH_CALIPER=On -DWITH_CUDA=On -DMFEM_DIR=$(spack location -i mfem) -DUMPIRE_DIR=$(spack location -i umpire) -DWITH_FAISS=On -DFAISS_DIR=$(spack location -i faiss) ../ \
    && make -j ${build_jobs}

# Strip all the binaries
RUN find -L /opt/view/* -type f -exec readlink -f '{}' \; | \
    xargs file -i | \
    grep 'charset=binary' | \
    grep 'x-executable\|x-archive\|x-sharedlib' | \
    awk -F: '{print $1}' | xargs strip -s

# Now we build the miniapp runtime image
FROM nvidia/cuda:11.4.3-cudnn8-runtime-ubuntu20.04

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

COPY --from=builder /opt/spack-environment /opt/spack-environment
COPY --from=builder /opt/software /opt/software
COPY --from=builder /opt/view /opt/view
COPY --from=builder /etc/profile.d/z10_spack_environment.sh /etc/profile.d/z10_spack_environment.sh

RUN touch /etc/apt/apt.conf.d/99verify-peer.conf \
    && echo >>/etc/apt/apt.conf.d/99verify-peer.conf "Acquire { https::Verify-Peer false }"

RUN apt-get update
RUN DEBIAN_FRONTEND="noninteractive" apt-get -y install --no-install-recommends \
    libnuma-dev \
    libgomp1 \
    && apt-get clean \
    && apt-get autoremove \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN cd /root/
RUN mkdir /root/marbl-matprops-miniapp
COPY --from=builder /root/marbl-matprops-miniapp/ /root/marbl-matprops-miniapp/
RUN cd /root/marbl-matprops-miniapp/
WORKDIR /root/marbl-matprops-miniapp/

