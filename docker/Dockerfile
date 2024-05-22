# base_image
ARG PYTHON=python3
ARG PYTHON_VERSION=3.11.9
ARG PYTHON_SHORT_VERSION=3.11
ARG MINIFORGE3_VERSION=23.11.0-0
ARG PYTORCH_VERSION=2.3.0

FROM nvidia/cuda:12.1.0-base-ubuntu20.04 AS base_image

ENV DEBIAN_FRONTEND=noninteractive \
	LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/lib"

RUN printf "\
	deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal main restricted universe multiverse\n\
	# deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal main restricted universe multiverse\n\
	deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal-updates main restricted universe multiverse\n\
	# deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal-updates main restricted universe multiverse\n\
	deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal-backports main restricted universe multiverse\n\
	# deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal-backports main restricted universe multiverse\n\
	deb http://security.ubuntu.com/ubuntu/ focal-security main restricted universe multiverse\n\
	# deb-src http://security.ubuntu.com/ubuntu/ focal-security main restricted universe multiverse\n\
	" > /etc/apt/sources.list \
	&& apt-get update \
	&& apt-get upgrade -y \
	&& apt-get autoremove -y \
	&& apt-get clean \
	&& rm -rf /var/lib/apt/lists/*

# common
FROM base_image AS common

LABEL maintainer="Chengrui Zhang"

ARG PYTHON
ARG PYTHON_VERSION
ARG PYTHON_SHORT_VERSION
ARG MINIFORGE3_VERSION

# This arg required to stop docker build waiting for region configuration while installing tz data from ubuntu 20
ARG DEBIAN_FRONTEND=noninteractive

ENV CUDA_HOME=/opt/conda
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV LD_LIBRARY_PATH="/usr/local/lib:${LD_LIBRARY_PATH}"
ENV LD_LIBRARY_PATH="/opt/conda/lib:${LD_LIBRARY_PATH}"
ENV PYTHONIOENCODING=UTF-8
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PATH="/opt/conda/bin:${PATH}"
# older archs are not supported on CUDA12.1 like 3.7 for P2 instance
# 5.2 is G3 EC2 instance, 7.5 is G4*, 7.0 is p3*, 8.0 is P4*, 8.6 is G5* and 9.0 is P5*
ENV TORCH_CUDA_ARCH_LIST="5.2;7.0+PTX;7.5;8.0;8.6;9.0"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CUDNN_VERSION=8.9.2.26
ENV NCCL_VERSION=2.21.5
ENV EFA_VERSION=1.32.0
ENV GDRCOPY_VERSION=2.4.1

ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"
ENV OPEN_MPI_PATH=/opt/openmpi

ENV DGLBACKEND=pytorch
ENV MANUAL_BUILD=0
ENV DLC_CONTAINER_TYPE=training

RUN apt-get update \
	&& apt-get -y upgrade --only-upgrade systemd \
	&& apt-get install -y --allow-change-held-packages --no-install-recommends \
	build-essential \
	ca-certificates \
	cmake \
	libcudnn8=$CUDNN_VERSION-1+cuda12.1 \
	libcudnn8-dev=$CUDNN_VERSION-1+cuda12.1 \
	curl \
	git \
	jq \
	libcurl4-openssl-dev \
	libglib2.0-0 \
	libgl1-mesa-glx \
	libsm6 \
	libxext6 \
	libxrender-dev \
	libgomp1 \
	libibverbs-dev \
	libnuma1 \
	libnuma-dev \
	libssl1.1 \
	libssl-dev \
	libtool \
	openssl \
	python3-dev \
	unzip \
	vim \
	wget \
	zlib1g-dev \
	autoconf \
	pkg-config \
	check \
	libsubunit0 \
	libsubunit-dev \
	# ninja is needed for apex to speed up the build
	ninja-build \
	&& rm -rf /var/lib/apt/lists/* \
	&& apt-get clean

# for conda ssl verification
ENV REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
RUN curl -L -o ~/miniforge3.sh https://github.com/conda-forge/miniforge/releases/download/${MINIFORGE3_VERSION}/Miniforge3-${MINIFORGE3_VERSION}-Linux-x86_64.sh \
	&& chmod +x ~/miniforge3.sh \
	&& ~/miniforge3.sh -b -p /opt/conda \
	&& rm ~/miniforge3.sh

RUN pip config set global.index-url http://mirrors.aliyun.com/pypi/simple \
	&& pip install --upgrade pip --no-cache-dir --trusted-host pypi.org --trusted-host files.pythonhosted.org \
	&& ln -s /opt/conda/bin/pip /usr/local/bin/pip3

# Create conda env
RUN /opt/conda/bin/mamba init bash \
	&& . ~/.bashrc \
	&& mamba create -n main python==${PYTHON_VERSION} --yes \
	&& mamba activate main

RUN git clone https://github.com/jczhang02/CAMO /root/CAMO \
	&& cd /root/CAMO \
	&& . ~/.bashrc \
	&& mamba activate main \
	&& mamba install pdm --yes \
	&& pdm self update \
	&& pdm install

RUN apt-get update \
	&& apt-get install -y  --allow-downgrades --allow-change-held-packages --no-install-recommends \
	&& apt-get install -y --no-install-recommends openssh-client openssh-server \
	&& mkdir -p /var/run/sshd \
	&& cat /etc/ssh/ssh_config | grep -v StrictHostKeyChecking > /etc/ssh/ssh_config.new \
	&& echo "    StrictHostKeyChecking no" >> /etc/ssh/ssh_config.new \
	&& mv /etc/ssh/ssh_config.new /etc/ssh/ssh_config \
	&& rm -rf /var/lib/apt/lists/* \
	&& apt-get clean



# Configure OpenSSH so that nodes can communicate with each other
RUN mkdir -p /var/run/sshd && \
	sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd

RUN rm -rf /root/.ssh/ && \
	mkdir -p /root/.ssh/ && \
	ssh-keygen -q -t rsa -N '' -f /root/.ssh/id_rsa && \
	cp /root/.ssh/id_rsa.pub /root/.ssh/authorized_keys \
	&& printf "Host *\n StrictHostKeyChecking no\n" >> /root/.ssh/config

RUN mkdir -p /etc/pki/tls/certs && cp /etc/ssl/certs/ca-certificates.crt /etc/pki/tls/certs/ca-bundle.crt

# Removing the cache as it is needed for security verification
RUN rm -rf /root/.cache | true








# UNISON

# Neovim

# Zsh

# Slurm
