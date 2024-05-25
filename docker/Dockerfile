# Dockerfile for CAMO.
# Also a Dockerfile for jczhang02's DL/Shanhe Training.
# A100-40GB-SXM

# base_image
ARG PYTHON=python3
ARG PYTHON_VERSION=3.11.9
ARG PYTHON_SHORT_VERSION=3.11
ARG MINIFORGE3_VERSION=23.11.0-0

# Shanhe A100-40GB-SXM is compatiable with cuda-11.7.1
FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04 AS base_image

ENV DEBIAN_FRONTEND=noninteractive \
	LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/lib"

# set mirrors for apt, pip and conda, tuna.tsinghua
COPY ./files/sources.list /etc/apt/sources.list
COPY ./files/pip.conf /root/.pip/pip.conf
COPY ./files/condarc_tsinghua /root/.condarc

# common
FROM base_image AS common
LABEL maintainer="Chengrui Zhang"

ARG PYTHON
ARG PYTHON_VERSION
ARG PYTHON_SHORT_VERSION
ARG MINIFORGE3_VERSION
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
ENV TORCH_CUDA_ARCH_LIST="5.2;7.0+PTX;7.5;8.0;8.6;9.0"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

RUN apt-get update \
	&& apt-get install -y \
	build-essential \
	ca-certificates \
	net-tools \
	cmake \
	curl \
	git \
	jq \
	libssl-dev \
	libtool \
	openssl \
	python3-dev \
	unzip \
	vim \
	wget \
	tzdata \
	&& rm -rf /var/lib/apt/lists/* \
	&& apt-get clean

ENV TZ=Asia/Shanghai
ENV REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
RUN curl -L -o ~/miniforge3.sh https://github.com/conda-forge/miniforge/releases/download/${MINIFORGE3_VERSION}/Miniforge3-${MINIFORGE3_VERSION}-Linux-x86_64.sh \
	&& chmod +x ~/miniforge3.sh \
	&& ~/miniforge3.sh -b -p /opt/conda \
	&& rm ~/miniforge3.sh

RUN  pip install --upgrade pip --no-cache-dir \
	&& ln -s /opt/conda/bin/pip /usr/local/bin/pip3

RUN apt-get update \
	&& apt-get install -y  --allow-downgrades --allow-change-held-packages --no-install-recommends \
	&& apt-get install -y --no-install-recommends openssh-client openssh-server \
	&& mkdir -p /var/run/sshd \
	&& cat /etc/ssh/ssh_config | grep -v StrictHostKeyChecking > /etc/ssh/ssh_config.new \
	&& echo "    StrictHostKeyChecking no" >> /etc/ssh/ssh_config.new \
	&& mv /etc/ssh/ssh_config.new /etc/ssh/ssh_config \
	&& rm -rf /var/lib/apt/lists/* \
	&& apt-get clean

RUN mkdir -p /var/run/sshd && \
	sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd

RUN rm -rf /root/.ssh/ && \
	mkdir -p /root/.ssh/ && \
	ssh-keygen -q -t rsa -N '' -f /root/.ssh/id_rsa && \
	cp /root/.ssh/id_rsa.pub /root/.ssh/authorized_keys \
	&& printf "Host *\n StrictHostKeyChecking no\n" >> /root/.ssh/config

RUN mkdir -p /etc/pki/tls/certs && cp /etc/ssl/certs/ca-certificates.crt /etc/pki/tls/certs/ca-bundle.crt

RUN rm -rf /root/.cache | true

# UNISON
FROM common AS unison
ARG UNISON_VERSION=2.53.3

RUN mkdir -p /tmp/unison \
	&& curl -L https://github.com/bcpierce00/unison/releases/download/v2.53.3/unison-2.53.3+ocaml4.08-ubuntu-x86_64.tar.gz | tar zxv -C /tmp/unison \
	&& cp /tmp/unison/bin/* /usr/local/bin/ \
	&& rm -rf /tmp/unison

RUN echo "root:111111" | chpasswd \
	&& echo "PermitRootLogin yes" >> /etc/ssh/sshd_config

RUN rm -rf /root/.cache | true

# CAMO, used for install python packages
FROM unison AS CAMO

ARG PYTHON
ARG PYTHON_SHORT_VERSION
ARG CUDA_HOME
ARG TORCH_CUDA_ARCH_LIST

WORKDIR /

# Create conda env
RUN /opt/conda/bin/mamba init bash \
	&& /opt/conda/bin/mamba init zsh \
	&& . ~/.bashrc \
	&& mamba create -n CAMO python==${PYTHON_VERSION} --yes \
	&& mamba activate CAMO \
	&& mamba clean -afy

ENV PDM_NO_CACHE=0

RUN git clone https://github.com/jczhang02/CAMO /root/CAMO \
	&& cd /root/CAMO \
	&& . ~/.bashrc \
	&& mamba activate CAMO \
	&& mamba install pdm --yes \
	&& pdm config pypi.url https://pypi.tuna.tsinghua.edu.cn/simple \
	&& pdm self update \
	&& pdm --no-cache install \
	&& pdm cache clear \
	&& rm -rf /root/CAMO

RUN rm -rf /root/.cache | true

# Shanhe
FROM CAMO AS shanhe

# RUN echo "10.251.102.1 mirrors.shanhe.com" >> /etc/hosts
COPY ./files/condarc_shanhe /root/.condarc
COPY ./files/pip_shanhe.conf /root/.pip/pip.conf
COPY ./files/sources_shanhe.list /etc/apt/sources.list

# Entrypoint
CMD ["/usr/sbin/sshd", "-D"]
