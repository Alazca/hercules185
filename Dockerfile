# parameters
ARG REPO_NAME="hercules185"
ARG DESCRIPTION="185 Group Project"
ARG MAINTAINER="Hercules club"
# pick an icon from: https://fontawesome.com/v4.7.0/icons/
ARG ICON="cube"

# ==================================================>
# ==> Do not change the code below this line
ARG ARCH
ARG DISTRO=daffy
ARG DOCKER_REGISTRY=docker.io
ARG BASE_IMAGE=dt-ros-commons
ARG BASE_TAG=daffy
ARG LAUNCHER=default

# define base image
FROM ${DOCKER_REGISTRY}/duckietown/${BASE_IMAGE}:${BASE_TAG} AS base

# recall all arguments
ARG DISTRO
ARG REPO_NAME
ARG DESCRIPTION
ARG MAINTAINER
ARG ICON
ARG BASE_TAG
ARG BASE_IMAGE
ARG LAUNCHER
# - buildkit
ARG TARGETPLATFORM
ARG TARGETOS
ARG TARGETARCH
ARG TARGETVARIANT

# check build arguments
RUN dt-build-env-check "${REPO_NAME}" "${MAINTAINER}" "${DESCRIPTION}"

# Install CUDA for Jetson (L4T)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-dev \
    libopenblas-base \
    libopenmpi-dev \
    && rm -rf /var/lib/apt/lists/*

# Set CUDA environment variables for Jetson
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

# nvidia-container-runtime
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/user-guide.html
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES all

# VERSIONING CONFIGURATION
ENV CUDA_VERSION 10.2.89
ENV CUDA_PKG_VERSION 10-2=$CUDA_VERSION-1
ENV NCCL_VERSION 2.8.4
ENV CUDNN_VERSION 8.1.1.33

ENV PYTORCH_VERSION 1.7.0
ENV TORCHVISION_VERSION 0.8.1

#ENV TENSORRT_VERSION 7.1.3.4

ENV PYCUDA_VERSION 2021.1

# define/create repository path
ARG REPO_PATH="${CATKIN_WS_DIR}/src/${REPO_NAME}"
ARG LAUNCH_PATH="${LAUNCH_DIR}/${REPO_NAME}"
RUN mkdir -p "${REPO_PATH}" "${LAUNCH_PATH}"
WORKDIR "${REPO_PATH}"

# keep some arguments as environment variables
ENV DT_MODULE_TYPE="${REPO_NAME}" \
    DT_MODULE_DESCRIPTION="${DESCRIPTION}" \
    DT_MODULE_ICON="${ICON}" \
    DT_MAINTAINER="${MAINTAINER}" \
    DT_REPO_PATH="${REPO_PATH}" \
    DT_LAUNCH_PATH="${LAUNCH_PATH}" \
    DT_LAUNCHER="${LAUNCHER}"

# install apt dependencies
COPY ./dependencies-apt.txt "${REPO_PATH}/"
RUN dt-apt-install ${REPO_PATH}/dependencies-apt.txt

# install python3 dependencies
ARG PIP_INDEX_URL="https://pypi.org/simple"
ENV PIP_INDEX_URL=${PIP_INDEX_URL}
COPY ./dependencies-py3.* "${REPO_PATH}/"
RUN dt-pip3-install "${REPO_PATH}/dependencies-py3.*"

# copy the source code
COPY ./packages "${REPO_PATH}/packages"

# build packages
RUN . /opt/ros/${ROS_DISTRO}/setup.sh && \
  catkin build \
    --workspace ${CATKIN_WS_DIR}/

# Add sourcing commands to .bashrc
RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> ~/.bashrc && \
    echo "source ${CATKIN_WS_DIR}/devel/setup.bash" >> ~/.bashrc

# install launcher scripts
COPY ./launchers/. "${LAUNCH_PATH}/"
RUN dt-install-launchers "${LAUNCH_PATH}"

# define default command
CMD ["bash", "-c", "dt-launcher-${DT_LAUNCHER}"]

# store module metadata
LABEL org.duckietown.label.module.type="${REPO_NAME}" \
    org.duckietown.label.module.description="${DESCRIPTION}" \
    org.duckietown.label.module.icon="${ICON}" \
    org.duckietown.label.platform.os="${TARGETOS}" \
    org.duckietown.label.platform.architecture="${TARGETARCH}" \
    org.duckietown.label.platform.variant="${TARGETVARIANT}" \
    org.duckietown.label.code.location="${REPO_PATH}" \
    org.duckietown.label.code.version.distro="${DISTRO}" \
    org.duckietown.label.base.image="${BASE_IMAGE}" \
    org.duckietown.label.base.tag="${BASE_TAG}" \
    org.duckietown.label.maintainer="${MAINTAINER}"
# <== Do not change the code above this line
# <==================================================
