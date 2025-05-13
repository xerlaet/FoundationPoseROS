#!/bin/bash

source $(dirname $0)/config.sh

# Generate a .env file for Docker build arguments
log_message "Generating .env file for Docker build arguments..."

cat <<EOF >.env
DOCKER_USER_ARG=${USER_NAME}
DOCKER_USER_UID_ARG=${USER_ID}
DOCKER_USER_GID_ARG=${GROUP_ID}
TORCH_CUDA_ARCH_LIST_ARG=${TORCH_CUDA_ARCH_LIST}
ROS1_APT_PACKAGE_ARG=${ROS1_APT_PACKAGE}
ROS2_APT_PACKAGE_ARG=${ROS2_APT_PACKAGE}
DISPLAY=${DISPLAY:-:1}
TERM=$TERM
PROJ_ROOT=$PROJ_ROOT
ISAAC_LAB_VERSION_ARG=${ISAAC_LAB_VERSION}
ISAAC_SIM_VERSION_ARG=${ISAAC_SIM_VERSION}
ISAAC_CACHE_OV=${DOCKER_DIR}/volumes/isaac-sim/isaac-cache-ov
ISAAC_CACHE_PIP=${DOCKER_DIR}/volumes/isaac-sim/isaac-cache-pip
ISAAC_CACHE_GL=${DOCKER_DIR}/volumes/isaac-sim/isaac-cache-gl
ISAAC_CACHE_COMPUTE=${DOCKER_DIR}/volumes/isaac-sim/isaac-cache-compute
ISAAC_LOGS=${DOCKER_DIR}/volumes/isaac-sim/isaac-logs
ISAAC_DATA=${DOCKER_DIR}/volumes/isaac-sim/isaac-data
ISAAC_DOCS=${DOCKER_DIR}/volumes/isaac-sim/isaac-docs
EOF

log_message "Done!!!"
