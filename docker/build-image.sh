#!/bin/bash
BASE_IMAGE_TAG=noetic
IMAGE_NAME=air_hockey

ROS_IP="128.178.96.208" #$(hostname --ip-address | awk '{print $1}')

# Setup ROS MASTER communication here -> Use ROS_IP for single PC setup
ROS_MASTER_IP="128.178.145.165"

HELP_MESSAGE="Usage: ./build-server.sh [-b|--branch branch] [-r] [-v] [-s]
Build a Docker container for remote development and/or running unittests.
Options:
  --base-tag               The tag of ros2-control-libraries image.

  -t, --target             Rebuild the image from targeted stage.

  -r, --rebuild            Rebuild the image with no cache.

  -v, --verbose            Show all the output of the Docker
                           build process
                           
  -s, --serve              Start the remote development server.

  -h, --help               Show this help message.
"

BUILD_FLAGS=()
while [ "$#" -gt 0 ]; do
  case "$1" in
    --base-tag) BASE_IMAGE_TAG=$2; shift 2;;
    -t|--target) BUILD_FLAGS+=(--target $2); shift 2;;
    -r|--rebuild) BUILD_FLAGS+=(--no-cache); shift 1;;
    -v|--verbose) BUILD_FLAGS+=(--progress=plain); shift 1;;
    -h|--help) echo "${HELP_MESSAGE}"; exit 0;;
    *) echo "Unknown option: $1" >&2; echo "${HELP_MESSAGE}"; exit 1;;
  esac
done

# Add arguments to docker build command
BUILD_FLAGS+=(--build-arg BASE_IMAGE_TAG="${BASE_IMAGE_TAG}")
BUILD_FLAGS+=(--build-arg ROS_IP="${ROS_IP}")
BUILD_FLAGS+=(--build-arg ROS_MASTER_IP="${ROS_MASTER_IP}")
BUILD_FLAGS+=(-t "${IMAGE_NAME}:${BASE_IMAGE_TAG}")
BUILD_FLAGS+=(--build-arg HOST_GID=$(id -g))   # Pass the correct GID to avoid issues with mounted volumes
BUILD_FLAGS+=(--ssh default="${SSH_AUTH_SOCK}") # Pass git ssh key to be able to pull

DOCKER_BUILDKIT=1 docker build "${BUILD_FLAGS[@]}" -f ./docker/Dockerfile .


