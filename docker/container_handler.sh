#!/bin/bash

source $(dirname $0)/config.sh

# Check the .env file first
if [ ! -f ${DOCKER_DIR}/.env ]; then
    log_message "No .env file found. Please create one."
    exit 1
fi

# Check arguments
if [ "$#" -lt 1 ]; then
    log_message "Usage: $0 <case_name> [<profile_name>]"
    log_message "case_name: build, run, enter, stop, remove, default is build"
    log_message "profile_name: base, ros2, mano-isaac, default is base"
fi

CASE_NAME=${1:-build}
PROFILE_NAME=${2:-base}

# Check if the case name is valid
if [[ ! "$CASE_NAME" =~ ^(build|run|enter|stop|remove)$ ]]; then
    log_message "Invalid case name: $CASE_NAME. Valid cases are: build, run, enter, stop, remove."
    exit 1
fi
# Check if the profile name is valid
if [[ ! "$PROFILE_NAME" =~ ^(ros1|ros2)$ ]]; then
    log_message "Invalid profile name: $PROFILE_NAME. Valid profiles are: base, ros2, mano-isaac."
    exit 1
fi

# Get the container name from the profile name
CONTAINER_NAME="posekit-$PROFILE_NAME"

# Check if the container is running
if [ "$(docker ps -q -f name=$CONTAINER_NAME)" ]; then
    CONTAINER_STATUS="running"
else
    CONTAINER_STATUS="stopped"
fi

# Function to build the container
build_container() {
    # if the container is already running, stop it
    if [ "$CONTAINER_STATUS" == "running" ]; then
        log_message "Stopping the container $CONTAINER_NAME..."
        docker compose --env-file ${DOCKER_DIR}/.env --file ${DOCKER_DIR}/docker-compose.yaml --profile $PROFILE_NAME down
    fi
    # Build the container
    log_message "Building the container $CONTAINER_NAME..."
    docker compose --env-file ${DOCKER_DIR}/.env --file ${DOCKER_DIR}/docker-compose.yaml --profile $PROFILE_NAME build
}

# Function to run the container
run_container() {
    # if the container is already running, stop it
    if [ "$CONTAINER_STATUS" == "running" ]; then
        log_message "Container $CONTAINER_NAME is already running. Please stop it first."
        exit 1
    fi
    # Run the container
    log_message "Running the container $CONTAINER_NAME..."
    docker compose --env-file ${DOCKER_DIR}/.env --file ${DOCKER_DIR}/docker-compose.yaml --profile $PROFILE_NAME up -d
}

# Function to enter the container
enter_container() {
    # Check if the container is running
    if [ "$CONTAINER_STATUS" == "running" ]; then
        log_message "Entering the container $CONTAINER_NAME..."
        docker exec -it $CONTAINER_NAME zsh --login
    else
        log_message "Container $CONTAINER_NAME is not running. Please start it first."
        exit 1
    fi
}

# Function to stop the container
stop_container() {
    # Check if the container is running
    if [ "$CONTAINER_STATUS" == "running" ]; then
        log_message "Stopping the container $CONTAINER_NAME..."
        docker compose --env-file ${DOCKER_DIR}/.env --file ${DOCKER_DIR}/docker-compose.yaml --profile $PROFILE_NAME down
    else
        log_message "Container $CONTAINER_NAME is not running."
    fi
}

# Function to remove the container
remove_container() {
    # Check if the container is running
    if [ "$CONTAINER_STATUS" == "running" ]; then
        log_message "Stopping the container $CONTAINER_NAME..."
        docker compose --env-file ${DOCKER_DIR}/.env --file ${DOCKER_DIR}/docker-compose.yaml --profile $PROFILE_NAME down
    fi
    # Remove the container
    log_message "Removing the container $CONTAINER_NAME..."
    docker compose --env-file ${DOCKER_DIR}/.env --file ${DOCKER_DIR}/docker-compose.yaml --profile $PROFILE_NAME rm -f
}

# Execute the case based on the first argument
case $CASE_NAME in
    build)
        build_container
        ;;
    run)
        run_container
        ;;
    enter)
        enter_container
        ;;
    stop)
        stop_container
        ;;
    remove)
        remove_container
        ;;
esac
