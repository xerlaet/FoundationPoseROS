#!/bin/bash

# Load shared config
source "$(dirname "$0")/config.sh"

# Set flags for optional downloads
DOWNLOAD_SAM2=false
DOWNLOAD_XMEM=false
DOWNLOAD_MEDIAPIPE=false
DOWNLOAD_FOUNDATIONPOSE=false

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --sam2) DOWNLOAD_SAM2=true ;;
        --xmem) DOWNLOAD_XMEM=true ;;
        --mediapipe) DOWNLOAD_MEDIAPIPE=true ;;
        --foundationpose) DOWNLOAD_FOUNDATIONPOSE=true ;;
        --all) DOWNLOAD_SAM2=true; DOWNLOAD_XMEM=true; DOWNLOAD_MEDIAPIPE=true; DOWNLOAD_FOUNDATIONPOSE=true ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Function to download SAM2 checkpoints
download_sam2_checkpoints() {
    BASE_URL="https://dl.fbaipublicfiles.com/segment_anything_2"
    CHECKPOINT_DIR="${PROJ_ROOT}/config/checkpoints/sam2"
    mkdir -p "$CHECKPOINT_DIR"

    log_message "Starting SAM2 checkpoints download..."

    declare -A checkpoints=(
        ["sam2.1_hiera_large.pt"]="092824/sam2.1_hiera_large.pt"
    )

    for key in "${!checkpoints[@]}"; do
        url="${BASE_URL}/${checkpoints[$key]}"
        log_message "Downloading ${key}..."
        wget -q --show-progress "$url" -O "${CHECKPOINT_DIR}/${key}" || handle_error "Failed to download ${key}"
    done

    log_message "SAM2 checkpoints downloaded successfully."
}

# Function to download X-Mem checkpoints
download_xmem_checkpoints() {
    BASE_URL="https://github.com/hkchengrex/XMem/releases/download"
    CHECKPOINT_DIR="${PROJ_ROOT}/config/checkpoints/xmem"
    mkdir -p "$CHECKPOINT_DIR"

    log_message "Starting XMem checkpoints download..."

    declare -A checkpoints=(
        ["XMem.pth"]="v1.0/XMem.pth"
    )

    for key in "${!checkpoints[@]}"; do
        url="${BASE_URL}/${checkpoints[$key]}"
        log_message "Downloading ${key}..."
        wget -q --show-progress "$url" -O "${CHECKPOINT_DIR}/${key}" || handle_error "Failed to download ${key}"
    done

    log_message "XMem checkpoints downloaded successfully."
}

# Function to download MediaPipe checkpoints
download_mediapipe_checkpoints() {
    BASE_URL="https://storage.googleapis.com/mediapipe-models"
    CHECKPOINT_DIR="${PROJ_ROOT}/config/checkpoints/mediapipe"
    mkdir -p "$CHECKPOINT_DIR"

    log_message "Starting MediaPipe checkpoints download..."

    declare -A checkpoints=(
        ["hand_landmarker.task"]="hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
    )

    for key in "${!checkpoints[@]}"; do
        url="${BASE_URL}/${checkpoints[$key]}"
        log_message "Downloading ${key}..."
        wget -q --show-progress "$url" -O "${CHECKPOINT_DIR}/${key}" || handle_error "Failed to download ${key}"
    done

    log_message "MediaPipe checkpoints downloaded successfully."
}

# Function to download FoundationPose checkpoints
download_fd_checkpoints() {
    refiner_url="https://drive.google.com/drive/folders/1BEQLZH69UO5EOfah-K9bfI3JyP9Hf7wC?usp=sharing"
    scorer_url="https://drive.google.com/drive/folders/12Te_3TELLes5cim1d7F7EBTwUSe7iRBj?usp=sharing"
    CHECKPOINT_DIR="${PROJ_ROOT}/third_party/FoundationPose/weights"

    log_message "Starting FoundationPose checkpoints download..."

    # Refiner
    log_message "Downloading refiner checkpoint..."
    gdown --fuzzy "$refiner_url" --folder --output "${CHECKPOINT_DIR}/" || handle_error "Failed to download refiner checkpoint"

    # Scorer
    log_message "Downloading scorer checkpoint..."
    gdown --fuzzy "$scorer_url" --folder --output "${CHECKPOINT_DIR}/" || handle_error "Failed to download scorer checkpoint"

    log_message "FoundationPose checkpoints downloaded successfully."
}

# Start the downloads
start_time=$(date +%s)

log_message "Initiating checkpoint downloads..."

if $DOWNLOAD_SAM2; then download_sam2_checkpoints; fi
if $DOWNLOAD_XMEM; then download_xmem_checkpoints; fi
if $DOWNLOAD_MEDIAPIPE; then download_mediapipe_checkpoints; fi
if $DOWNLOAD_FOUNDATIONPOSE; then download_fd_checkpoints; fi

end_time=$(date +%s)
duration=$((end_time - start_time))

log_message "All checkpoints downloaded successfully in $duration seconds."
