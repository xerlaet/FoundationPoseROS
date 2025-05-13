#! /bin/bash
# author: Jikai Wang
# email: jikai.wang AT utdallas DOT edu

# Paths
CURR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT=$(dirname ${CURR_DIR})
LAUNCH_DIR="${PROJ_ROOT}/config/launch_files"

# RealSense settings
RS_CAMERAS=(
  "105322251564"
  "043422252387"
  "037522251142"
  "105322251225"
  "108222250342"
  "117222250549"
  "046122250168"
  "115422250549"
)
RS_WIDTH=640
RS_HEIGHT=480
RS_FPS=30
RS_ALIGN_DEPTH="true"
RS_CLIP_DISTANCE=1.5

# Azure Kinect settings
K4A_CAMERAS=(
  "000684312712"
)
K4A_FLAG_BODY_TRACKING="false"
K4A_DEPTH_MOD="NFOV_UNBINNED"
K4A_RESOLUTION="720P"
K4A_FPS=30

# Launch files settings
RS_LAUNCH_FILE="${LAUNCH_DIR}/rs_camera.launch"
K4A_LAUNCH_FILE="${LAUNCH_DIR}/kinect_driver.launch"
TAG_LAUNCH_FILE="${LAUNCH_DIR}/apriltag_detection.launch"
