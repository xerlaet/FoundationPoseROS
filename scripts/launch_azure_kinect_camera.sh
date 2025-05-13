#! /bin/bash
# author: Jikai Wang
# email: jikai.wang AT utdallas DOT edu

# load variables from settings.sh
source "$(dirname "$0")/ros_config.sh"

# Check if the number of arguments is correct
if [ "$1" -ge "${#K4A_CAMERAS[@]}" ]; then
  echo "[ERROR] Invalid camera index: $1. Must be between 0 and $((${#K4A_CAMERAS[@]} - 1))"
  exit 1
fi

if [ $# -eq 0 ]; then
  echo "Usage: bash ${0} <camera_id>"
  exit 1
fi

CAM=${K4A_CAMERAS[$1]}

# launch Azure Kinect
roslaunch ${K4A_LAUNCH_FILE} \
  sensor_sn:=${CAM} \
  camera:=${CAM} \
  # tf_prefix:=kinect_${CAM} \
  depth_enabled:=true depth_mode:=${K4A_DEPTH_MOD} \
  color_enabled:=true color_resolution:=${K4A_RESOLUTION} \
  fps:=${K4A_FPS} \
  point_cloud:=false rgb_point_cloud:=false point_cloud_in_depth_frame:=false \
  body_tracking_enabled:=${K4A_FLAG_BODY_TRACKING}
