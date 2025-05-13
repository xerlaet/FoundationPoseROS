#! /bin/bash
# author: Jikai Wang
# email: jikai.wang AT utdallas DOT edu

# Load variables from settings.sh
source "$(dirname "$0")/ros_config.sh"

# Check if the number of arguments is correct
if [ "$1" -ge "${#RS_CAMERAS[@]}" ]; then
  echo "[ERROR] Invalid camera index: $1. Must be between 0 and $((${#RS_CAMERAS[@]} - 1))"
  exit 1
fi

if [ $# -eq 0 ]; then
  echo "Usage: bash ${0} <camera_id>"
  exit 1
fi

CAM=${RS_CAMERAS[$1]}

roslaunch ${RS_LAUNCH_FILE} \
  serial_no:=${CAM} \
  camera:=${CAM} \
  depth_width:=${RS_WIDTH} depth_height:=${RS_HEIGHT} depth_fps:=${RS_FPS} \
  color_width:=${RS_WIDTH} color_height:=${RS_HEIGHT} color_fps:=${RS_FPS} \
  align_depth:=${RS_ALIGN_DEPTH}
