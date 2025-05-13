#! /bin/bash
# author: Jikai Wang
# email: jikai.wang AT utdallas DOT edu


# get the current directory path
CUR_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# load variables from settings.sh
source ${CUR_DIR}/config.sh

HOLO_NAMESAPCE="hololens_kv5h72"

# AprilTag
gnome-terminal --window \
    -- zsh -c \
    "ROS_NAMESPACE=${HOLO_NAMESAPCE} \
    roslaunch ${TAG_LAUNCH_FILE} \
    camera_name:=/${HOLO_NAMESAPCE}/sensor_pv \
    image_topic:=image_raw \
    camera_frame:=${HOLO_NAMESAPCE}_pv_optical_frame \
    publish_tag_detections_image:=false"