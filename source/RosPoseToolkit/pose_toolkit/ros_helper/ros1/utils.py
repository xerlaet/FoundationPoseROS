import numpy as np
import cv2
import sys
import rospy


# Encoding mapping to dtype and channel count
encoding_dict = {
    "16UC1": (np.uint16, 1),
    "8UC3": (np.uint8, 3),
    "bgr8": (np.uint8, 3),
    "rgb8": (np.uint8, 3),
    "bgra8": (np.uint8, 4),
    "rgba8": (np.uint8, 4),
}
# Color conversion mapping
color_conversion = {
    "rgb8": cv2.COLOR_RGB2BGR,
    "rgba8": cv2.COLOR_RGBA2BGR,
    "bgra8": cv2.COLOR_BGRA2BGR,
}


def msgImage_to_cv2(img_msg):
    if img_msg.encoding not in encoding_dict:
        raise ValueError(f"Unknown encoding: {img_msg.encoding}")

    dtype, n_channels = encoding_dict[img_msg.encoding]
    dtype = np.dtype(dtype).newbyteorder(">" if img_msg.is_bigendian else "<")

    # Convert data buffer to numpy array
    cv_im = np.ndarray(
        shape=((img_msg.height, img_msg.width, n_channels) if n_channels > 1 else (img_msg.height, img_msg.width)),
        dtype=dtype,
        buffer=img_msg.data,
    )

    # If the byt order is different between the message and the system.
    if img_msg.is_bigendian == (sys.byteorder == "little"):
        cv_im = cv_im.byteswap().newbyteorder()

    # Apply color conversion if necessary
    if img_msg.encoding in color_conversion:
        cv_im = cv2.cvtColor(cv_im, color_conversion[img_msg.encoding])

    return cv_im


def is_roscore_running():
    """Check if roscore is running"""
    try:
        rospy.get_master().getSystemState()
        return True
    except (rospy.ROSException, rospy.exceptions.ROSInterruptException):
        return False


def init_ros_node(node_name, anonymous=True, log_level=rospy.INFO):
    """Initialize a ROS node"""
    if not is_roscore_running():
        raise RuntimeError("roscore is not running. Please start roscore first.")
    try:
        rospy.init_node(node_name, anonymous=anonymous, log_level=log_level)
        rospy.loginfo(f"ROS node '{node_name}' initialized successfully.")
    except Exception as e:
        raise RuntimeError(f"Failed to initialize ROS node: {e}")
