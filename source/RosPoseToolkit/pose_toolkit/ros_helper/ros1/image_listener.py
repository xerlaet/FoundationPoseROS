import sys
import cv2
import numpy as np
import rospy
from message_filters import ApproximateTimeSynchronizer, Subscriber
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
from image_geometry import PinholeCameraModel


class ImageListener:
    def __init__(self, topic_list, slop=0.1, debug=False):
        self._init_node(node_id="ImageListenserNode", debug=debug)
        if not self._is_roscore_running():
            raise RuntimeError("roscore is not running")
        self._topic_list = topic_list
        self._queue_size = len(topic_list) * 2
        self._slop = slop
        self._prev_stamp = -1
        self.synced_imgs = []
        self._K_dict, self._img_dims = self._load_intrinsics_from_camInfo()

    def start(self):
        def callback(*msgs):
            if msgs:
                stamp = msgs[0].header.stamp.to_nsec()
                if stamp != self._prev_stamp:
                    self._prev_stamp = stamp
                    self.synced_imgs = [self._imgmsg_to_cv2(msg) for msg in msgs]
            else:
                rospy.logwarn("No synced messages received")

        self._sub_list = [Subscriber(t, Image, queue_size=10) for t in self._topic_list]
        self._ts = ApproximateTimeSynchronizer(self._sub_list, 3 * len(self._sub_list), self._slop)
        self._ts.registerCallback(callback)

    def stop(self):
        for sub in self._sub_list:
            sub.unregister()
        rospy.signal_shutdown("user requested...")

    def _init_node(self, node_id, debug):
        if not self._is_roscore_running():
            raise RuntimeError("roscore is not running")
        # Initialize node
        if not rospy.core.is_initialized():
            try:
                rospy.init_node(node_id, anonymous=False, log_level=rospy.DEBUG if debug else rospy.INFO)
                rospy.loginfo_once(f"Node '{node_id}' initialized success...")
            except Exception as e:
                rospy.logerr_once(f"Node '{node_id}' initialization failed: {e}.")

    def _imgmsg_to_cv2(self, img_msg):
        """Convert sensor_msgs.Image to numpy.ndarray.

        Args:
            img_msg (sensor_msgs.Image): Image message.

        Returns:
            np.ndarray: Image as numpy array.
        """
        if img_msg.encoding == "16UC1":
            dtype, n_channels = np.uint16, 1
        elif img_msg.encoding == "32FC1":
            dtype, n_channels = np.float32, 1
        elif img_msg.encoding == "bgr8":
            dtype, n_channels = np.uint8, 3
        elif img_msg.encoding == "rgb8":
            dtype, n_channels = np.uint8, 3
        elif img_msg.encoding == "bgra8":
            dtype, n_channels = np.uint8, 4
        elif img_msg.encoding == "rgba8":
            dtype, n_channels = np.uint8, 4
        else:
            raise ValueError(f"Unknown encoding: {img_msg.encoding}")

        dtype = np.dtype(dtype)
        dtype = dtype.newbyteorder(">" if img_msg.is_bigendian else "<")
        tmp_data = img_msg.data

        if n_channels == 1:
            im = np.ndarray(shape=(img_msg.height, img_msg.width), dtype=dtype, buffer=tmp_data)
        else:
            im = np.ndarray(
                shape=(img_msg.height, img_msg.width, n_channels),
                dtype=dtype,
                buffer=tmp_data,
            )

        # If the byt order is different between the message and the system.
        if img_msg.is_bigendian == (sys.byteorder == "little"):
            im = im.byteswap().newbyteorder()

        if img_msg.encoding == "bgr8":
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        if img_msg.encoding == "rgba8":
            im = cv2.cvtColor(im, cv2.COLOR_RGBA2RGB)
        if img_msg.encoding == "bgra8":
            im = cv2.cvtColor(im, cv2.COLOR_BGRA2RGB)

        return im

    def _load_intrinsics_from_camInfo(self):
        rospy.logdebug("Retrieving intrinsics...")
        cam_model = PinholeCameraModel()
        K = {}
        image_dims = {}
        for topic in self._topic_list:
            caminfo_topic = topic.replace("image_raw", "camera_info")
            camInfo_msg = rospy.wait_for_message(caminfo_topic, CameraInfo)
            cam_model.fromCameraInfo(camInfo_msg)
            K[topic] = cam_model.fullIntrinsicMatrix().astype("float32")
            image_dims[topic] = (cam_model.width, cam_model.height)
        rospy.logdebug("Intrinsics retrieved.")
        return K, image_dims

    def _is_roscore_running(self):
        try:
            rospy.get_master().getPid()
            return True
        except:
            return False

    @property
    def next_images(self):
        """Return the next images as a list of numpy arrays. (RGB, Depth)"""
        return self.synced_imgs if self.synced_imgs else None

    @property
    def camera_intrinsics(self):
        """Return the camera intrinsics as a dictionary."""
        return self._K_dict

    @property
    def image_dims(self):
        """Return the image dimensions as a dictionary."""
        return self._img_dims
