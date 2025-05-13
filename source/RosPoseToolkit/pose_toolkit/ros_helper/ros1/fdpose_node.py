import numpy as np
import trimesh

# ROS imports
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from tf import transformations

from pose_toolkit.ros_helper.ros1 import ImageListener
from pose_toolkit.wrappers.foundationpose import (
    FoundationPose,
    ScorePredictor,
    PoseRefinePredictor,
    set_seed,
    dr,
    draw_posed_3d_box,
    draw_xyz_axis,
)
from pose_toolkit.utils.commons import read_rgb_image, read_depth_image, read_mask_image, make_clean_folder
from pose_toolkit.utils import PROJ_ROOT


class FoundationPoseNode:
    def __init__(self, cam_K, mesh_file=None, est_refine_iter=15, track_refine_iter=5, device="cuda"):
        set_seed(0)

        self._device = device
        self._est_refine_iter = est_refine_iter
        self._track_refine_iter = track_refine_iter
        self._cam_K = np.array(cam_K, dtype=np.float32)
        self._cam_RT = np.eye(4, dtype=np.float32)

        self._prev_pose = None
        self._lose_track = True
        self._to_origin = None
        self._extents = None
        self._bbox = None

        # ROS parameters
        self._node_id = "foundation_pose"
        self._pubImageTopic = "/fd_pose/debug/image_raw"
        self._pubPoseTopic = "/fd_pose/track/pose"

        self._estimator = self._init_estimator(mesh_file)

        self._init_ros_node()
        self._bridge = CvBridge()

    def _init_ros_node(self):
        # Initialize node
        try:
            rospy.init_node(name=self._node_id, anonymous=False, log_level=rospy.INFO)
            rospy.loginfo_once("Node '{}' initialized success...".format(self._node_id))
        except Exception as e:
            rospy.logerr_once(f"Node '{self._node_id}' initialization failed: {e}.")

        self._fd_pose_frame_id = f"fd_pose_debug_optical_frame"
        self._imagePub = rospy.Publisher(self._pubImageTopic, Image, queue_size=2)
        self._posePub = rospy.Publisher(self._pubPoseTopic, PoseStamped, queue_size=2)

    def publish_image(self, image, timestamp):
        try:
            msgImage = self._create_msgImage(image, timestamp, self._fd_pose_frame_id)
            self._imagePub.publish(msgImage)
        except Exception as e:
            rospy.logerr_once(f"Failed to publish image: {e}")

    def publish_pose(self, pose, timestamp):
        # Publish pose
        try:
            translation = transformations.translation_from_matrix(pose)
            quaternion = transformations.quaternion_from_matrix(pose)
            msgPose = self._create_msgPose(translation, quaternion, timestamp, self._fd_pose_frame_id)
            self._posePub.publish(msgPose)
        except:
            rospy.logwarn_once("Failed to publish pose from PV camera!")

    def _create_msgImage(self, image_array, timestamp, frame_id):
        """
        ref: http://docs.ros.org/en/noetic/api/sensor_msgs/html/msg/Image.html
        ref: http://wiki.ros.org/cv_bridge/Tutorials/ConvertingBetweenROSImagesAndOpenCVImagesPython
        :param image_array
            numpy array of the image
        :param encoding
            encoding type of the image ()

        :returns msgImage
        """
        msg = self._bridge.cv2_to_imgmsg(image_array)
        msg.header.stamp = timestamp
        msg.header.frame_id = frame_id
        return msg

    def _create_msgPose(self, translation, quaternion, timestamp, frame_id):
        msg = PoseStamped()
        msg.header.stamp = timestamp
        msg.header.frame_id = frame_id
        msg.pose.position.x = translation[0]
        msg.pose.position.y = translation[1]
        msg.pose.position.z = translation[2]
        msg.pose.orientation.x = quaternion[0]
        msg.pose.orientation.y = quaternion[1]
        msg.pose.orientation.z = quaternion[2]
        msg.pose.orientation.w = quaternion[3]
        return msg

    def _init_estimator(self, mesh_file=None):
        rospy.loginfo_once(f"Initializing FoundationPose...")
        if mesh_file is None:
            box_mesh = trimesh.primitives.Box(extents=np.ones((3)), transform=np.eye(4))
            obj_mesh = trimesh.Trimesh(vertices=box_mesh.vertices.copy(), faces=box_mesh.faces.copy())
        else:
            obj_mesh = trimesh.load(str(mesh_file), process=False)
            self._to_origin, self._extents = trimesh.bounds.oriented_bounds(obj_mesh)
            self._bbox = np.stack([-self._extents / 2, self._extents / 2], axis=0).reshape(2, 3)
        estimator = FoundationPose(
            model_pts=obj_mesh.vertices.copy().astype(np.float32),
            model_normals=obj_mesh.vertex_normals.copy().astype(np.float32),
            mesh=obj_mesh,
            symmetry_tfs=None,
            scorer=ScorePredictor(),
            refiner=PoseRefinePredictor(),
            glctx=dr.RasterizeCudaContext(),
            debug=0,
            debug_dir=f"{PROJ_ROOT}/debug",
            rotation_grid_min_n_views=1,
            rotation_grid_inplane_step=360,
        )
        return estimator

    def _is_valid_pose(self, pose):
        return (
            isinstance(pose, np.ndarray) and pose.shape == (4, 4) and not np.all(pose == -1) and not np.all(pose == 0)
        ) and pose is not None

    def set_object_mesh(self, mesh_file, symmetry_tfs=None):
        """Sets the object mesh for the estimator."""
        rospy.loginfo(f"Object mesh updated: {mesh_file}")
        obj_mesh = trimesh.load(mesh_file, process=False)
        self._to_origin, self._extents = trimesh.bounds.oriented_bounds(obj_mesh)
        self._bbox = np.stack([-self._extents / 2, self._extents / 2], axis=0)

        self._estimator.reset_object(
            model_pts=obj_mesh.vertices.copy().astype(np.float32),
            model_normals=obj_mesh.vertex_normals.copy().astype(np.float32),
            mesh=obj_mesh,
            symmetry_tfs=symmetry_tfs,
        )

        rospy.loginfo(f"Object mesh is updated")

    def register(self, rgb_image, depth_image, mask_image):
        """Registers the object model to the scene.
        Args:
            rgb_image (np.ndarray): RGB image.
            depth_image (np.ndarray): Depth image.
            mask_image (np.ndarray): Mask image.
        """
        # Convert images to appropriate format

        # Perform registration
        ob_in_cam = self._estimator.register(
            rgb=rgb_image,
            depth=depth_image,
            ob_mask=mask_image,
            K=self._cam_K,
            iteration=self._est_refine_iter,
        )
        # Update the previous pose
        if not self._is_valid_pose(ob_in_cam):
            ob_in_cam = None
        self._prev_pose = ob_in_cam
        self._lose_track = False
        return ob_in_cam

    def track_one(self, rgb_image, depth_image=None):
        """Tracks the object model in the scene."""
        if self._lose_track:
            return None

        ob_in_cam = self._estimator.track_one(
            rgb=rgb_image,
            depth=depth_image,
            K=self._cam_K,
            iteration=self._track_refine_iter,
            prev_pose=self._prev_pose,
        )
        if not self._is_valid_pose(ob_in_cam):
            ob_in_cam = None
            self._lose_track = True
        self._prev_pose = ob_in_cam
        return ob_in_cam

    def draw_debug_image(self, rgb_image, ob_in_cam):
        """Draws a 3D bounding box around the object."""
        if ob_in_cam is None:
            return rgb_image
        center_pose = ob_in_cam @ np.linalg.inv(self._to_origin)
        vis = draw_posed_3d_box(self._cam_K, rgb_image, ob_in_cam, self._bbox)
        vis = draw_xyz_axis(
            vis,
            center_pose,
            scale=0.05,
            K=self._cam_K,
            thickness=3,
            transparency=0,
            is_input_rgb=True,
        )
        return vis

    # set cam_K
    @property
    def cam_K(self):
        return self._cam_K

    @cam_K.setter
    def cam_K(self, cam_K):
        self._cam_K = cam_K


if __name__ == "__main__":
    IMAGE_TOPIC_LIST = [
        "/000684312712/rgb/image_raw",
        "/000684312712/depth_to_rgb/image_raw",
    ]
    # Initialize the ROS node
    image_listener = ImageListener(IMAGE_TOPIC_LIST)
    image_listener.start()

    cam_K = np.loadtxt(f"{PROJ_ROOT}/recordings/ros_test/cam_K.txt", dtype=np.float32).reshape(3, 3)

    # FoundationPoseNode
    fd_pose_node = FoundationPoseNode(
        cam_K=cam_K,
        mesh_file=f"{PROJ_ROOT}/assets/Objects/DexCube/model.obj",
        device="cuda",
    )

    # Run FoundationPose Register
    rgb_image = read_rgb_image(f"{PROJ_ROOT}/recordings/ros/ros_image_color.jpg")
    depth_image = read_depth_image(f"{PROJ_ROOT}/recordings/ros/ros_image_depth.png")
    mask_image = read_mask_image(f"{PROJ_ROOT}/recordings/ros/ros_image_mask.png")
    ob_in_cam = fd_pose_node.register(
        rgb_image=rgb_image,
        depth_image=depth_image,
        mask_image=mask_image,
    )

    # img_height = 720
    # img_width = 1280

    # ob_in_cam = fd_pose_node.register(
    #     # rgb_image=rgb_image,
    #     # depth_image=depth_image,
    #     # mask_image=mask_image,
    #     rgb_image=np.zeros((img_height, img_width, 3), dtype=np.uint8),
    #     depth_image=np.zeros((img_height, img_width), dtype=np.float32),
    #     mask_image=np.zeros((img_height, img_width), dtype=np.uint8),
    # )
    # init_pose = np.loadtxt(
    #     f"{PROJ_ROOT}/recordings/ros_test/ob_in_cam_000000.txt", dtype=np.float32
    # )
    # fd_pose_node._prev_pose = quat_to_mat(init_pose)

    make_clean_folder(f"{PROJ_ROOT}/debug")

    counter = 0
    while not rospy.is_shutdown():
        # Get the next images
        images = image_listener.next_images
        if images is None:
            continue
        rgb_image, depth_image = images

        ob_in_cam = fd_pose_node.track_one(rgb_image, depth_image)
        vis = fd_pose_node.draw_debug_image(rgb_image, ob_in_cam)

        counter += 1
        # Publish image
        fd_pose_node.publish_image(vis[:, :, ::-1], rospy.Time.now())
        # write_rgb_image(f"{PROJ_ROOT}/debug/vis_{counter:06d}.jpg", vis)
        if ob_in_cam is not None:
            # Publish pose
            fd_pose_node.publish_pose(ob_in_cam, rospy.Time.now())
        else:
            rospy.logwarn("Lost track of the object!")

    # Stop the image listener
    image_listener.stop()
