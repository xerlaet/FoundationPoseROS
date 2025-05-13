import argparse
import numpy as np
import rospy
from pose_toolkit.utils import PROJ_ROOT
from pose_toolkit.ros_helper.ros1 import ImageListener, FoundationPoseNode
from pose_toolkit.ros_helper.ros1.utils import init_ros_node
from pose_toolkit.utils.commons import read_rgb_image, read_depth_image, read_mask_image, make_clean_folder, quat_to_mat


def main():
    args = args_parser.parse_args()
    if args.init_pose is None:
        assert (
            all([args.init_color, args.init_depth, args.init_mask]) is not None
        ), "Please provide initial color, depth and mask images, or initial pose."

    # Initialize the ROS node
    init_ros_node("foundation_pose")

    # Initialize the ImageListener
    image_listener = ImageListener([args.color_topic, args.depth_topic])
    image_listener.start()

    cam_K = image_listener.camera_intrinsics[args.color_topic]
    W, H = image_listener.image_dims[args.color_topic]

    # Initialize the FoundationPoseNode
    fd_pose_node = FoundationPoseNode(
        cam_K=cam_K,
        mesh_file=args.mesh_file,
        device=args.device,
    )

    # Run FoundationPose Register
    if args.init_pose is None:
        ob_in_cam = fd_pose_node.register(
            rgb_image=read_rgb_image(args.init_color),
            depth_image=read_depth_image(args.init_depth),
            mask_image=read_mask_image(args.init_mask),
        )
    else:
        ob_in_cam = fd_pose_node.register(
            rgb_image=np.zeros((H, W, 3), dtype=np.uint8),
            depth_image=np.zeros((H, W), dtype=np.float32),
            mask_image=np.zeros((H, W), dtype=np.uint8),
        )
        fd_pose_node._prev_pose = quat_to_mat(np.asarray(args.init_pose, dtype=np.float32))

    make_clean_folder(f"{PROJ_ROOT}/debug")

    while not rospy.is_shutdown():
        # Get the next images
        images = image_listener.next_images
        if images is None:
            continue
        rgb_image, depth_image = images

        ob_in_cam = fd_pose_node.track_one(rgb_image, depth_image)
        vis = fd_pose_node.draw_debug_image(rgb_image, ob_in_cam)

        # Publish image
        fd_pose_node.publish_image(vis[:, :, ::-1], rospy.Time.now())
        if ob_in_cam is not None:
            fd_pose_node.publish_pose(ob_in_cam, rospy.Time.now())
        else:
            rospy.logwarn("Lost track of the object!")

    # Stop the ROS node
    rospy.signal_shutdown("Shutting down FoundationPoseNode")


def parse_pose(s):
    try:
        values = [x.strip() for x in s.split(",")]
        print(values)
        exit()
        if len(values) != 7:
            raise argparse.ArgumentTypeError("Expected 7 comma-separated floats for --init_pose")
        return np.array(values, dtype=np.float32)
    except ValueError:
        raise argparse.ArgumentTypeError("All values must be floats")


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser(description="FoundationPoseNode Launcher")
    args_parser.add_argument(
        "--init_color",
        type=str,
        default=f"{PROJ_ROOT}/recordings/ros/ros_image_color.jpg",
        help="Path to the initial color image",
    )
    args_parser.add_argument(
        "--init_depth",
        type=str,
        default=f"{PROJ_ROOT}/recordings/ros/ros_image_depth.png",
        help="Path to the initial depth image",
    )
    args_parser.add_argument(
        "--init_mask",
        type=str,
        default=f"{PROJ_ROOT}/recordings/ros/ros_image_mask.png",
        help="Path to the initial mask image",
    )
    args_parser.add_argument(
        "--init_pose",
        # type=parse_pose,
        type=str,
        nargs="*",
        default=None,
        help="Initial pose for the object in the camera frame, format: qx, qy, qz, qw, tx, ty, tz",
    )
    args_parser.add_argument(
        "--mesh_file",
        type=str,
        default=f"{PROJ_ROOT}/assets/Objects/DexCube/model.obj",
        help="Path to the mesh file",
    )
    args_parser.add_argument(
        "--color_topic",
        type=str,
        default="/000684312712/rgb/image_raw",
        help="ROS topic for the color image",
    )
    args_parser.add_argument(
        "--depth_topic",
        type=str,
        default="/000684312712/depth_to_rgb/image_raw",
        help="ROS topic for the depth image",
    )
    args_parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for computation (cpu or cuda)",
    )

    IMAGE_TOPIC_LIST = [
        "/000684312712/rgb/image_raw",
        "/000684312712/depth_to_rgb/image_raw",
    ]

    main()
