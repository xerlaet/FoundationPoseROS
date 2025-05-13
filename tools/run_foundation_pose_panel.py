import argparse
from open3d.visualization import gui
from pose_toolkit.gui.main_window import MainWindow


def main():
    args = args_parser.parse_args()

    # We need to initialize the application, which finds the necessary shaders for
    # rendering and prepares the cross-platform window abstraction.
    gui.Application.instance.initialize()

    w = MainWindow(topic_dict={"color": args.color_topic, "depth": args.depth_topic})

    # Run the event loop. This will not return until the last window is closed.
    gui.Application.instance.run()


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser(description="Run Foundation Pose Panel")
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

    main()
