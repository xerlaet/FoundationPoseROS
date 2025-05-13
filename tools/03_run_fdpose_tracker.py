import os

from tqdm import tqdm


os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["MPLBACKEND"] = "Agg"  # Disable matplotlib GUI backend

import open3d as o3d
import pyrender
import argparse
import logging
import time
import trimesh
import concurrent.futures
from pose_toolkit.utils.commons import *
from pose_toolkit.rendering.offscreen_renderer import OffscreenRenderer
from pose_toolkit.wrappers.foundationpose import (
    FoundationPose,
    ScorePredictor,
    PoseRefinePredictor,
    set_logging_format,
    set_seed,
    dr,
)


def is_valid_pose(pose):
    return (
        isinstance(pose, np.ndarray)
        and pose.shape == (4, 4)
        and not np.all(pose == -1)
        and not np.all(pose == 0)
    ) and pose is not None


def depth2xyz(depth, K, T=None):
    """Convert depth image to xyz point cloud"""
    H, W = depth.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H), indexing="xy")

    u_flat = u.flatten()
    v_flat = v.flatten()
    depth_flat = depth.flatten()
    depth_flat[depth_flat < 0] = 0

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    x_norm = (u_flat - cx) / fx
    y_norm = (v_flat - cy) / fy

    x = depth_flat * x_norm
    y = depth_flat * y_norm
    z = depth_flat

    xyz = np.stack((x, y, z), axis=1)  # camera space

    if T is not None:
        xyz = xyz @ T[:3, :3].T + T[:3, 3]

    return xyz


def get_init_translation(depth, mask, K, T=None, kernel_size=3):
    """Get initial translation from depth and mask"""
    pts = depth2xyz(depth, K, T)
    pts = pts[erode_mask(mask, kernel_size).flatten().astype(bool)]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    # Remove outliers
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=300, std_ratio=2.0)
    pcd = pcd.select_by_index(ind)
    pts = np.asarray(pcd.points, dtype=np.float32)
    if len(pts) < 100:
        center = None
    else:
        center = pcd.get_center()
    return center, pcd


def load_meta_data(meta_file):
    data = read_data_from_json(meta_file)
    im_height = data["height"]
    im_width = data["width"]
    cam_K = np.array(data["intrinsic_matrix"], dtype=np.float32)
    cam_K = cam_K.reshape((3, 3)).transpose()
    cam_RT = np.eye(4, dtype=np.float32)
    return im_height, im_width, cam_K, cam_RT


def save_poses(poses, save_folder):
    make_clean_folder(save_folder)
    logging.info(f"Saving poses to {save_folder}")
    tqbar = tqdm(total=len(poses))
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(
                write_pose_to_txt,
                Path(save_folder, f"ob_in_cam_{i:06d}.txt"),
                pose,
            )
            for i, pose in enumerate(poses)
        }
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                logging.error(f"Error saving pose: {e}")
            tqbar.update(1)
    tqbar.close()


def run_pose_estimation(
    sequence_folder, est_refine_iter, track_refine_iter, mesh_file, render_pose=False
):
    sequence_folder = Path(sequence_folder)

    # Load parameters from data_loader
    # im_height, im_width, cam_K, cam_RT = load_meta_data(
    #     sequence_folder / "intrinsic.json"
    # )
    im_height = 720
    im_width = 1280
    cam_K = np.loadtxt(sequence_folder/"cam_K.txt", dtype=np.float32).reshape((3, 3))
    cam_RT = np.eye(4, dtype=np.float32)

    # color_files = sorted(sequence_folder.glob("color/*.jpg"))
    # depth_files = sorted(sequence_folder.glob("depth/*.png"))
    color_files = sorted(sequence_folder.glob("images/*.jpg"))
    depth_files = sorted(sequence_folder.glob("images/*.png"))
    mask_files = sorted(sequence_folder.glob("segmentation/sam2/mask/*.png"))
    object_mesh = trimesh.load(mesh_file, process=False)
    num_frames = len(color_files)
    empty_mat_pose = np.full((4, 4), -1.0, dtype=np.float32)

    save_folder = sequence_folder / "fdpose_solver"
    make_clean_folder(save_folder)
    save_pose_folder = save_folder / "ob_in_cam"
    save_pose_folder.mkdir(parents=True, exist_ok=True)
    if render_pose:
        render_images = []
        renderer = OffscreenRenderer(znear=0.1, zfar=100.0)
        renderer.add_camera(cam_K, "camera")
        renderer.add_mesh(object_mesh, "object")
        save_vis_folder = save_folder / "vis"
        save_vis_folder.mkdir(parents=True, exist_ok=True)

    set_seed(0)

    estimator = FoundationPose(
        model_pts=object_mesh.vertices.astype(np.float32),
        model_normals=object_mesh.vertex_normals.astype(np.float32),
        mesh=object_mesh,
        scorer=ScorePredictor(),
        refiner=PoseRefinePredictor(),
        glctx=dr.RasterizeCudaContext(),
        debug=0,
        debug_dir=save_folder / "debug",
        rotation_grid_min_n_views=120,
        rotation_grid_inplane_step=60,
    )

    # Initialize poses
    poses_in_cam = []
    ob_in_cam = empty_mat_pose.copy()
    for frame_id in range(num_frames):
        color = read_rgb_image(color_files[frame_id])
        depth = read_depth_image(depth_files[frame_id], scale=1000.0)
        depth[depth < 0.1] = 0
        depth[depth > 3.0] = 0
        mask = read_mask_image(mask_files[frame_id])

        ob_in_cam_mat = empty_mat_pose.copy()
        if mask.sum() >= 100:
            if is_valid_pose(ob_in_cam):
                ob_in_cam_mat = estimator.track_one(
                    rgb=color,
                    depth=depth,
                    K=cam_K,
                    iteration=track_refine_iter,
                    prev_pose=ob_in_cam,
                )
            else:
                init_ob_pos_center, pcd = get_init_translation(depth, mask, cam_K)
                if init_ob_pos_center is not None:
                    ob_in_cam_mat = estimator.register(
                        rgb=color,
                        depth=depth,
                        ob_mask=mask,
                        K=cam_K,
                        iteration=est_refine_iter,
                        init_ob_pos_center=init_ob_pos_center,
                    )
                    if not is_valid_pose(ob_in_cam_mat):
                        ob_in_cam_mat = empty_mat_pose.copy()

        ob_in_cam = ob_in_cam_mat.copy()
        poses_in_cam.append(mat_to_quat(ob_in_cam_mat))

        if render_pose:
            r_colors = renderer.get_render_colors(
                width=im_width,
                height=im_height,
                cam_names=["camera"],
                cam_poses=[cam_RT],
                mesh_names=["object"],
                mesh_poses=[ob_in_cam_mat],
            )
            vis = draw_image_overlay(color, r_colors[0], 0.6)
            render_images.append(vis)
            # write_rgb_image(save_vis_folder / f"{frame_id:06d}.png", vis)

    save_poses(poses_in_cam, save_pose_folder)
    if render_pose:
        logging.info(f"Saving rendered images...")
        tqbar = tqdm(total=len(render_images))
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(
                    write_rgb_image,
                    save_vis_folder / f"{frame_id:06d}.png",
                    vis,
                )
                for frame_id, vis in enumerate(render_images)
            }
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logging.error(f"Error saving image: {e}")
                tqbar.update(1)
        tqbar.close()
        logging.info("Creating video frames...")
        for frame_id, vis in tqdm(enumerate(render_images)):
            color = read_rgb_image(color_files[frame_id])
            render_images[frame_id] = draw_image_grid([color, vis], facecolor="black")
        logging.info("Creating video...")
        create_video_from_rgb_images(save_folder / "vis.mp4", render_images)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sequence_folder", type=str, default=None, help="Path to the sequence folder."
    )
    parser.add_argument(
        "--est_refine_iter",
        type=int,
        default=15,
        help="number of iterations for estimation",
    )
    parser.add_argument(
        "--track_refine_iter",
        type=int,
        default=5,
        help="number of iterations for tracking",
    )
    parser.add_argument(
        "--mesh_file", type=str, default=None, help="Path to the mesh file."
    )
    parser.add_argument("--render_pose", action="store_true", help="Render the pose.")
    args = parser.parse_args()

    args.sequence_folder = "recordings/G01_1"
    args.mesh_file = "config/mods/G01_1/textured_mesh.obj"
    args.render_pose = True

    if args.sequence_folder is None:
        raise ValueError("Please specify the sequence folder.")
    if args.mesh_file is None:
        raise ValueError("Please specify the mesh file.")

    set_logging_format()
    t_start = time.time()

    run_pose_estimation(
        args.sequence_folder,
        args.est_refine_iter,
        args.track_refine_iter,
        args.mesh_file,
        args.render_pose,
    )

    logging.info(f"done!!! time: {time.time() - t_start:.3f}s.")
