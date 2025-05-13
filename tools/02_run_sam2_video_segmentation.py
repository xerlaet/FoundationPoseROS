import gc
import os

from tqdm import tqdm

os.environ["MPLBACKEND"] = "Agg"  # Disable matplotlib GUI backend

import concurrent.futures
import torch
from pose_toolkit.utils.commons import *
from pose_toolkit.wrappers.sam2 import build_sam2_video_predictor


class Sam2VideoSegmentation:
    def __init__(self, sequence_folder, device="cuda"):
        self._logger = get_logger(self.__class__.__name__)
        self._device = device

        self._data_folder = Path(sequence_folder)
        self._init_seg_folder = self._data_folder / "segmentation/init"
        self._xmem_seg_folder = self._data_folder / "segmentation/xmem"
        self._sam2_seg_folder = self._data_folder / "segmentation/sam2"
        self._load_metadata()

        if "cuda" in self._device:
            torch.autocast(self._device, dtype=torch.bfloat16).__enter__()
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

        self._predictor = build_sam2_video_predictor(
            config_file="config/sam2_config/sam2.1_hiera_l.yaml",
            ckpt_path="config/checkpoints/sam2/sam2.1_hiera_large.pt",
            device=self._device,
        )

    def _load_metadata(self):
        image_files = sorted(self._data_folder.glob("color/*.jpg"))
        img = read_rgb_image(image_files[0])
        self._num_frames = len(image_files)
        self._width = img.shape[1]
        self._height = img.shape[0]

    def _frame_id_to_index(self, frame_id, start, end):
        return frame_id - start if start < end else start - frame_id

    def _index_to_frame_id(self, index, start, end):
        return start + index if start < end else start - index

    def _get_mask_indices(self, file_paths):
        mask_inds = []
        for file_path in file_paths:
            mask_inds.append(int(file_path.stem.split("_")[-1]))
        return mask_inds

    def _get_init_mask_files(self):
        mask_files = sorted(self._init_seg_folder.glob("mask_*.png"))
        mask_inds = self._get_mask_indices(mask_files)
        return mask_files, mask_inds

    def _retrieve_masks_from_video_segments(self, video_segments):
        def _get_mask_image(frame_id):
            mask_image = np.zeros((height, width), dtype=np.uint8)
            for obj_id, obj_mask in frame_segments[frame_id].items():
                mask_image[obj_mask > 0] = obj_id
            return mask_image

        frame_segments = video_segments["frame_segments"]
        height = video_segments["height"]
        width = video_segments["width"]

        mask_images = [None] * len(frame_segments)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(_get_mask_image, frame_id): frame_id
                for frame_id in frame_segments
            }
            for future in concurrent.futures.as_completed(futures):
                frame_id = futures[future]
                mask_images[frame_id] = future.result()
        return mask_images

    def _get_video_segments(self, mask_files, mask_inds):
        height, width = read_mask_image(mask_files[0]).shape[:2]
        frame_segments = {}
        for i in range(len(mask_inds)):
            start = mask_inds[i]
            end = mask_inds[i + 1] if i + 1 < len(mask_inds) else self._num_frames
            mask_file = mask_files[i]

            inference_state = self._predictor.init_state(
                img_paths=[
                    self._data_folder / "color" / f"color_{frame_idx:06d}.jpg"
                    for frame_idx in range(start, end)
                ],
                offload_video_to_cpu=True,
                offload_state_to_cpu=False,
                async_loading_frames=False,
            )
            self._predictor.reset_state(inference_state)

            mask = read_mask_image(mask_file)
            for ann_obj_id in np.unique(mask):
                if ann_obj_id == 0:  # Skip background
                    continue
                self._predictor.add_new_mask(
                    inference_state=inference_state,
                    frame_idx=self._frame_id_to_index(start, start, end),
                    obj_id=ann_obj_id,
                    mask=(mask == ann_obj_id).astype(np.uint8),
                )

            for (
                out_frame_idx,
                out_obj_ids,
                out_mask_logits,
            ) in self._predictor.propagate_in_video(inference_state):
                out_frame_id = self._index_to_frame_id(out_frame_idx, start, end)
                frame_segments[out_frame_id] = {
                    out_obj_id: (out_mask_logits[idx] > 0.0)
                    .cpu()
                    .numpy()
                    .astype(np.uint8)
                    .reshape(height, width)
                    for idx, out_obj_id in enumerate(out_obj_ids)
                }

        # Handle case where initial mask index is greater than 0
        if mask_inds[0] > 0:
            start = mask_inds[0]
            end = -1
            mask_file = mask_files[0]

            inference_state = self._predictor.init_state(
                img_paths=[
                    self._data_folder / "color" / f"color_{frame_idx:06d}.jpg"
                    for frame_idx in range(start, end, -1)
                ],
                offload_video_to_cpu=True,
                offload_state_to_cpu=False,
                async_loading_frames=False,
            )
            self._predictor.reset_state(inference_state)

            mask = read_mask_image(mask_file)
            for ann_obj_id in np.unique(mask):
                if ann_obj_id == 0:
                    continue
                self._predictor.add_new_mask(
                    inference_state=inference_state,
                    frame_idx=self._frame_id_to_index(start, start, end),
                    obj_id=ann_obj_id,
                    mask=(mask == ann_obj_id).astype(np.uint8),
                )

            for (
                out_frame_idx,
                out_obj_ids,
                out_mask_logits,
            ) in self._predictor.propagate_in_video(inference_state):
                out_frame_id = self._index_to_frame_id(out_frame_idx, start, end)
                frame_segments[out_frame_id] = {
                    out_obj_id: (out_mask_logits[idx] > 0.0)
                    .cpu()
                    .numpy()
                    .astype(np.uint8)
                    .reshape(height, width)
                    for idx, out_obj_id in enumerate(out_obj_ids)
                }

        video_segments = {
            "frame_segments": frame_segments,
            "height": height,
            "width": width,
        }

        return video_segments

    def _save_mask_images(
        self,
        mask_images,
        save_mask_folder=None,
        save_vis_folder=None,
        save_video_path=None,
    ):
        num_frames = len(mask_images)

        if save_mask_folder is not None:
            tqdm.write("  - Saving SAM2 masks...")
            make_clean_folder(save_mask_folder)
            tqbar = tqdm(total=num_frames, ncols=100)
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = {
                    executor.submit(
                        write_mask_image,
                        save_mask_folder / f"mask_{i:06d}.png",
                        mask_images[i],
                    ): i
                    for i in range(num_frames)
                }
                for future in concurrent.futures.as_completed(futures):
                    future.result()
                    tqbar.update()
            tqbar.close()

        if save_vis_folder is not None or save_video_path is not None:
            tqdm.write("  - Generating vis images...")
            vis_images = [None] * num_frames
            tqbar = tqdm(total=num_frames, ncols=100)
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = {
                    executor.submit(
                        draw_segmentation_mask_overlay,
                        read_rgb_image(
                            self._data_folder / "color" / f"color_{i:06d}.jpg"
                        ),
                        mask_images[i],
                        alpha=0.65,
                        reduce_bg=True,
                    ): i
                    for i in range(num_frames)
                }
                for future in concurrent.futures.as_completed(futures):
                    i = futures[future]
                    vis_images[i] = future.result()
                    tqbar.update()
            tqbar.close()

        if save_vis_folder is not None:
            make_clean_folder(save_vis_folder)
            tqdm.write("  - Saving vis images...")
            tqbar = tqdm(total=self._num_frames, ncols=100)
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = {
                    executor.submit(
                        write_rgb_image,
                        save_vis_folder / f"vis_{i:06d}.png",
                        vis_images[i],
                    ): i
                    for i in range(self._num_frames)
                }
                for future in concurrent.futures.as_completed(futures):
                    future.result()
                    tqbar.update()
            tqbar.close()

        if save_video_path is not None:
            tqdm.write("  - Saving vis video...")
            create_video_from_rgb_images(save_video_path, vis_images, fps=30)

    def _save_vis_images(self, mask_images, save_vis_folder=None, save_video_path=None):
        tqdm.write("  - Generating vis images...")
        num_frames = len(mask_images)
        vis_images = [None] * num_frames
        tqbar = tqdm(total=num_frames, ncols=100)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(
                    draw_segmentation_mask_overlay,
                    read_rgb_image(self._data_folder / "color" / f"color_{i:06d}.jpg"),
                    mask_images[i],
                    alpha=0.65,
                    reduce_bg=False,
                ): i
                for i in range(num_frames)
            }
            for future in concurrent.futures.as_completed(futures):
                i = futures[future]
                try:
                    vis_images[i] = future.result()
                except Exception as e:
                    self._logger.error(f"Error processing frame {i}: {e}")
                tqbar.update()
        tqbar.close()

        if save_vis_folder is not None:
            make_clean_folder(save_vis_folder)
            tqdm.write("  - Saving vis images...")
            tqbar = tqdm(total=self._num_frames, ncols=100)
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = {
                    executor.submit(
                        write_rgb_image,
                        save_vis_folder / f"vis_{i:06d}.png",
                        vis_images[i],
                    ): i
                    for i in range(self._num_frames)
                }
                for future in concurrent.futures.as_completed(futures):
                    i = futures[future]
                    try:
                        future.result()
                    except Exception as e:
                        self._logger.error(f"Error saving vis image {i}: {e}")
                    tqbar.update()
            tqbar.close()

        if save_video_path is not None:
            tqdm.write("  - Saving vis video...")
            create_video_from_rgb_images(save_video_path, vis_images, fps=30)

    def run(self):
        self._logger.info(f"Processing sequence: {self._data_folder.name}")

        mask_files, mask_inds = self._get_init_mask_files()
        if mask_files is None:
            return

        # Predict video segments using SAM2
        video_segments = self._get_video_segments(mask_files, mask_inds)

        # Retrieve mask images from video segments
        mask_images = self._retrieve_masks_from_video_segments(video_segments)
        video_segments.clear()

        # Save mask images, vis images and video
        save_mask_folder = self._sam2_seg_folder / "mask"
        save_vis_folder = self._sam2_seg_folder / "vis"
        save_video_path = self._sam2_seg_folder.parent / f"vis_sam2.mp4"
        self._save_mask_images(
            mask_images, save_mask_folder, save_vis_folder, save_video_path
        )
        mask_images.clear()

        del mask_images, video_segments
        gc.collect()


if __name__ == "__main__":
    segmenter = Sam2VideoSegmentation("recordings/demo_video")
    segmenter.run()
