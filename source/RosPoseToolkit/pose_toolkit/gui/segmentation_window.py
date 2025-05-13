import open3d as o3d
from open3d.visualization import gui
from open3d.visualization import rendering

from pathlib import Path
import numpy as np
import cv2
from ultralytics.models.sam import SAM, SAM2Predictor

# ROS imports
import rospy
from sensor_msgs.msg import Image
from pose_toolkit.ros_helper.ros1.utils import msgImage_to_cv2

from pose_toolkit.utils import PROJ_ROOT
from pose_toolkit.utils.commons import (
    draw_annotated_image,
    read_rgb_image,
    write_rgb_image,
    write_depth_image,
    write_mask_image,
    write_data_to_yaml,
)


class SegmentationWindow:
    def __init__(self, image_topics={}, rgb_image=None, device="cuda:0") -> None:
        self._device = device
        self._color_topic = image_topics.get("color", None)
        self._depth_topic = image_topics.get("depth", None)

        # Initialize ROS node
        if self._color_topic is not None:
            self._init_ros_node(node_id="SegmentationWindowNode")

        # Initialize SAM2 Predictor
        self._predictor = self._init_sam_predictor()

        self._image_name = "mask"
        self._save_folder = f"{PROJ_ROOT}/recordings/ros"
        self._points = []
        self._curr_label = 0
        self._masks = []
        self._mask_ids = []
        self._prompt_data = {}
        self._cv_image = None
        self._img_width = 800
        self._img_height = 600
        self._raw_image = (
            np.zeros((self._img_width, self._img_height, 3), dtype=np.uint8) if rgb_image is None else rgb_image.copy()
        )
        self._gui_image_data = (
            cv2.resize(rgb_image, (self._img_width, self._img_height))
            if rgb_image is not None
            else np.zeros((self._img_height, self._img_width, 3), dtype=np.uint8)
        )
        self._curr_mask = np.zeros(self._raw_image.shape[:2], dtype=np.uint8)

        self._create_window("Segmentation Window", self._img_width, self._img_height)

    def _init_ros_node(self, node_id):
        """Initializes the ROS node."""
        if not rospy.core.is_initialized():
            try:
                rospy.init_node(node_id, anonymous=True)
                rospy.loginfo_once(f"Node '{node_id}' initialized success...")
            except rospy.ROSException as e:
                rospy.logerr_once(f"Node '{node_id}' initialization failed: {e}.")
                raise e

    def _create_window(self, title, width=800, height=600):
        self._window = gui.Application.instance.create_window(title, width, height)
        w = self._window
        self._em = w.theme.font_size
        margin = 0.25 * self._em

        # Add 3D scene widget
        self._widget3d = gui.SceneWidget()
        self._widget3d.enable_scene_caching(True)
        self._widget3d.scene = rendering.Open3DScene(w.renderer)
        self._widget3d.scene.set_background([0, 0, 0, 1], o3d.geometry.Image(self._gui_image_data))
        w.add_child(self._widget3d)

        self._panel = gui.Vert(margin, gui.Margins(margin, margin, margin, margin))
        # Add image file chooser
        img_selector = self._create_image_file_chooser_widget()
        self._panel.add_child(img_selector)

        # Add get-latest-image button
        if self._color_topic is not None:
            self._panel.add_child(gui.Label("Fetch Latest Image from ROS"))
            blk = self._create_image_fetch_widget()
            self._panel.add_child(blk)

        # Add buttons to the panel
        button_layout = self._create_mask_control_buttons()
        self._panel.add_child(button_layout)
        # Add mask label
        blk = self._create_mask_label_widget()
        self._panel.add_child(blk)

        # Add widget Proxy
        self._proxy = gui.WidgetProxy()
        self._proxy.set_widget(None)
        self._panel.add_child(self._proxy)

        # Quit button. (Typically this is a menu item)
        button_layout = gui.Horiz()
        ok_button = gui.Button("Save & Exit")
        ok_button.set_on_clicked(self._on_save_and_exit)
        button_layout.add_stretch()
        button_layout.add_child(ok_button)

        self._panel.add_stretch()
        self._panel.add_child(button_layout)

        w.add_child(self._panel)

        # Set callbacks for window
        self._widget3d.set_on_mouse(self._on_mouse_widget3d)
        w.set_on_layout(self._on_layout)
        w.set_on_key(self._on_key)

    def _init_sam_predictor(self):
        """Initializes the SAM predictor."""
        predictor = SAM2Predictor(overrides={"device": self._device, "save": False})
        predictor.setup_model(model=SAM(f"{PROJ_ROOT}/checkpoints/sam2.1_t.pt"))
        return predictor

    def _on_layout(self, ctx):
        """Handles layout changes."""
        panel_width = 20 * self._em
        self._widget3d.frame = gui.Rect(0, 0, self._img_width, self._img_height)
        self._panel.frame = gui.Rect(self._widget3d.frame.get_right(), 0, panel_width, self._img_height)
        self._window.size = gui.Size(self._img_width + panel_width, self._img_height)

    def _reset(self):
        """Resets the current points and mask."""
        self._points = []
        self._update_frame_elements()

    def _update_frame_elements(self):
        """Updates the displayed image."""

        def update_image():
            self._widget3d.scene.set_background([0, 0, 0, 1], o3d.geometry.Image(self._gui_image_data))
            self._widget3d.force_redraw()

        gui.Application.instance.post_to_main_thread(self._window, update_image)

    def _create_image_file_chooser_widget(self):
        """Adds image file chooser to the panel."""

        def on_filedlg_cancel():
            """Handles file dialog cancel."""
            self._window.close_dialog()

        def on_filedlg_done(file_path: str):
            """Handles file dialog done."""
            self._fileedit.text_value = file_path
            _path = Path(file_path).resolve()
            if not _path.exists():
                return
            self._image_name = _path.stem
            self._save_folder = _path.parent
            rgb_image = read_rgb_image(file_path)
            self._raw_image = rgb_image.copy()
            # Resize image to fit the widget
            self._gui_image_data = cv2.resize(rgb_image, (self._img_width, self._img_height))
            self._predictor.set_image(cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB))
            self._masks = []
            self._mask_ids = []
            self._curr_label = 1
            self._intedit.int_value = 1
            self._prompt_data = {}
            self._curr_mask = np.zeros(self._raw_image.shape[:2], dtype=np.uint8)
            self._reset()
            self._proxy.set_widget(self._create_mask_block())
            self._window.set_needs_layout()
            self._window.close_dialog()

        def on_filedlg_button():
            """Handles file dialog button click."""
            filedlg = gui.FileDialog(gui.FileDialog.OPEN, "Select file", self._window.theme)
            filedlg.add_filter(".png .jpg .jpeg", "Image files (*.png;*.jpg;*.jpeg)")
            filedlg.add_filter("", "All files")
            filedlg.set_on_cancel(on_filedlg_cancel)
            filedlg.set_on_done(on_filedlg_done)
            self._window.show_dialog(filedlg)

        self._fileedit = gui.TextEdit()
        filedlgbutton = gui.Button("...")
        filedlgbutton.horizontal_padding_em = 0.5
        filedlgbutton.vertical_padding_em = 0
        filedlgbutton.set_on_clicked(on_filedlg_button)

        margin = 0.25 * self._em
        fileedit_layout = gui.Horiz(margin, gui.Margins(margin, margin, margin, margin))
        fileedit_layout.add_child(gui.Label("Image file"))
        fileedit_layout.add_child(self._fileedit)
        fileedit_layout.add_child(filedlgbutton)

        return fileedit_layout

    def _create_image_fetch_widget(self):
        """Adds image fetch button to the panel."""

        margin = 0.25 * self._em
        button_layout = gui.Vert(margin, gui.Margins(margin, margin, margin, margin))

        # Add text edit for color topic
        def on_color_topic_changed(new_text):
            self._color_topic = new_text
            rospy.loginfo(f"Subscribe to Topic: {self._color_topic}")

        edit_layout = gui.Horiz(0, gui.Margins(margin, margin, margin, margin))
        edit_layout.add_child(gui.Label("Color Topic: "))
        tedit = gui.TextEdit()
        tedit.placeholder_text = f"{self._color_topic}"
        tedit.set_on_value_changed(on_color_topic_changed)
        edit_layout.add_child(tedit)
        button_layout.add_child(edit_layout)

        # Add text edit for depth topic
        def on_depth_topic_changed(new_text):
            self._depth_topic = new_text
            rospy.loginfo(f"Subscribe to Topic: {self._depth_topic}")

        edit_layout = gui.Horiz(0, gui.Margins(margin, margin, margin, margin))
        edit_layout.add_child(gui.Label("Depth Topic: "))
        tedit = gui.TextEdit()
        tedit.placeholder_text = f"{self._depth_topic}"
        tedit.set_on_value_changed(on_depth_topic_changed)
        edit_layout.add_child(tedit)
        button_layout.add_child(edit_layout)

        # Add button to fetch image
        def on_fetch_image():
            """Handles fetch image button click."""
            rospy.loginfo(f"Fetching Image from ROS...")
            self._save_folder = PROJ_ROOT / "recordings" / "ros"
            self._save_folder.mkdir(parents=True, exist_ok=True)
            self._image_name = "ros_image"

            cv_image = None
            depth_image = None
            while not rospy.is_shutdown():
                try:
                    if self._color_topic is not None:
                        color_msg = rospy.wait_for_message(self._color_topic, Image)
                        if color_msg is not None:
                            cv_image = msgImage_to_cv2(color_msg)
                            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
                            write_rgb_image(f"{self._save_folder}/{self._image_name}_color.jpg", rgb_image)
                    if self._depth_topic is not None:
                        depth_msg = rospy.wait_for_message(self._depth_topic, Image)
                        if depth_msg is not None:
                            depth_image = msgImage_to_cv2(depth_msg)
                            write_depth_image(f"{self._save_folder}/{self._image_name}_depth.png", depth_image)
                    if cv_image is not None and self._depth_topic is None:
                        break
                    if cv_image is not None and depth_image is not None:
                        break
                except rospy.ROSException:
                    rospy.logwarn(f"Waiting for image...")
                    continue
            rospy.loginfo(f"Image fetched successfully.")

            self._predictor.set_image(cv_image)
            self._raw_image = rgb_image.copy()
            self._gui_image_data = cv2.resize(rgb_image, (self._img_width, self._img_height))
            self._masks = []
            self._mask_ids = []
            self._curr_label = 1
            self._intedit.int_value = 1
            self._prompt_data = {}
            self._curr_mask = np.zeros(self._raw_image.shape[:2], dtype=np.uint8)
            self._reset()
            self._proxy.set_widget(self._create_mask_block())
            self._window.set_needs_layout()

        fetch_button = gui.Button("Fetch Image")
        fetch_button.set_on_clicked(on_fetch_image)
        button_layout.add_child(fetch_button)

        return button_layout

    def _create_mask_control_buttons(self):
        """Adds buttons to the panel."""

        def on_add_mask():
            """Adds the current mask to the mask list."""
            self._masks.append(self._curr_mask)
            self._mask_ids.append(self._intedit.int_value)
            self._prompt_data[self._intedit.int_value] = {
                "points": np.array(self._points)[:, :2].tolist(),
                "labels": np.array(self._points)[:, 2].tolist(),
            }
            self._curr_mask = np.zeros(self._raw_image.shape[:2], dtype=np.uint8)
            self._curr_label = 0
            self._reset()
            self._proxy.set_widget(self._mask_block())

        def on_remove_mask():
            """Removes the last mask from the mask list."""
            if self._masks:
                self._masks.pop()
                self._mask_ids.pop()
                self._proxy.set_widget(self._mask_block())

        margin = 0.25 * self._em
        button_layout = gui.Horiz(margin, gui.Margins(margin, margin, margin, margin))
        addButton = gui.Button("Add Mask")
        removeButton = gui.Button("Remove Mask")
        addButton.set_on_clicked(on_add_mask)
        removeButton.set_on_clicked(on_remove_mask)
        button_layout.add_stretch()
        button_layout.add_child(addButton)
        button_layout.add_stretch()
        button_layout.add_child(removeButton)
        button_layout.add_stretch()
        return button_layout

    def _create_mask_block(self):
        """Creates a block of mask labels."""
        if not self._mask_ids:
            return None

        layout = gui.Vert(0, gui.Margins(0, 0, 0, 0))
        for idx, label in enumerate(self._mask_ids):
            blk = gui.Horiz(0, gui.Margins(0, 0, 0, 0))
            blk.add_stretch()
            blk.add_child(gui.Label(f"Mask {idx}:"))
            blk.add_stretch()
            blk.add_child(gui.Label(f"Label: {label}"))
            blk.add_stretch()
            layout.add_child(blk)
        return layout

    def _mask_block(self):
        """Creates a block of mask labels."""
        if not self._mask_ids:
            return None
        layout = gui.Vert(0, gui.Margins(0, 0, 0, 0))
        for idx, label in enumerate(self._mask_ids):
            blk = gui.Horiz(0, gui.Margins(0, 0, 0, 0))
            blk.add_stretch()
            blk.add_child(gui.Label(f"Mask {idx}:"))
            blk.add_stretch()
            blk.add_child(gui.Label(f"Label: {label}"))
            blk.add_stretch()
            layout.add_child(blk)
        return layout

    def _create_mask_label_widget(self):
        """Adds mask label to the panel."""
        blk = gui.Horiz(0, gui.Margins(0, 0, 0, 0))

        blk.add_stretch()
        blk.add_child(gui.Label(f"---Current Mask---"))
        blk.add_stretch()

        blk.add_child(gui.Label(f"Label:"))
        self._intedit = gui.NumberEdit(gui.NumberEdit.INT)
        self._intedit.int_value = 0
        self._intedit.set_on_value_changed(self._on_intedit_changed)
        blk.add_child(self._intedit)

        blk.add_stretch()
        return blk

    def _on_key(self, event):
        """Handles key events."""
        if event.key == gui.KeyName.R:  # Reset points
            if event.type == gui.KeyEvent.DOWN:
                self._reset()
                return True

        return False

    def _on_mouse_widget3d(self, event):
        """Handles mouse events in the 3D widget."""
        if event.type == gui.MouseEvent.Type.BUTTON_DOWN and event.is_modifier_down(gui.KeyModifier.CTRL):
            # x = int(event.x - self._widget3d.frame.x)
            # y = int(event.y - self._widget3d.frame.y)
            x = int(event.x - self._widget3d.frame.x) / self._widget3d.frame.width
            y = int(event.y - self._widget3d.frame.y) / self._widget3d.frame.height
            if event.buttons == gui.MouseButton.LEFT.value:
                prompt = (x, y, True)
            elif event.buttons == gui.MouseButton.RIGHT.value:
                prompt = (x, y, False)
            else:
                return gui.Widget.EventCallbackResult.IGNORED

            self._points.append(prompt)
            self._update_sam_mask()
            curr_masks = self._masks + [self._curr_mask]
            curr_labels = [f"obj_{i}" for i in self._mask_ids] + ["curr_obj"]
            curr_image = draw_annotated_image(self._raw_image, masks=curr_masks, labels=curr_labels)
            curr_image = self._draw_points(curr_image, self._points)
            self._gui_image_data = o3d.geometry.Image(cv2.resize(curr_image, (self._img_width, self._img_height)))
            self._update_frame_elements()

            return gui.Widget.EventCallbackResult.HANDLED

        return gui.Widget.EventCallbackResult.IGNORED

    def _update_sam_mask(self):
        """Updates the SAM mask."""
        if self._points:
            points = np.array(self._points)[:, :2][None, ...]
            points *= np.array(self._raw_image.shape[:2][::-1])
            labels = np.array(self._points)[:, 2][None, ...]
            result = self._predictor(points=points, labels=labels)[0]
            self._curr_mask = result.masks.cpu().numpy().data[0]
        else:
            self._curr_mask = np.zeros(self._raw_image.shape[:2], dtype=np.uint8)

    def _draw_points(self, img: np.ndarray, points: list, is_rgb: bool = True) -> np.ndarray:
        """Draws points on the image."""
        img_copy = img.copy()
        for x, y, label in points:
            center = (int(x * img.shape[1]), int(y * img.shape[0]))
            color = (0, 255, 0) if label == 1 else (0, 0, 255)
            if is_rgb:
                color = color[::-1]
            cv2.circle(img_copy, center, 3, color, -1)
        return img_copy

    def _on_intedit_changed(self, value: int):
        """Handles changes in the mask label."""
        self._curr_label = int(value)

    def get_mask(self):
        """Gets the current mask."""
        mask = np.zeros(self._raw_image.shape[:2], dtype=np.uint8)

        if self._masks:
            for idx, m in enumerate(self._masks):
                mask[m > 0] = self._mask_ids[idx]
            mask_vis = draw_annotated_image(
                self._raw_image, masks=self._masks, labels=[f"obj_{i}" for i in self._mask_ids]
            )
        else:
            mask[self._curr_mask > 0] = self._curr_label
            self._prompt_data[self._curr_label] = {
                "points": np.array(self._points)[:, :2].tolist(),
                "labels": np.array(self._points)[:, 2].tolist(),
            }
            mask_vis = draw_annotated_image(self._raw_image, masks=[mask], labels=[f"obj_{self._curr_label}"])

        return mask, mask_vis

    def _on_ok(self):
        """Handles window close."""
        # Save the mask
        mask, mask_vis = self.get_mask()
        write_mask_image(f"{self._save_folder}/ros_mask_image.png", mask)
        # Save the mask visualization
        write_rgb_image(f"{self._save_folder}/ros_mask_vis.png", mask_vis)

        # Close the window
        self._window.close()

    def _on_save_and_exit(self):
        """Saves the current mask and its overlay."""
        self._save_folder.mkdir(parents=True, exist_ok=True)
        mask, mask_vis = self.get_mask()
        # Save mask
        write_mask_image(f"{self._save_folder}/{self._image_name}_mask.png", mask)
        # Save the mask visualization
        write_rgb_image(f"{self._save_folder}/{self._image_name}_mask_vis.png", mask_vis)
        # Save prompt data
        write_data_to_yaml(self._save_folder / f"prompts_{self._image_name}.yaml", self._prompt_data)
        rospy.loginfo(f"Mask saved to {self._save_folder}")
        # Close the window
        self._window.close()


if __name__ == "__main__":
    IMAGE_TOPIC_LIST = {
        "color": "/000684312712/rgb/image_raw",
        "depth": "/000684312712/depth_to_rgb/image_raw",
    }
    gui.Application.instance.initialize()
    w = SegmentationWindow(image_topics=IMAGE_TOPIC_LIST, rgb_image=None)
    # w = SegmentationWindow(image_topics=None, rgb_image=None)
    gui.Application.instance.run()
