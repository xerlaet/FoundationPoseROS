from open3d.visualization import gui

import numpy as np
import multiprocessing

from pose_toolkit.utils import PROJ_ROOT
from pose_toolkit.gui.segmentation_window import SegmentationWindow
from pose_toolkit.utils.commons import read_rgb_image, read_depth_image, read_mask_image, make_clean_folder

# Set multiprocessing start method to spawn
try:
    multiprocessing.set_start_method("spawn")
except RuntimeError:
    pass


def run_foundation_pose_node(topic_dict, obj_file, device):
    """Run the FoundationPose node in a separate process."""
    import rospy
    from pose_toolkit.ros_helper.ros1 import FoundationPoseNode, ImageListener

    make_clean_folder(f"{PROJ_ROOT}/debug")

    # Initialize ROS node
    if not rospy.core.is_initialized():
        try:
            rospy.init_node("foundation_pose", anonymous=False)
            rospy.loginfo_once("Node 'foundation_pose' initialized successfully.")
        except Exception as e:
            rospy.logerr_once(f"Node 'foundation_pose' initialization failed: {e}.")
            return

    # Run Image Listener
    image_listener = ImageListener([topic_dict["color"], topic_dict["depth"]])
    image_listener.start()

    cam_K = image_listener.camera_intrinsics[topic_dict["color"]]
    cam_K = np.array(cam_K, dtype=np.float32)

    # Create the FoundationPose node
    fd_pose_node = FoundationPoseNode(cam_K, obj_file, device=device)

    # Run FoundationPose Register
    ob_in_cam = fd_pose_node.register(
        rgb_image=read_rgb_image(f"{PROJ_ROOT}/recordings/ros/ros_image_color.jpg"),
        depth_image=read_depth_image(f"{PROJ_ROOT}/recordings/ros/ros_image_depth.png"),
        mask_image=read_mask_image(f"{PROJ_ROOT}/recordings/ros/ros_image_mask.png"),
    )

    # Run FoundationPose Tracking
    counter = 0
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
        # counter += 1
        # write_rgb_image(f"{PROJ_ROOT}/debug/vis_{counter:06d}.jpg", vis)
        if ob_in_cam is not None:
            # Publish pose
            fd_pose_node.publish_pose(ob_in_cam, rospy.Time.now())
        else:
            rospy.logwarn("Lost track of the object!")

    # Stop the image listener
    image_listener.stop()


class MainWindow:
    MENU_CHECKABLE = 1
    MENU_DISABLED = 2
    MENU_QUIT = 3

    def __init__(self, topic_dict, title="Main Window", device="cuda"):
        self._device = device
        self._assets_dir = PROJ_ROOT / "assets/Objects"
        self._obj_file = None
        self._fdpose_node = None
        self._topic_dict = topic_dict

        self._window = gui.Application.instance.create_window(title, 400, 768)
        w = self._window  # for more concise code
        self._em = w.theme.font_size
        margin = 0.25 * self._em
        layout = gui.Vert(0, gui.Margins(margin, margin, margin, margin))

        # Create menu
        self._create_main_menu()

        fileedit_layout = self._create_file_chooser_widget()
        layout.add_child(fileedit_layout)

        # Create a collapsible vertical widget, which takes up enough vertical
        # space for all its children when open, but only enough for text when
        # closed. This is useful for property pages, so the user can hide sets
        # of properties they rarely use. All layouts take a spacing parameter,
        # which is the spacinging between items in the widget, and a margins
        # parameter, which specifies the spacing of the left, top, right,
        # bottom margins. (This acts like the 'padding' property in CSS.)
        collapse = gui.CollapsableVert("Widgets", margin, gui.Margins(margin, margin, margin, margin))
        # Step 1: Object model selection
        collapse.add_child(gui.Label("Step 1: Choose the object model file"))
        combo = self._create_model_selection_widget()
        collapse.add_child(combo)

        # Step 2: Segmentation
        collapse.add_child(gui.Label("Step 2: Object Segmentation"))
        seg_widget = self._create_segmentation_widget()
        collapse.add_child(seg_widget)

        # Step 3: Initialize FoundationPose
        collapse.add_child(gui.Label("Step 3: FoundationPose ROS"))
        fdpose_widget = self._create_fd_pose_widget()
        collapse.add_child(fdpose_widget)

        # Add tab view widget
        tabs = self._create_tab_view_widget()
        collapse.add_child(tabs)

        # Quit button. (Typically this is a menu item)
        button_layout = gui.Horiz()
        ok_button = gui.Button("Ok")
        ok_button.set_on_clicked(self._on_ok)
        button_layout.add_stretch()
        button_layout.add_child(ok_button)

        layout.add_child(collapse)
        layout.add_stretch()
        layout.add_child(button_layout)

        # We're done, set the window's layout
        w.add_child(layout)

    def _create_main_menu(self):
        if gui.Application.instance.menubar is None:
            menubar = gui.Menu()
            w_menu = gui.Menu()
            w_menu.add_item("An option", MainWindow.MENU_CHECKABLE)
            w_menu.set_checked(MainWindow.MENU_CHECKABLE, True)
            w_menu.add_item("Unavailable feature", MainWindow.MENU_DISABLED)
            w_menu.set_enabled(MainWindow.MENU_DISABLED, False)
            w_menu.add_separator()
            w_menu.add_item("Quit", MainWindow.MENU_QUIT)
            # On macOS the first menu item is the application menu item and will
            # always be the name of the application (probably "Python"),
            # regardless of what you pass in here. The application menu is
            # typically where About..., Preferences..., and Quit go.
            menubar.add_menu("Menu", w_menu)
            gui.Application.instance.menubar = menubar
        self._window.set_on_menu_item_activated(MainWindow.MENU_CHECKABLE, self._on_menu_checkable)
        self._window.set_on_menu_item_activated(MainWindow.MENU_QUIT, self._on_menu_quit)

    def _create_file_chooser_widget(self):
        # Create a file-chooser widget. One part will be a text edit widget for
        # the filename and clicking on the button will let the user choose using
        # the file dialog.
        self._fileedit = gui.TextEdit()
        filedlgbutton = gui.Button("...")
        filedlgbutton.horizontal_padding_em = 0.5
        filedlgbutton.vertical_padding_em = 0
        filedlgbutton.set_on_clicked(self._on_filedlg_button)

        # (Create the horizontal widget for the row. This will make sure the
        # text editor takes up as much space as it can.)
        fileedit_layout = gui.Horiz()
        fileedit_layout.add_child(gui.Label("Model file"))
        fileedit_layout.add_child(self._fileedit)
        fileedit_layout.add_fixed(0.25 * self._em)
        fileedit_layout.add_child(filedlgbutton)

        return fileedit_layout

    def _create_model_selection_widget(self):
        all_mods = sorted([dir.name for dir in self._assets_dir.iterdir() if dir.is_dir()])
        combo = gui.Combobox()
        for mod in all_mods:
            combo.add_item(mod)
        combo.selected_index = 0
        self._obj_file = self._assets_dir / all_mods[0] / "model.obj"
        combo.set_on_selection_changed(self._on_mod_selection)
        return combo

    def _create_segmentation_widget(self):
        def open_window():
            w = SegmentationWindow(image_topics=self._topic_dict, device=self._device)

        btn = gui.Button("Launch Segmentation Window")
        btn.vertical_padding_em = 0.0
        # btn.background_color = gui.Color(r=0, b=0.5, g=0)
        btn.set_on_clicked(open_window)
        return btn

    def _create_fd_pose_widget(self):
        margin = 0.25 * self._em
        fdpose_layout = gui.Vert(margin, gui.Margins(margin, margin, margin, margin))

        # Add register button
        def on_run_fdpose():
            if self._obj_file is None:
                self.show_message_dialog("ERROR", "Please select an object model file first.")
                return
            p = multiprocessing.Process(
                target=run_foundation_pose_node, args=(self._topic_dict, self._obj_file, self._device)
            )
            self._fdpose_node = p
            self._fdpose_node.start()

        buttonReg = gui.Button("Run FoundationPoseROS")
        buttonReg.set_on_clicked(on_run_fdpose)
        buttonReg.vertical_padding_em = 0.0
        fdpose_layout.add_child(buttonReg)
        return fdpose_layout

    def _create_tab_view_widget(self):
        em = 0.25 * self._em
        tabs = gui.TabControl()
        # ========== Tab 1 ==========
        tab1 = gui.Vert(em, gui.Margins(em, em, em, em))
        tab1.add_child(gui.Checkbox("Enable option 1"))
        tab1.add_child(gui.Checkbox("Enable option 2"))
        tab1.add_child(gui.Checkbox("Enable option 3"))
        tabs.add_tab("Options", tab1)

        # ========== Tab 2 ==========
        tab2 = gui.Vert(em, gui.Margins(em, em, em, em))
        tab2.add_child(gui.Label("No plugins detected"))
        tab2.add_stretch()
        tabs.add_tab("Plugins", tab2)

        # ========== Tab 3 ==========
        tab3 = gui.RadioButton(gui.RadioButton.VERT)
        tab3.set_items(["Apple", "Orange"])

        def vt_changed(idx):
            print(f"current cargo: {tab3.selected_value}")

        tab3.set_on_selection_changed(vt_changed)
        tabs.add_tab("Cargo", tab3)

        # ========== Tab 4 ==========
        tab4 = gui.RadioButton(gui.RadioButton.HORIZ)
        tab4.set_items(["Air plane", "Train", "Bus"])

        def hz_changed(idx):
            print(f"current traffic plan: {tab4.selected_value}")

        tab4.set_on_selection_changed(hz_changed)
        tabs.add_tab("Traffic", tab4)
        return tabs

    def _on_filedlg_button(self):
        filedlg = gui.FileDialog(gui.FileDialog.OPEN, "Select file", self._window.theme)
        filedlg.add_filter(".obj .ply .stl", "Triangle mesh (.obj, .ply, .stl)")
        filedlg.add_filter("", "All files")
        filedlg.set_on_cancel(self._on_filedlg_cancel)
        filedlg.set_on_done(self._on_filedlg_done)
        self._window.show_dialog(filedlg)

    def _on_filedlg_cancel(self):
        self._window.close_dialog()

    def _on_filedlg_done(self, path):
        self._fileedit.text_value = path
        self._window.close_dialog()

    def _on_switch(self, is_on):
        if is_on:
            print("Camera would now be running")
        else:
            print("Camera would now be off")

    # This function is essentially the same as window.show_message_box(),
    # so for something this simple just use that, but it illustrates making a
    # dialog.
    def show_message_dialog(self, title, message):
        # A Dialog is just a widget, so you make its child a layout just like
        # a Window.
        dlg = gui.Dialog(title)

        # Add the message text
        self._em = self._window.theme.font_size
        dlg_layout = gui.Vert(self._em, gui.Margins(self._em, self._em, self._em, self._em))
        dlg_layout.add_child(gui.Label(message))

        # Add the Ok button. We need to define a callback function to handle
        # the click.
        ok_button = gui.Button("Ok")
        ok_button.set_on_clicked(self._on_dialog_ok)

        # We want the Ok button to be an the right side, so we need to add
        # a stretch item to the layout, otherwise the button will be the size
        # of the entire row. A stretch item takes up as much space as it can,
        # which forces the button to be its minimum size.
        button_layout = gui.Horiz()
        button_layout.add_stretch()
        button_layout.add_child(ok_button)

        # Add the button layout,
        dlg_layout.add_child(button_layout)
        # ... then add the layout as the child of the Dialog
        dlg.add_child(dlg_layout)
        # ... and now we can show the dialog
        self._window.show_dialog(dlg)

    def _on_dialog_ok(self):
        self._window.close_dialog()

    def _on_mod_selection(self, new_val, new_idx):
        print(new_idx, new_val)
        self._obj_file = self._assets_dir / new_val / "model.obj"
        print(f"[INFO] Object Model Selected: {self._obj_file}")

    def _on_ok(self):
        gui.Application.instance.quit()
        self._fdpose_node.join()

    def _on_menu_checkable(self):
        gui.Application.instance.menubar.set_checked(
            MainWindow.MENU_CHECKABLE,
            not gui.Application.instance.menubar.is_checked(MainWindow.MENU_CHECKABLE),
        )

    def _on_menu_quit(self):
        gui.Application.instance.quit()


if __name__ == "__main__":
    gui.Application.instance.initialize()
    w = MainWindow()
    gui.Application.instance.run()
