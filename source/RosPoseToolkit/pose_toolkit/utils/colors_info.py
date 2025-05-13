class RGBA:
    def __init__(self, red, green, blue, alpha=255):
        """
        Initialize an RGBA color.
        :param red: Red channel (0-255)
        :param green: Green channel (0-255)
        :param blue: Blue channel (0-255)
        :param alpha: Alpha channel (0-255), default is 255 (opaque)
        """
        self.red = red
        self.green = green
        self.blue = blue
        self.alpha = alpha

    def __str__(self):
        return "({},{},{},{})".format(self.red, self.green, self.blue, self.alpha)

    @property
    def hex(self):
        """Return the hexadecimal representation of the color."""
        return "#{:02X}{:02X}{:02X}".format(self.red, self.green, self.blue)

    @property
    def rgba(self):
        """Return a tuple of the RGBA values."""
        return (self.red, self.green, self.blue, self.alpha)

    @property
    def rgb(self):
        """Return a tuple of the RGB values."""
        return (self.red, self.green, self.blue)

    @property
    def bgra(self):
        """Return a tuple of the BGRA values (Blue, Green, Red, Alpha)."""
        return (self.blue, self.green, self.red, self.alpha)

    @property
    def bgr(self):
        """Return a tuple of the BGR values (Blue, Green, Red)."""
        return (self.blue, self.green, self.red)

    @property
    def rgba_norm(self):
        """Return normalized RGBA values (0 to 1)."""
        return (
            self.red / 255.0,
            self.green / 255.0,
            self.blue / 255.0,
            self.alpha / 255.0,
        )

    @property
    def rgb_norm(self):
        """Return normalized RGB values (0 to 1)."""
        return (self.red / 255.0, self.green / 255.0, self.blue / 255.0)

    @property
    def bgra_norm(self):
        """Return normalized BGRA values (0 to 1)."""
        return (
            self.blue / 255.0,
            self.green / 255.0,
            self.red / 255.0,
            self.alpha / 255.0,
        )

    @property
    def bgr_norm(self):
        """Return normalized BGR values (0 to 1)."""
        return (self.blue / 255.0, self.green / 255.0, self.red / 255.0)


COLORS = {
    "red": RGBA(255, 0, 0),
    "dark_red": RGBA(139, 0, 0),
    "green": RGBA(0, 255, 0),
    "dark_green": RGBA(0, 100, 0),
    "blue": RGBA(0, 0, 255),
    "yellow": RGBA(255, 255, 0),
    "magenta": RGBA(255, 0, 255),
    "cyan": RGBA(0, 255, 255),
    "orange": RGBA(255, 165, 0),
    "purple": RGBA(128, 0, 128),
    "brown": RGBA(165, 42, 42),
    "pink": RGBA(255, 192, 203),
    "lime": RGBA(0, 255, 0),
    "navy": RGBA(0, 0, 128),
    "teal": RGBA(0, 128, 128),
    "olive": RGBA(128, 128, 0),
    "maroon": RGBA(128, 0, 0),
    "coral": RGBA(255, 127, 80),
    "turquoise": RGBA(64, 224, 208),
    "indigo": RGBA(75, 0, 130),
    "violet": RGBA(238, 130, 238),
    "gold": RGBA(255, 215, 0),
    "skin": RGBA(255, 219, 172),
    "white": RGBA(255, 255, 255),
    "black": RGBA(0, 0, 0),
    "gray": RGBA(128, 128, 128),
    "dark_gray": RGBA(64, 64, 64),
    "light_gray": RGBA(211, 211, 211),
    "tomato": RGBA(255, 99, 71),
    "deepskyblue": RGBA(0, 128, 255),
    # Tab10 colors
    "tab10_0": RGBA(31, 119, 180),
    "tab10_1": RGBA(255, 127, 14),
    "tab10_2": RGBA(44, 160, 44),
    "tab10_3": RGBA(214, 39, 40),
    "tab10_4": RGBA(148, 103, 189),
    "tab10_5": RGBA(140, 86, 75),
    "tab10_6": RGBA(227, 119, 194),
    "tab10_7": RGBA(127, 127, 127),
    "tab10_8": RGBA(188, 189, 34),
    "tab10_9": RGBA(23, 190, 207),
    "test": RGBA(250, 74, 43),
}

# RGB colors for Object classes
OBJ_CLASS_COLORS = [
    COLORS["black"],  # background
    COLORS["tab10_0"],  # object 1
    COLORS["tab10_1"],  # object 2
    COLORS["tab10_2"],  # object 3
    COLORS["tab10_3"],  # object 4
]

# RGB colors for Hands
HAND_COLORS = [
    COLORS["black"],  # background
    # COLORS["tab10_5"],  # right hand
    # COLORS["tab10_6"],  # left hand
    COLORS["test"],  # right hand
    COLORS["cyan"],  # left hand
]

# RGB colors for HOCap Dataset Segmentation
HO_CAP_SEG_COLOR = [
    COLORS["black"],  # background
    OBJ_CLASS_COLORS[1],  # object 1
    OBJ_CLASS_COLORS[2],  # object 2
    OBJ_CLASS_COLORS[3],  # object 3
    OBJ_CLASS_COLORS[4],  # object 4
    HAND_COLORS[1],  # right hand
    HAND_COLORS[2],  # left hand
]

# RGB colors for Hand Bones
HAND_BONE_COLORS = [
    # Palm connections
    COLORS["gray"],  # (0, 1)
    COLORS["gray"],  # (0, 5)
    COLORS["gray"],  # (0, 17)
    COLORS["gray"],  # (5, 9)
    COLORS["gray"],  # (9, 13)
    COLORS["gray"],  # (13, 17)
    # Thumb
    COLORS["red"],  # (1, 2)
    COLORS["red"],  # (2, 3)
    COLORS["red"],  # (3, 4)
    # Index
    COLORS["green"],  # (5, 6)
    COLORS["green"],  # (6, 7)
    COLORS["green"],  # (7, 8)
    # Middle
    COLORS["blue"],  # (9, 10)
    COLORS["blue"],  # (10, 11)
    COLORS["blue"],  # (11, 12)
    # Ring
    COLORS["yellow"],  # (13, 14)
    COLORS["yellow"],  # (14, 15)
    COLORS["yellow"],  # (15, 16)
    # Pinky
    COLORS["pink"],  # (17, 18)
    COLORS["pink"],  # (18, 19)
    COLORS["pink"],  # (19, 20)
]

# RGB colors for Hand Joints
HAND_JOINT_COLORS = [
    # Wrist (root)
    COLORS["black"],  # 0
    # Thumb joints
    COLORS["red"],  # 1
    COLORS["red"],  # 2
    COLORS["red"],  # 3
    COLORS["red"],  # 4
    # Index joints
    COLORS["green"],  # 5
    COLORS["green"],  # 6
    COLORS["green"],  # 7
    COLORS["green"],  # 8
    # Middle joints
    COLORS["blue"],  # 9
    COLORS["blue"],  # 10
    COLORS["blue"],  # 11
    COLORS["blue"],  # 12
    # Ring joints
    COLORS["yellow"],  # 13
    COLORS["yellow"],  # 14
    COLORS["yellow"],  # 15
    COLORS["yellow"],  # 16
    # Pinky joints
    COLORS["pink"],  # 17
    COLORS["pink"],  # 18
    COLORS["pink"],  # 19
    COLORS["pink"],  # 20
]

SEG_CLASS_COLORS = [
    COLORS["black"],  # background
    COLORS["tab10_0"],  # object 1
    COLORS["tab10_1"],  # object 2
    COLORS["tab10_2"],  # object 3
    COLORS["tab10_3"],  # object 4
    COLORS["tab10_4"],  # object 5
    COLORS["tab10_5"],  # object 6
    COLORS["tab10_6"],  # object 7
    COLORS["tab10_7"],  # object 8
    COLORS["tab10_8"],  # object 9
    COLORS["tab10_9"],  # object 10
]
