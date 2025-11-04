import os
import time

import cv2
import pyrealsense2 as rs
import numpy as np

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Define red color range in HSV
LOWER_RED_1 = np.array([0,160,60], dtype=np.uint8)
UPPER_RED_1 = np.array([5,255,255], dtype=np.uint8)       
LOWER_RED_2 = np.array([174,100,60], dtype=np.uint8)
UPPER_RED_2 = np.array([179,255,255], dtype=np.uint8)       

# Define green color range in HSV
LOWER_GREEN = np.array([30, 80, 55], dtype=np.uint8)
UPPER_GREEN = np.array([90, 255, 255], dtype=np.uint8)

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
config_dir = os.path.join(parent_dir, "configs")


colorizer = None
align_to_depth = None
align_to_color = None
class IntelD455ImagePacket:
    """
    Class that contains image and associated processing data.
    """

    @property
    def frame_id(self):
        return self._frame_id

    @property
    def timestamp(self):
        return self._timestamp

    @property
    def image_color(self):
        return self._image_color

    @property
    def image_color_frame(self):
        return self._image_color_frame

    @property
    def image_depth(self):
        return self._image_depth

    @property
    def image_depth_frame(self):
        return self._image_depth_frame

    @property
    def image_color_aligned_frame(self):
        return self._image_color_aligned_frame

    @property
    def image_color_aligned(self):
        if self._image_color_aligned is None:
            self._image_color_aligned = self._align_color_to_depth(self.image_color, self.depth_intrinsics,
                                                                   self.color_intrinsics,
                                                                   self.color_to_depth_extrinsics)
        return self._image_color_aligned

    @property
    def image_depth_aligned_frame(self):
        return self._image_depth_aligned_frame

    @property
    def image_depth_aligned(self):
        if self._image_depth_aligned is None:
            self._image_depth_aligned = self._align_depth_to_color(self.image_depth, self.depth_intrinsics,
                                                                   self.color_intrinsics,
                                                                   self.depth_to_color_extrinsics)
        return self._image_depth_aligned

    @property
    def image_depth_colorized(self, colormap=None, min_value=None, max_value=None, visual_preset=None):
        if self._image_depth_colorized is None:
            self._image_depth_colorized = self._colorize(self.image_depth, colormap, min_value, max_value, visual_preset)
        return self._image_depth_colorized

    @property
    def image_depth_8U(self, min_value=None, max_value=None):
        if min_value is None:
            min_value = np.amin(self.image_depth)
        if max_value is None:
            max_value = np.amax(self.image_depth)
        if (self._image_depth_8U_min != min_value) or (self._image_depth_8U_max != max_value):
            self._image_depth_8U = None
        if self._image_depth_8U is None:
            self._image_depth_8U_min = min_value
            self._image_depth_8U_max = max_value
            self._image_depth_8U = self._convert_to_GRAY_8U(self.image_depth, min_value, max_value)
        return self._image_depth_8U

    @property
    def depth_intrinsics(self):
        return self._depth_intrinsics

    @property
    def depth_intrinsics_raw(self):
        return self._depth_intrinsics_raw

    @property
    def color_intrinsics(self):
        return self._color_intrinsics

    @property
    def color_intrinsics_raw(self):
        return self._color_intrinsics_raw

    @property
    def depth_to_color_extrinsics(self):
        return self._depth_to_color_extrinsics

    @property
    def depth_to_color_extrinsics_raw(self):
        return self._depth_to_color_extrinsics_raw

    @property
    def color_to_depth_extrinsics(self):
        return self._color_to_depth_extrinsics

    @property
    def color_to_depth_extrinsics_raw(self):
        return self._color_to_depth_extrinsics_raw

    @property
    def pointcloud(self):
        if self._pointcloud is None:
            self._pointcloud = self._create_pointcloud(self.image_depth, self.depth_intrinsics).reshape(-1, 3)
        return self._pointcloud

    @property
    def pointcloud_texture(self):
        if self._pointcloud is None:
            self._pointcloud_texture = self._create_pointcloud_texture(self.image_color, self.color_intrinsics,
                                                                       self.color_to_depth_extrinsics).reshape(-1, 3)
        return self._pointcloud_texture

    @staticmethod
    def _align_color_to_depth(image_color, depth_intrinsics, color_intrinsics, color_to_depth_extrinsics):
        convert_3x3_to_4x4 = np.eye(3, 4)
        convert_4x4_to_3x3 = np.eye(4, 3)
        H = depth_intrinsics @ convert_3x3_to_4x4 @ color_to_depth_extrinsics @ convert_4x4_to_3x3 @ np.linalg.inv(
            color_intrinsics)
        H = H / H[2][2]
        return cv2.warpPerspective(image_color, H, (image_color.shape[1], image_color.shape[0]))

    @staticmethod
    def _align_depth_to_color(image_depth, depth_intrinsics, color_intrinsics, depth_to_color_extrinsics):
        convert_3x3_to_4x4 = np.eye(3, 4)
        convert_4x4_to_3x3 = np.eye(4, 3)
        H = color_intrinsics @ convert_3x3_to_4x4 @ depth_to_color_extrinsics @ convert_4x4_to_3x3 @ np.linalg.inv(
            depth_intrinsics)
        H = H / H[2][2]
        return cv2.warpPerspective(image_depth, H, (image_depth.shape[1], image_depth.shape[0]))

    def _convert_to_GRAY_8U(self, image, min_value=None, max_value=None, dynamic_range=None):
        if dynamic_range is None:
            dynamic_range = False

        # Perform grayscale conversion if needed
        image_gray = None
        if len(image.shape) == 3:
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif len(image.shape) == 2:
            image_gray = image
        else:
            raise ValueError('image was an unknown format.  Expected len(image.shape) = 2 or 3')

        # check if further processing is needed
        if dynamic_range is None and image_gray.dtype == np.uint8:
            return image_gray

        # Determine range of normalization
        if min_value is None or max_value is None:
            if dynamic_range:
                # assume full range based on min/max picel values
                if min_value is None:
                    min_value = np.amin(self.image_depth)
                if max_value is None:
                    max_value = np.amax(self.image_depth)
            else:
                # assume full range based on data type
                if np.issubdtype(image_gray.dtype, np.integer) or np.issubdtype(image_gray.dtype, np.signedinteger):
                    # int range is full range of value type
                    dtype_info = np.iinfo(image_gray.dtype)
                    if min_value is None:
                        min_value = dtype_info.min
                    if max_value is None:
                        max_value = dtype_info.max
                else:
                    # float range is 0.0-1.0
                    if min_value is None:
                        min_value = 0.0
                    if max_value is None:
                        max_value = 1.0
        # return cv2.normalize(image_gray, None, min_value, max_value, cv2.NORM_MINMAX, cv2.CV_8U)
        range_scale_factor = 255.0 / (max_value - min_value)
        return np.round((image_gray - min_value) * range_scale_factor).astype(np.uint8)

    def _colorize(self, image, colormap=None, min_value=None, max_value=None, dynamic_range=None):
        if colormap is None:
            colormap = cv2.COLORMAP_JET
        if dynamic_range is None:
            dynamic_range = True
        scaled = self._convert_to_GRAY_8U(image, min_value, max_value, dynamic_range)
        return cv2.applyColorMap(scaled, colormap)

    @staticmethod
    def _rs_colorize(depth_frame, colormap=None, min_value=None, max_value=None, visual_preset=None):
        global colorizer

        if colormap is None:
            colormap = 0
        if visual_preset is None:
            if min_value and max_value:
                visual_preset = 1
            elif min_value and max_value is None:
                visual_preset = 2
            elif min_value is None and max_value:
                visual_preset = 3
            else:
                visual_preset = 0

        # Apply options to colorizer
        if min_value:
            colorizer.set_option(rs.option.min_distance, min_value)
        if max_value:
            colorizer.set_option(rs.option.max_distance, max_value)
        # colorizer.set_option(rs.option.histogram_equalization_enabled, 0)
        colorizer.set_option(rs.option.visual_preset, visual_preset)  # 0=Dynamic, 1=Fixed, 2=Near, 3=Far
        colorizer.set_option(rs.option.color_scheme,
                             colormap)  # 0=Jet, 1=Classic, 2=WhiteToBlack, 3=BlackToWhite, 4=Bio, 5=Cold, 6=Warm, 7=Quantized, 8=Pattern

        return colorizer.colorize(depth_frame)

    @staticmethod
    def _convert_rs_intrinsics_to_opencv_matrix(rs_intrinsics):
        fx = rs_intrinsics.fx
        fy = rs_intrinsics.fy
        cx = rs_intrinsics.ppx
        cy = rs_intrinsics.ppy
        s = 0  # skew
        return np.array([[fx, s, cx],
                         [0.0, fy, cy],
                         [0.0, 0.0, 1.0]]).copy()

    @staticmethod
    def _convert_rs_extrinsics_to_opencv_matrix(rs_extrinsics):
        # https://www.brainvoyager.com/bv/doc/UsersGuide/CoordsAndTransforms/SpatialTransformationMatrices.html
        extrinsics_rotation = np.asanyarray(rs_extrinsics.rotation).reshape(3, 3).copy()
        extrinsics_translation = np.asanyarray(rs_extrinsics.translation).reshape(1, 3).copy()
        extrinsics_projection = np.concatenate((extrinsics_rotation, extrinsics_translation.T), axis=1)
        extrinsics_projection = np.concatenate((extrinsics_projection, np.array([[0, 0, 0, 1]])), axis=0)
        return extrinsics_projection  # , extrinsics_rotation, extrinsics_translation

    @staticmethod
    def _create_rs_pointcloud(pointcloud, depth_frame):
        points = pointcloud.calculate(depth_frame)
        vtx = np.asanyarray(points.get_vertices())
        return vtx.view(np.float32).reshape(vtx.shape + (-1,)).copy(), points

    @staticmethod
    def _create_rs_pointcloud_texture(pointcloud, color_frame, points):
        pointcloud.map_to(color_frame)
        tex = np.asanyarray(points.get_texture_coordinates())
        return tex.view(np.float32).reshape(tex.shape + (-1,)).copy()

    @staticmethod
    def _create_pointcloud(image_depth, depth_intrinsics):
        arr_depth = image_depth * 0.001
        cx = depth_intrinsics[0, 2]
        cy = depth_intrinsics[1, 2]
        fx = depth_intrinsics[0, 0]
        fy = depth_intrinsics[1, 1]
        arr = np.indices(arr_depth.shape).transpose(1, 2, 0).astype(arr_depth.dtype)
        arr = arr[..., ::-1]
        arr[:, :, 0] = (arr[:, :, 0] - cx) / fx * arr_depth
        arr[:, :, 1] = (arr[:, :, 1] - cy) / fy * arr_depth
        return np.dstack((arr, arr_depth))

    def _create_pointcloud_texture(self, image_color, color_intrinsics, color_to_depth_extrinsics):
        raise NotImplementedError()

    def __init__(self, frame_set, frame_id=None, timestamp=None, use_frame_blocking_methods=True, *args, **kwargs):
        # Frame must be cloned and released to allow the next acquisition to occur
        # Frame is not serializable, so the main goal is to capture the color/depth images and their intrinsics/extrinsics for downstream processing
        # It's unfortunate that the D435 doesn't provide a non-blocking way yto do this, as their built-in functions are efficient but limit performance to 15FPS
        # https://dev.intelrealsense.com/docs/projection-texture-mapping-and-occlusion-with-intel-realsense-depth-cameras
        self._use_frame_blocking_methods = use_frame_blocking_methods

        # Extract color image
        color_frame = frame_set.get_color_frame()
        self._image_color_frame = color_frame
        self._image_color = np.asanyarray(color_frame.get_data()).copy()

        # Extract depth image
        depth_frame = frame_set.get_depth_frame()
        self._image_depth_frame = depth_frame
        self._image_depth = np.asanyarray(depth_frame.get_data()).copy()

        self._pointcloud = None
        self._pointcloud_texture = None
        self._image_color_aligned = None
        self._image_depth_aligned = None
        self._image_depth_colorized = None
        self._image_depth_aligned_colorized = None
        self._image_depth_8U = None
        self._image_depth_8U_min = None
        self._image_depth_8U_max = None
        if frame_id:
            self._frame_id = frame_id
        else:
            self._frame_id = frame_set.frame_number
        if timestamp:
            self._timestamp = timestamp
        else:
            self._timestamp = frame_set.timestamp
        self.__dict__.update(kwargs)

        # Intrinsics map from camera space to world coordinate space
        # Extrinsics map within world space
        # ie, to get pixels from dept to color, intrinsic from depth to depth-in-world, extrinsics from depth-in-world to color-in-world, intrinsic from color-in-world to color

        # Get depth intrinsics
        depth_profile = frame_set.get_depth_frame().get_profile()
        depth_video_stream_profile = depth_profile.as_video_stream_profile()
        rs_depth_intrinsics = depth_video_stream_profile.get_intrinsics()
        self._depth_intrinsics_raw = rs_depth_intrinsics
        self._depth_intrinsics = self._convert_rs_intrinsics_to_opencv_matrix(rs_depth_intrinsics)

        # Get color intrinsics
        color_profile = frame_set.get_color_frame().get_profile()
        color_video_stream_profile = color_profile.as_video_stream_profile()
        rs_color_intrinsics = color_video_stream_profile.get_intrinsics()
        self._color_intrinsics_raw = rs_color_intrinsics
        self._color_intrinsics = self._convert_rs_intrinsics_to_opencv_matrix(rs_color_intrinsics)

        # Get depth_to_color extrinsics
        rs_depth_to_color_extrinsics = depth_video_stream_profile.get_extrinsics_to(color_video_stream_profile)
        self._depth_to_color_extrinsics_raw = rs_depth_to_color_extrinsics
        self._depth_to_color_extrinsics = self._convert_rs_extrinsics_to_opencv_matrix(rs_depth_to_color_extrinsics)

        # Get color_to_depth extrinsics
        rs_color_to_depth_extrinsics = color_video_stream_profile.get_extrinsics_to(depth_video_stream_profile)
        self._color_to_depth_extrinsics_raw = rs_color_to_depth_extrinsics
        self._color_to_depth_extrinsics = self._convert_rs_extrinsics_to_opencv_matrix(rs_color_to_depth_extrinsics)

        if use_frame_blocking_methods:
            # Align the color frame to depth frame and extract color image
            global align_to_depth
            if align_to_depth is None:
                align_to_depth = rs.align(rs.stream.depth)
            color_frame_aligned = align_to_depth.process(frame_set).get_color_frame()
            self._image_color_aligned_frame = color_frame_aligned
            self._image_color_aligned = np.asanyarray(color_frame_aligned.get_data()).copy()

            # Align the depth frame to color frame and extract depth image
            global align_to_color
            if align_to_color is None:
                align_to_color = rs.align(rs.stream.color)
            depth_frame_aligned = align_to_color.process(frame_set).get_depth_frame()
            self._image_depth_aligned_frame = depth_frame_aligned
            self._image_depth_aligned = np.asanyarray(depth_frame_aligned.get_data()).copy()

            # Colorize depth images
            global colorizer
            if colorizer is None:
                colorizer = rs.colorizer()
            depth_frame_colorized = self._rs_colorize(depth_frame)
            self._image_depth_colorized = np.asanyarray(depth_frame_colorized.get_data()).copy()
            depth_frame_aligned_colorized = self._rs_colorize(depth_frame_aligned)
            self._image_depth_aligned_colorized = np.asanyarray(depth_frame_aligned_colorized.get_data()).copy()


class Camera:
    """
    Interface for using Intel RealSense D435 
    Currently we can use camera to detect a green cube
    or detect Aruco markers for camera extrinsics calibration
    """
    def __init__(self, debug=False, imsize=(640, 480), device_id='036322250488'):
        self.debug = debug

        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(device_id)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        
        # Start streaming
        self.profile = self.pipeline.start(config)                
        self.imsize = imsize

        # For FPS calculation
        self.last_frame_time = None
        self.fps_alpha = 0.9  # Smoothing factor for FPS calculation
        self.current_fps = 0.0


    @staticmethod
    def get_bbox(frame, color='g'):
        cnt = Camera.get_contour(frame, color=color)
        if cnt is not None:
            return cv2.boundingRect(cnt)
        else:
            return None

    @staticmethod
    def get_contour(frame, color='g'):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        if color == 'g':
            mask = cv2.inRange(hsv, LOWER_GREEN, UPPER_GREEN)
        elif color == 'r':
            mask1 = cv2.inRange(hsv, LOWER_RED_1, UPPER_RED_1)
            mask2 = cv2.inRange(hsv, LOWER_RED_2, UPPER_RED_2)
            mask = mask1 + mask2
        # res = cv2.bitwise_and(frame, frame, mask=mask)    

        # Perform adaptive thresholding to handle varying lighting conditions
        blurred = cv2.GaussianBlur(mask, (5, 5), 0)
        _, adaptive_threshold = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Perform morphological operations to clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.erode(adaptive_threshold, kernel, iterations=2)
        mask = cv2.dilate(mask, kernel, iterations=2)

        # Find contours of the largest green/red object
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            cnt = max(contours, key = cv2.contourArea)
            if cv2.contourArea(cnt) > 50:
                return cnt
        return None

    def fetch_image(self, middle_crop_imsize=None, border_crops=(0,0,0,0), detect_object=False, color='g'):
        success, frames = self.pipeline.try_wait_for_frames()
        if not success:
            return None

        image_packet = IntelD455ImagePacket(frames, use_frame_blocking_methods=True)
        color_image = image_packet.image_color
        depth_image = image_packet.image_depth
        assert color_image.shape == (480, 640, 3), f"Color image shape {color_image.shape} is not (480, 640, 3)"

        if not middle_crop_imsize is None:
            assert middle_crop_imsize[0] <= color_image.shape[0] and middle_crop_imsize[1] <= color_image.shape[1], \
                "Crop image size is larger than the color image size"
            assert middle_crop_imsize[0]//2 <= depth_image.shape[0] and middle_crop_imsize[1]//2 <= depth_image.shape[1], \
                "Crop image size is larger than the depth image size (accounting for 2x resolution difference)"


            color_top = (color_image.shape[0] - middle_crop_imsize[0]) // 2
            color_left = (color_image.shape[1] - middle_crop_imsize[1]) // 2
            color_image = color_image[color_top:color_top + middle_crop_imsize[0], 
                                            color_left:color_left + middle_crop_imsize[1]]


            depth_crop_h, depth_crop_w = middle_crop_imsize[0] // 2, middle_crop_imsize[1] // 2
            depth_top = (depth_image.shape[0] - depth_crop_h) // 2
            depth_left = (depth_image.shape[1] - depth_crop_w) // 2
            depth_image = depth_image[depth_top:depth_top + depth_crop_h,
                                            depth_left:depth_left + depth_crop_w]
        
        if border_crops != (0,0,0,0):
            top, right, bottom, left = border_crops
            H, W = color_image.shape[:2]

            assert top >= 0 and bottom >= 0 and left >= 0 and right >= 0, \
                "Border crop values must be non-negative"
            assert top + bottom < H and left + right < W, \
                "Border crop boundaries exceed image dimensions"

            color_image = color_image[top:H-bottom, left:W-right]
            depth_image = depth_image[top//2:H//2-bottom//2, left//2:W//2-right//2]

        if self.imsize != (640, 480):
            assert self.imsize[0] <= color_image.shape[0] and self.imsize[1] <= color_image.shape[1], \
                "Resize image size is larger than the cropped color image size"
            assert self.imsize[0]//2 <= depth_image.shape[0] and self.imsize[1]//2 <= depth_image.shape[1], \
                "Resize image size is larger than the cropped depth image size (accounting for 2x resolution difference)"

            color_image = cv2.resize(color_image, self.imsize, interpolation=cv2.INTER_LINEAR)
            depth_image = cv2.resize(depth_image, self.imsize, interpolation=cv2.INTER_LINEAR)

        pos = None
        if detect_object:
            bbox = self.get_bbox(color_image, color=color)
            if bbox is not None:
                u = int((bbox[0] + bbox[2] / 2))
                v = int((bbox[1] + bbox[3] / 2))
                pos = np.array([u, v])

        if self.debug:
            # Calculate FPS
            current_time = time.time()
            if self.last_frame_time is not None:
                frame_time = current_time - self.last_frame_time
                if frame_time > 0:
                    fps = 1.0 / frame_time
                    # Apply exponential moving average for smoothing
                    self.current_fps = self.fps_alpha * self.current_fps + (1 - self.fps_alpha) * fps
            self.last_frame_time = current_time
            
            # Concatenate color and depth images
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_TURBO)
            depth_colormap = cv2.resize(depth_colormap, (color_image.shape[1], color_image.shape[0]))
            rgb_depth_concat = np.hstack((color_image, depth_colormap))
            
            # Draw FPS on the image
            fps_text = f"FPS: {self.current_fps:.1f}"
            cv2.putText(rgb_depth_concat, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("RGBD Image", rgb_depth_concat)
            cv2.waitKey(1)

        if detect_object:
            return color_image, depth_image, pos
        else:
            return color_image, depth_image


    def close(self):
        try:
            self.pipeline.stop()
        except RuntimeError:
            pass

    def flush(self):
        for _ in range(50):
            frames = self.pipeline.wait_for_frames()
    
    def __del__(self):
        self.close()


def debug_camera():
    cam = Camera(debug=True)

    depth_min = 0.11 #meter
    depth_max = 4.0 #meter
    try:
        while True:
            frames = cam.pipeline.wait_for_frames()
            image_packet = IntelD455ImagePacket(frames, use_frame_blocking_methods=True)
            bbox = Camera.get_bbox(image_packet.image_color, color='r')
            if bbox is not None:
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                cv2.rectangle(image_packet.image_color, p1, p2, (0,0,255), 2, 1)
                u = int((bbox[0] + bbox[2] / 2))
                v = int((bbox[1] + bbox[3] / 2))
                
                depth_scale = cam.profile.get_device().first_depth_sensor().get_depth_scale()
                depth_point_pixel = rs.rs2_project_color_pixel_to_depth_pixel(
                    image_packet.image_depth_frame.get_data(), depth_scale,
                    depth_min, depth_max,
                    image_packet.depth_intrinsics_raw, image_packet.color_intrinsics_raw, 
                    image_packet.color_to_depth_extrinsics_raw, image_packet.depth_to_color_extrinsics_raw, 
                    [u, v]
                )
                depth = image_packet.image_depth_frame.get_distance(int(depth_point_pixel[0]), int(depth_point_pixel[1]))
                
                # Debug displays in RGB image
                cv2.circle(image_packet.image_color, (u, v), 5, (255, 255, 255), -1)
                cv2.putText(image_packet.image_color, f"Depth: {depth:.3f}m", (u, v), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                # Debug displays in depth image
                cv2.circle(image_packet.image_depth_colorized, (int(depth_point_pixel[0]), int(depth_point_pixel[1])), 5, (255, 255, 255), -1)
                cv2.putText(image_packet.image_depth_colorized, f"Depth: {depth:.3f}m", (int(depth_point_pixel[0]), int(depth_point_pixel[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow("Depth Alignment Debug - RGB", image_packet.image_color)
            cv2.imshow("Depth Alignment Debug - Depth", image_packet.image_depth_colorized)
            cv2.waitKey(1)

    finally:
        cam.close()
        cv2.destroyAllWindows()



def debug_camera_depth():
    '''
    Method to visualize depth image superimposed on color image to see if there is an alignment issue.
    The following info for point under the mouse cursor is also printed on the window - depth, x, y, z, color image pixel, depth image pixel.
    '''
    cam = Camera(debug=False)
    
    # mouse callback function
    mouse_coords = (0, 0)
    def mouse_callback(event, x, y, flags, param):
        nonlocal mouse_coords
        if event == cv2.EVENT_MOUSEMOVE:
            mouse_coords = (x, y)

    cv2.namedWindow('Depth Alignment Debug', cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback('Depth Alignment Debug', mouse_callback)

    try:
        while True:
            # Wait for a coherent pair of frames: depth and color
            frames = cam.pipeline.wait_for_frames()

            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            if not depth_frame or not color_frame:
                continue

            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # Grab new intrinsics (may be changed by decimation)
            depth_scale = cam.profile.get_device().first_depth_sensor().get_depth_scale()
            depth_min = 0.11 # meter
            depth_max = 4.0 # meter

            depth_intrin = cam.profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
            color_intrin = cam.profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

            depth_to_color_extrin =  cam.profile.get_stream(rs.stream.depth).as_video_stream_profile().get_extrinsics_to(cam.profile.get_stream(rs.stream.color))
            color_to_depth_extrin =  cam.profile.get_stream(rs.stream.color).as_video_stream_profile().get_extrinsics_to(cam.profile.get_stream(rs.stream.depth))

            # Red bounding box on color image
            bbox = Camera.get_bbox(color_image, color='r')
            
            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_TURBO)
            
            # Resize depth colormap to match color image dimensions
            if depth_colormap.shape != color_image.shape:
                depth_colormap = cv2.resize(depth_colormap, (color_image.shape[1], color_image.shape[0]), interpolation=cv2.INTER_AREA)

            # Blend color and depth images
            display_image = cv2.addWeighted(color_image, 0.5, depth_colormap, 0.5, 0)

            # Get info at mouse cursor
            u_mouse, v_mouse = mouse_coords
            depth = depth_frame.get_distance(u_mouse, v_mouse)

            info_text = f"Mouse at:({u_mouse},{v_mouse})"
            if depth > 0:
                point_3d = rs.rs2_deproject_pixel_to_point(color_intrin, [u_mouse, v_mouse], depth)
                x, y, z = point_3d[0], point_3d[1], point_3d[2]
                info_text += f" Depth: {depth:.3f}m | 3D:({x:.3f},{y:.3f},{z:.3f})"
            else:
                info_text += " Depth: N/A | 3D: N/A"

            # Draw text on image
            cv2.putText(display_image, info_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Get info for red bbox
            if bbox is not None:
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                cv2.rectangle(display_image, p1, p2, (0,0,255), 2, 1)

                u_bbox, v_bbox = int((bbox[0] + bbox[2] / 2)), int((bbox[1] + bbox[3] / 2))
                depth_point_pixel = rs.rs2_project_color_pixel_to_depth_pixel(
                        depth_frame.get_data(), depth_scale,
                        depth_min, depth_max,
                        depth_intrin, color_intrin, depth_to_color_extrin, color_to_depth_extrin, [u_bbox, v_bbox])
                
                bbox_info_text = f"Red Box at:({u_bbox},{v_bbox})"
                if depth_point_pixel[0] >= 0 and depth_point_pixel[1] >=0:
                    print(f"[Red Box] | RGB imsize: {color_image.shape} | RGB image pixel: {u_bbox, v_bbox}")
                    print(f"[Red Box] | Depth imsize: {depth_image.shape} | Depth image pixel: {depth_point_pixel[0], depth_point_pixel[1]}")
                    d = depth_frame.get_distance(int(depth_point_pixel[0]), int(depth_point_pixel[1]))
                    if d > 0:
                        x, y, z = rs.rs2_deproject_pixel_to_point(depth_intrin, depth_point_pixel, d)
                        bbox_info_text += f" Depth: {d:.3f}m | 3D:({x:.3f},{y:.3f},{z:.3f})"
                    else:
                        bbox_info_text += " Depth: N/A | 3D: N/A"
                else:
                    bbox_info_text += " Depth: N/A | 3D: N/A"
                cv2.putText(display_image, bbox_info_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


            # Show images
            cv2.imshow('Depth Alignment Debug', display_image)
            
            key = cv2.waitKey(1)
            # if 'q' is pressed or window is closed
            if key & 0xFF == ord('q') or key == 27 or cv2.getWindowProperty('Depth Alignment Debug', cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        cam.close()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    debug_camera()
    # debug_camera_depth()

    # cam = Camera(debug=True, imsize=(128, 128))
    # for _ in range(1000):
    #     cam.fetch_image(middle_crop_imsize=(480, 480))
    # cam.close()