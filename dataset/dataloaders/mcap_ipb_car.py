# MIT License
#
# Copyright (c) 2023 Ignacio Vizzo, Tiziano Guadagnino, Benedikt Mersch, Cyrill
# Stachniss.
# 2025 Yue Pan
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import importlib
import os
import sys
import cv2
import numpy as np
import yaml
import glob
from typing import Dict, List, Optional, Tuple

try:
    from natsort import natsorted
except ImportError:
    print("natsort not installed: 'pip install natsort'")
    exit(1)

class McapIPBCarDataloader:
    def __init__(self, data_dir: str, topic: str, *_, **__):
        """Standalone .mcap dataloader with support for both lidar and camera data for IPB car dataset."""
        # Conditional imports to avoid injecting dependencies for non mcap users

        self.contains_image = True  # Now supports images
        self.load_img = True  # Default to loading images
        self.max_time_diff = 0.1  # Maximum time difference for synchronization (seconds)
        self.camera_names = ["front", "left", "right", "rear"]  # Default camera names
        self.min_lidar_radius_m = 0.5  # Minimum radius filter for LiDAR points (meters)

        try:
            self.make_reader = importlib.import_module("mcap.reader").make_reader
            self.read_ros2_messages = importlib.import_module("mcap_ros2.reader").read_ros2_messages
        except ModuleNotFoundError:
            print("mcap plugins not installed: 'pip install mcap-ros2-support'")
            exit(1)

        from utils.point_cloud2 import read_point_cloud

        # Handle both single file and directory inputs
        if os.path.isfile(data_dir):
            self.mcap_files = [data_dir]
            self.data_dir = os.path.dirname(data_dir)
        elif os.path.isdir(data_dir):
            self.mcap_files = natsorted([
                os.path.join(data_dir, f) for f in os.listdir(data_dir)
                if f.endswith('.mcap')
            ])
            assert len(self.mcap_files) > 0, f"No .mcap files found in directory: {data_dir}"
            if len(self.mcap_files) > 1:
                print("Reading multiple .mcap files in directory:")
                print("\n".join([os.path.basename(path) for path in self.mcap_files]))
            self.data_dir = data_dir
        else:
            raise ValueError(f"Input path {data_dir} is neither a file nor directory")

        # Initialize with first file
        self.current_file_idx = 0
        self.sequence_id = os.path.basename(self.mcap_files[0]).split(".")[0]
        self.bag_total_scan = 0
        
        # Initialize camera calibration data (load only once)
        self.camera_calibrations = {}
        self.K_mats = {}
        self.dist_coeffs = {}
        self.T_c_l_mats = {}
        self.cam_widths = {}
        self.cam_heights = {}
        
        self.cam_valid_v_ratios_minmax = {
            "front": [0.44, 1.0],
            "left": [0.0, 1.0],
            "right": [0.0, 1.0],
            "rear": [0.47, 1.0]
        }

        self._load_camera_calibrations()

        self.lidar_topic = "/lidar/horizontal/points" # default lidar topic
        
        self._initialize_current_file(self.mcap_files[0], topic, is_first_bag=True)
        
        self.read_point_cloud = read_point_cloud
        
        # Calculate total number of scans across all files
        self.total_scans = self._calculate_total_scans(self.lidar_topic)

    def _initialize_current_file(self, mcap_file: str, lidar_topic: str, is_first_bag: bool = False):
        """Initialize readers for a new mcap file."""
        if hasattr(self, 'bag'):
            del self.bag
        self.bag = self.make_reader(open(mcap_file, "rb"))
        self.summary = self.bag.get_summary()
        if is_first_bag:
            self.lidar_topic = self.check_lidar_topic(lidar_topic)
        self.n_scans = self._get_n_scans()
        self.lidar_msgs = self.read_ros2_messages(mcap_file, topics=self.lidar_topic)
        self.current_scan = 0
        
        # Initialize image topics and readers
        if is_first_bag:
            self._initialize_image_topics()
            self.image_msg_buffers = {}  # Buffer for each camera's messages
        self._initialize_image_readers(mcap_file, is_first_bag)

    def _initialize_image_topics(self):
        """Find and initialize image topics from the MCAP file."""
        # Extract schema ids for image messages
        image_schema_ids = []
        for schema_id, schema in self.summary.schemas.items():
            if schema.name in ["sensor_msgs/msg/Image", "sensor_msgs/msg/CompressedImage"]:
                image_schema_ids.append(schema_id)
        
        # Find channels that use image schemas
        self.image_topics = []
        self.image_schemas = {}
        for channel_id, channel in self.summary.channels.items():
            if channel.schema_id in image_schema_ids:
                self.image_topics.append(channel.topic)
                self.image_schemas[channel.topic] = channel.schema_id
        
        print(f"Found {len(self.image_topics)} image topics: {self.image_topics}")

    def _initialize_image_readers(self, mcap_file: str, is_first_bag: bool = False):
        """Initialize image message readers for each topic."""
        self.image_readers = {}
        for topic in self.image_topics:
            try:
                reader = self.read_ros2_messages(mcap_file, topics=topic)
                self.image_readers[topic] = reader
                if is_first_bag:
                    self.image_msg_buffers[topic] = []
            except Exception as e:
                print(f"Warning: Could not initialize reader for image topic {topic}: {e}")
                continue

    def _get_timestamp_from_message(self, msg) -> float:
        """Extract timestamp from ROS2 message."""
        if hasattr(msg, 'header') and hasattr(msg.header, 'stamp'):
            # Convert ROS time to seconds
            return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        return 0.0

    def _decode_image_message(self, msg) -> Optional[np.ndarray]:
        """Decode ROS2 image message to numpy array."""
        try:
            if hasattr(msg, 'data'):
                # Regular Image message
                height = msg.height
                width = msg.width
                encoding = msg.encoding
                
                # Convert encoding to OpenCV format
                if encoding == 'rgb8':
                    channels = 3
                elif encoding == 'bgr8':
                    channels = 3
                elif encoding == 'mono8':
                    channels = 1
                elif encoding == 'bayer_rggb8':
                    # Bayer RGGB pattern - convert to RGB
                    channels = 1  # Bayer is single channel
                else:
                    print(f"Warning: Unsupported image encoding: {encoding}")
                    return None
                
                # Reshape data to image
                img_data = np.frombuffer(msg.data, dtype=np.uint8)
                
                if encoding == 'bayer_rggb8':
                    # Reshape to 2D array for Bayer decoding
                    # print("Bayer RGGB8 image found")
                    img = img_data.reshape(height, width)
                    img = cv2.cvtColor(img, cv2.COLOR_BayerRG2BGR)
                else:
                    img = img_data.reshape(height, width, channels)
                    # Convert BGR to RGB if needed
                    if encoding == 'bgr8':
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                return img
                
            elif hasattr(msg, 'format') and hasattr(msg, 'data'):
                # CompressedImage message
                format_str = msg.format.lower()
                img_data = np.frombuffer(msg.data, dtype=np.uint8)
                
                if 'jpeg' in format_str or 'jpg' in format_str:
                    img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
                    if img is not None:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        return img
                elif 'png' in format_str:
                    img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
                    if img is not None:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        return img
                else:
                    print(f"Warning: Unsupported compressed image format: {format_str}")
                    return None
                    
        except Exception as e:
            print(f"Error decoding image message: {e}")
            return None
        
        return None

    def _undistort_image(self, img: np.ndarray, cam_name: str) -> np.ndarray:
        """Apply camera undistortion using loaded calibration data."""
        if cam_name not in self.K_mats or cam_name not in self.dist_coeffs:
            print(f"Warning: No calibration data for camera {cam_name}, returning original image")
            return img
        
        K = self.K_mats[cam_name]
        dist_coeffs = self.dist_coeffs[cam_name]
        
        # Check if distortion coefficients are all zero (no distortion)
        if np.allclose(dist_coeffs, 0):
            return img
        
        try:
            # Apply undistortion
            undistorted_img = cv2.undistort(img, K, dist_coeffs)
            return undistorted_img
        except Exception as e:
            print(f"Error applying undistortion for camera {cam_name}: {e}")
            return img

    def _find_closest_image_msg(self, target_timestamp: float, topic: str, max_time_diff: float = 0.2) -> Optional[np.ndarray]:
        """Find the closest image to the target timestamp within the time window."""

        # most of the time, lidar late than cam by about 0.05s

        if topic not in self.image_msg_buffers:
            return None
            
        buffer = self.image_msg_buffers[topic]
        
        # print("lidar timestamp: ", target_timestamp)

        # Fill buffer if needed
        while len(buffer) < 5:  # Keep at least 5 messages in buffer
            try:
                msg = next(self.image_readers[topic]).ros_msg
                timestamp = self._get_timestamp_from_message(msg)
                buffer.append((timestamp, msg))
            except StopIteration:
                break
        
        if not buffer:
            return None
        
        # Find closest timestamp
        closest_msg = None
        min_diff = float('inf')
        
        for timestamp, msg in buffer:
            time_diff = abs(timestamp - target_timestamp)
            if time_diff < min_diff:
                min_diff = time_diff
                closest_msg = msg
                closest_timestamp = timestamp
            
        # print(f"associated {topic} image timestamp: {closest_timestamp}")
        
        # Check if within acceptable time window
        if min_diff <= max_time_diff and closest_msg is not None:
            # Remove older messages from buffer
            self.image_msg_buffers[topic] = [(t, m) for t, m in buffer if t >= target_timestamp - max_time_diff]
            
            return closest_msg
        
        return None

    def __del__(self):
        if hasattr(self, "bag"):
            del self.bag

    def _calculate_total_scans(self, topic: str) -> int:
        """Calculate total number of scans across all mcap files."""
        total = 0
        for file in self.mcap_files:
            bag = self.make_reader(open(file, "rb"))
            summary = bag.get_summary()
            total += sum(
                count
                for (id, count) in summary.statistics.channel_message_counts.items()
                if summary.channels[id].topic == topic
            )
            del bag  # Clean up the reader
        return total

    def __getitem__(self, idx):

        while idx >= self.bag_total_scan:

            # print("current scan: ", self.bag_total_scan)

            # Check if we need to move to next file
            while self.current_scan >= self.n_scans:
                self.current_file_idx += 1
                if self.current_file_idx >= len(self.mcap_files):
                    raise IndexError("Index out of range")
                self._initialize_current_file(self.mcap_files[self.current_file_idx], self.lidar_topic)

            # Get lidar data
            lidar_msg = next(self.lidar_msgs).ros_msg
            
            self.current_scan += 1
            self.bag_total_scan += 1

            # Get synchronized image data using timestamp-based synchronization
            if self.image_topics:
                img_msg_dict = {}
                lidar_timestamp = self._get_timestamp_from_message(lidar_msg)
                
                for topic in self.image_topics:
                    if topic in self.image_readers:
                        img_msg = self._find_closest_image_msg(lidar_timestamp, topic)
                        # print(f"img: {img}")
                        if img_msg is not None:
                            # Use topic name as camera identifier
                            cam_name = topic.split('/')[2] if '/' in topic else topic
                            img_msg_dict[cam_name] = img_msg
                        else:
                            print(f"Warning: No temporal consistent image found for topic {topic}")

        # assign frame data
        frame_data = {}
        
        points, point_ts = self.read_point_cloud(lidar_msg)
        
        # Apply minimum radius filtering
        valid_mask = ~np.all(np.abs(points[:,:3]) < self.min_lidar_radius_m, axis=1)
        points = points[valid_mask]
        if point_ts is not None:
            point_ts = point_ts[valid_mask]

        if self.image_topics:
            if img_msg_dict:

                img_dict = {}

                for cam_name, img_msg in img_msg_dict.items():
                    # Decode and undistort image
                    img = self._decode_image_message(img_msg)
                    if img is not None:
                        img = self._undistort_image(img, cam_name)
                        img_dict[cam_name] = img
                
                frame_data["img"] = img_dict

            points_rgb = -1.0 * np.ones_like(points) # N,4, last channel for the mask # set to invalid (indicated by negative value) at first
            points = np.hstack((points[:,:3], points_rgb[:,:3])) # N, 6  

        frame_data["points"] = points
        frame_data["point_ts"] = point_ts
        
        return frame_data

    def __len__(self):
        return self.total_scans

    def _get_n_scans(self) -> int:
        return sum(
            count
            for (id, count) in self.summary.statistics.channel_message_counts.items()
            if self.summary.channels[id].topic == self.lidar_topic
        )

    def check_lidar_topic(self, topic: str) -> str:
        # Extract schema id from the .mcap file that encodes the PointCloud2 msg
        schema_id = [
            schema.id
            for schema in self.summary.schemas.values()
            if schema.name == "sensor_msgs/msg/PointCloud2"
        ][0]

        point_cloud_topics = [
            channel.topic
            for channel in self.summary.channels.values()
            if channel.schema_id == schema_id
        ]

        def print_available_topics_and_exit():
            print("Select from the following topics:")
            print(50 * "-")
            for t in point_cloud_topics:
                print(f"{t}")
            print(50 * "-")
            sys.exit(1)

        # Default topic for IPB car dataset
        default_lidar_topic = self.lidar_topic

        if topic and topic in point_cloud_topics:
            print(f"Using LiDAR topic: {topic}")
            return topic
        # when user specified the topic check that exists
        if topic and topic not in point_cloud_topics:
            print(
                f'[ERROR] Dataset does not containg any msg with the topic name "{topic}". '
                "Specify the correct topic name by python pin_slam.py path/to/config/file.yaml mcap your/topic ... ..."
            )
            print_available_topics_and_exit()
        
        # If no topic specified, try to use default topic
        if not topic and default_lidar_topic in point_cloud_topics:
            print(f"Using default LiDAR topic: {default_lidar_topic}")
            return default_lidar_topic

        if len(point_cloud_topics) == 0:
            print("[ERROR] Your dataset does not contain any sensor_msgs/msg/PointCloud2 topic")
            sys.exit(1)
        if len(point_cloud_topics) == 1:
            return point_cloud_topics[0]
        
        # This should never be reached, but return empty string to satisfy type checker
        return ""

    def _load_camera_calibrations(self):
        """Load camera calibration data from calibration folder."""
        # Look for calibration folder in multiple locations
        calib_dirs = []
        
        # 1. Parallel to data directory
        calib_dir_parallel = os.path.join(os.path.dirname(self.data_dir), "calibration")
        calib_dirs.append(calib_dir_parallel)
        
        # 2. In parent folder of data directory
        calib_dir_parent = os.path.join(os.path.dirname(os.path.dirname(self.data_dir)), "calibration")
        calib_dirs.append(calib_dir_parent)
        
        # Search for calibration folder
        calib_dir = None
        for dir_path in calib_dirs:
            if os.path.exists(dir_path):
                calib_dir = dir_path
                print(f"Found calibration directory: {calib_dir}")
                break
        
        if calib_dir is None:
            print(f"Warning: Calibration directory not found in any of these locations:")
            for dir_path in calib_dirs:
                print(f"  - {dir_path}")
            print("Using default calibration values")
            self._setup_default_calibrations()
            return
        
        # Look for calibration files
        calib_files = []
        for ext in ['*.yaml', '*.yml', '*.json']:
            calib_files.extend(glob.glob(os.path.join(calib_dir, ext)))
        
        if not calib_files:
            print(f"Warning: No calibration files found in {calib_dir}")
            self._setup_default_calibrations()
            return
        
        # Use the first calibration file found
        calib_file = calib_files[0]
        print(f"Loading calibration from: {calib_file}")
        
        try:
            self._read_calib_file(calib_file)
        except Exception as e:
            print(f"Error reading calibration file: {e}")
            print("Using default calibration values")
            self._setup_default_calibrations()

    def _setup_default_calibrations(self):
        """Setup default calibration values when no calibration file is available."""
        # Default image dimensions (adjust as needed)
        default_width, default_height = 2048, 1536
        
        # Default camera intrinsics (adjust as needed)
        default_fx = default_width / 2.0
        default_fy = default_height / 2.0
        default_cx = default_width / 2.0
        default_cy = default_height / 2.0
        
        # Default K matrix
        default_K = np.array([
            [default_fx, 0, default_cx],
            [0, default_fy, default_cy],
            [0, 0, 1]
        ])
        
        # Default distortion coefficients (no distortion)
        default_dist = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        
        # Default extrinsics (identity matrix - adjust based on your setup)
        default_T_c_l = np.eye(4)
        
        # Setup for each camera
        for cam_name in self.camera_names:
            self.K_mats[cam_name] = default_K.copy()
            self.dist_coeffs[cam_name] = default_dist.copy()
            self.T_c_l_mats[cam_name] = default_T_c_l.copy()
            self.cam_widths[cam_name] = default_width
            self.cam_heights[cam_name] = default_height
        
        print("Using default calibration values")

    def _read_calib_file(self, calib_file_path: str):
        """Read calibration file in YAML format similar to ipb_car.py."""
        try:
            with open(calib_file_path, 'r') as file:
                calib_dict = yaml.safe_load(file)

            # load lidar calibration
            lidar_h_calib = calib_dict["lidarhorizontalpoints"]
            lidar_v_calib = calib_dict["lidarverticalpoints"]
            T_cf_lh = np.array(lidar_h_calib["extrinsics"])
            T_cf_lv = np.array(lidar_v_calib["extrinsics"])
            self.T_lv_lh = np.linalg.inv(T_cf_lv) @ T_cf_lh
            
            # Extract camera calibrations
            for cam_name in self.camera_names:
                # Try different possible key formats
                cam_key = None
                for key_format in [
                    f"camera{cam_name}image_raw",
                    f"camera_{cam_name}",
                    f"camera{cam_name}",
                    cam_name
                ]:
                    if key_format in calib_dict:
                        cam_key = key_format
                        break
                
                if cam_key is None:
                    print(f"Warning: No calibration found for camera {cam_name}, using defaults")
                    continue
                
                camera_calib = calib_dict[cam_key]
                
                # Extract intrinsics
                if "K" in camera_calib:
                    self.K_mats[cam_name] = np.array(camera_calib["K"])
                elif "intrinsics" in camera_calib:
                    self.K_mats[cam_name] = np.array(camera_calib["intrinsics"])
                else:
                    print(f"Warning: No intrinsics found for camera {cam_name}")
                    continue
                
                # Extract distortion coefficients
                if "distortion_coeff" in camera_calib:
                    self.dist_coeffs[cam_name] = np.array(camera_calib["distortion_coeff"])
                elif "dist_coeffs" in camera_calib:
                    self.dist_coeffs[cam_name] = np.array(camera_calib["dist_coeffs"])
                else:
                    # Default to no distortion
                    self.dist_coeffs[cam_name] = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
                
                # Extract extrinsics (camera to lidar transformation)
                if "extrinsics" in camera_calib:
                    T_c_l = np.array(camera_calib["extrinsics"])
                    # If it's camera to lidar, we need to invert it
                    if T_c_l.shape == (4, 4):
                        self.T_c_l_mats[cam_name] = np.linalg.inv(T_c_l) @ T_cf_lh 
                    else:
                        print(f"Warning: Invalid extrinsics format for camera {cam_name}")
                        self.T_c_l_mats[cam_name] = np.eye(4)
                else:
                    print(f"Warning: No extrinsics found for camera {cam_name}, using identity")
                    self.T_c_l_mats[cam_name] = np.eye(4)
                
                # Extract image dimensions
                if "width" in camera_calib and "height" in camera_calib:
                    self.cam_widths[cam_name] = int(camera_calib["width"])
                    self.cam_heights[cam_name] = int(camera_calib["height"])
                else:
                    self.cam_widths[cam_name] = 2048
                    self.cam_heights[cam_name] = 1536
                
                # print(f"Loaded calibration for camera {cam_name}")
                # print(f"  Image size: {self.cam_widths[cam_name]}x{self.cam_heights[cam_name]}")
                # print(f"  K matrix shape: {self.K_mats[cam_name].shape}")
                # print(f"  Distortion coeffs: {self.dist_coeffs[cam_name]}")
                # print(f"  Extrinsics: {self.T_c_l_mats[cam_name]}")
        
        except Exception as e:
            print(f"Error parsing calibration file: {e}")
            raise
