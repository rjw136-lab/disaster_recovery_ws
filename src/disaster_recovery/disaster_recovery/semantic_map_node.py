#!/usr/bin/env python3
"""
=================================================================================================
Project: Disaster Recovery TurtleBot3 Burger System
Node: semantic_map_node.py
Author: Kyungpook National University / Jong Woo Ro

Description:
  Host PC 2D Semantic SLAM Mapping Node (Layer 3 Supervision Extension).
  Fuses 2D LiDAR (/scan), Monocular RGB YOLO26n detections (/detection/objects),
  Cartographer 2D OccupancyGrid (/map), MQ-2 Gas Sensor (/gas/level), and
  Faster-Whisper Audio Distress Alerts (/disaster/audio_alert) into a persistent
  2D Disaster Semantic Map without modifying or burdening the Raspberry Pi 4
  onboard survival/backtracking control loop.

Reference:
  Lee & Kim (2025), "Development and Implementation of a SLAM-Based Real-Time
  Environment Mapping Algorithm Including Obstacle Information", J. Korea Robotics Soc. 20(2).
=================================================================================================
"""

from collections import deque
import json
import os
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time

from builtin_interfaces.msg import Duration as MsgDuration
from nav_msgs.msg import OccupancyGrid, Odometry
from sensor_msgs.msg import CameraInfo, Image, LaserScan
from std_msgs.msg import ColorRGBA, Float32, String
from std_srvs.srv import Trigger
from visualization_msgs.msg import Marker, MarkerArray

import tf2_ros

from disaster_recovery.semantic_mapper_core import (
    CameraIntrinsics,
    DEFAULT_CLASS_PALETTE,
    DepthEstimate,
    DepthEstimator,
    PermanentMarker,
    SemanticMarkerManager,
    SemanticObservation,
    create_depth_estimator,
    default_t_base_cam,
    default_t_cam_scan,
    find_closest_timed_item,
    pose_to_matrix,
    save_semantic_map_bundle,
    transform_point,
    unrotate_bbox_xyxy,
)


def stamp_to_sec(stamp_msg) -> float:
    """Convert ROS 2 builtin_interfaces/Time or dict {'sec': ..., 'nanosec': ...} to float seconds."""
    if isinstance(stamp_msg, dict):
        return float(stamp_msg.get('sec', 0)) + float(stamp_msg.get('nanosec', 0)) * 1e-9
    sec = float(getattr(stamp_msg, 'sec', 0))
    nanosec = float(getattr(stamp_msg, 'nanosec', 0))
    return sec + nanosec * 1e-9


def transform_stamped_to_matrix(tf_msg) -> np.ndarray:
    """Convert geometry_msgs/TransformStamped into a 4x4 homogeneous matrix (target <- source)."""
    t = tf_msg.transform.translation
    q = tf_msg.transform.rotation
    return pose_to_matrix(
        tx=float(t.x),
        ty=float(t.y),
        tz=float(t.z),
        qx=float(q.x),
        qy=float(q.y),
        qz=float(q.z),
        qw=float(q.w),
    )


class SemanticMapNode(Node):
    """Real-Time 2D Disaster Semantic Mapping ROS 2 Node."""

    def __init__(self):
        super().__init__(
            'semantic_map_node',
            allow_undeclared_parameters=True,
            automatically_declare_parameters_from_overrides=True,
        )

        # -------------------------------------------------------------------------
        # Parameters
        # -------------------------------------------------------------------------
        self.cluster_radius_m = float(self._param('cluster_radius_m', 0.45))
        self.merge_radius_m = float(self._param('merge_radius_m', 0.60))
        self.min_hits = int(self._param('min_hits', 5))
        self.min_span_sec = float(self._param('min_span_sec', 1.0))
        self.temp_ttl_sec = float(self._param('temp_ttl_sec', 10.0))

        self.depth_source = str(self._param('depth_source', 'lidar_fusion'))
        self.enable_ground_fallback = bool(self._param('enable_ground_fallback', True))
        self.min_lidar_range_m = float(self._param('min_lidar_range_m', 0.12))
        self.max_lidar_range_m = float(self._param('max_lidar_range_m', 3.5))
        self.foreground_cluster_margin_m = float(self._param('foreground_cluster_margin_m', 0.25))
        self.bbox_horizontal_inner_ratio = float(self._param('bbox_horizontal_inner_ratio', 0.70))

        self.scan_buffer_size = max(1, int(self._param('scan_buffer_size', 20)))
        self.max_scan_sync_dt_sec = float(self._param('max_scan_sync_dt_sec', 0.15))
        self.max_angular_vel_rad_s = float(self._param('max_angular_vel_rad_s', 0.5))

        self.map_frame = str(self._param('map_frame', 'map'))
        self.odom_frame = str(self._param('odom_frame', 'odom'))
        self.base_frame = str(self._param('base_frame', 'base_link'))
        self.scan_frame = str(self._param('scan_frame', 'base_scan'))
        self.camera_optical_frame = str(self._param('camera_optical_frame', 'camera_rgb_optical_frame'))
        self.use_odom_fallback = bool(self._param('use_odom_fallback', True))
        self.map_qos_durability = str(self._param('map_qos_durability', 'volatile')).lower()

        self.gas_threshold = float(self._param('gas_threshold', 0.40))
        self.event_dedup_radius_m = float(self._param('event_dedup_radius_m', 0.60))
        self.ch1_obstacle_radius_m = float(self._param('ch1_obstacle_radius_m', 0.22))
        self.ch2_hazard_radius_m = float(self._param('ch2_hazard_radius_m', 0.55))

        self.camera_width = int(self._param('camera_width', 640))
        self.camera_height = int(self._param('camera_height', 480))
        self.camera_hfov_rad = float(self._param('camera_hfov_rad', 1.085595))
        self.camera_fx = float(self._param('camera_fx', 530.1254))
        self.camera_fy = float(self._param('camera_fy', 529.8971))
        self.camera_cx = float(self._param('camera_cx', 318.4521))
        self.camera_cy = float(self._param('camera_cy', 239.1124))

        self.output_dir = str(self._param('output_dir', 'semantic_maps'))
        self.save_on_shutdown = bool(self._param('save_on_shutdown', True))
        self.marker_publish_rate_hz = max(0.5, float(self._param('marker_publish_rate_hz', 2.0)))

        self.class_palette = self._load_class_palette()

        # -------------------------------------------------------------------------
        # Core State & Semantic Mapper Engine
        # -------------------------------------------------------------------------
        self.intrinsics = CameraIntrinsics(
            fx=self.camera_fx,
            fy=self.camera_fy,
            cx=self.camera_cx,
            cy=self.camera_cy,
            width=self.camera_width,
            height=self.camera_height,
        )
        self.camera_info_received = False

        self.depth_estimator: DepthEstimator = create_depth_estimator(
            depth_source=self.depth_source,
            use_ground_fallback=self.enable_ground_fallback,
            min_lidar_range_m=self.min_lidar_range_m,
            max_lidar_range_m=self.max_lidar_range_m,
            center_ratio=self.bbox_horizontal_inner_ratio,
            foreground_margin_m=self.foreground_cluster_margin_m,
        )
        self.marker_manager = SemanticMarkerManager(
            cluster_radius_m=self.cluster_radius_m,
            merge_radius_m=self.merge_radius_m,
            min_hits=self.min_hits,
            min_span_sec=self.min_span_sec,
            temp_ttl_sec=self.temp_ttl_sec,
            max_angular_vel_rad_s=self.max_angular_vel_rad_s,
        )

        self.scan_buffer: Deque[Tuple[float, LaserScan]] = deque(maxlen=self.scan_buffer_size)
        self.latest_depth_image: Optional[np.ndarray] = None
        self.latest_map: Optional[OccupancyGrid] = None
        self.current_angular_vel: float = 0.0
        self.latest_odom_xy: Tuple[float, float] = (0.0, 0.0)
        self.active_target_frame: str = self.map_frame
        self._saved_on_shutdown: bool = False

        # -------------------------------------------------------------------------
        # TF2 Buffer & Listener
        # -------------------------------------------------------------------------
        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=15.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # -------------------------------------------------------------------------
        # QoS Profiles & Subscriptions
        # -------------------------------------------------------------------------
        qos_sensor = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        qos_reliable = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        map_durability = (
            DurabilityPolicy.TRANSIENT_LOCAL
            if self.map_qos_durability == 'transient_local'
            else DurabilityPolicy.VOLATILE
        )
        qos_map = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=map_durability,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.sub_detections = self.create_subscription(
            String, '/detection/objects', self.cb_detection_objects, qos_reliable
        )
        self.sub_scan = self.create_subscription(
            LaserScan, '/scan', self.cb_scan, qos_sensor
        )
        self.sub_camera_info = self.create_subscription(
            CameraInfo, '/camera/camera_info', self.cb_camera_info, qos_sensor
        )
        self.sub_map = self.create_subscription(
            OccupancyGrid, '/map', self.cb_map, qos_map
        )
        self.sub_odom = self.create_subscription(
            Odometry, '/odom', self.cb_odom, qos_sensor
        )
        self.sub_gas = self.create_subscription(
            Float32, '/gas/level', self.cb_gas_level, qos_reliable
        )
        self.sub_audio_alert = self.create_subscription(
            String, '/disaster/audio_alert', self.cb_audio_alert, qos_reliable
        )
        if self.depth_source == 'rgbd_depth':
            self.sub_depth_img = self.create_subscription(
                Image, '/camera/depth/image_raw', self.cb_depth_image, qos_sensor
            )

        # -------------------------------------------------------------------------
        # Publishers & Services
        # -------------------------------------------------------------------------
        self.pub_markers = self.create_publisher(MarkerArray, '/semantic_map/markers', 10)
        self.pub_cached_objects = self.create_publisher(String, '/semantic_map/cached_objects', 10)
        self.srv_save_map = self.create_service(Trigger, '/semantic_map/save_map', self.handle_save_map)

        self.timer_markers = self.create_timer(
            1.0 / self.marker_publish_rate_hz, self.publish_markers
        )

        self.get_logger().info(
            f"SemanticMapNode initialized (depth_source={self.depth_source}, "
            f"map_qos={self.map_qos_durability}, cluster_r={self.cluster_radius_m}m, "
            f"merge_r={self.merge_radius_m}m, min_hits={self.min_hits}, output_dir={self.output_dir})"
        )

    # -----------------------------------------------------------------------------
    # Parameter Helpers
    # -----------------------------------------------------------------------------
    def _param(self, name: str, default: Any) -> Any:
        if self.has_parameter(name):
            val = self.get_parameter(name).value
            if val is not None:
                return val
        return self.declare_parameter(name, default).value

    def _load_class_palette(self) -> Dict[str, Tuple[int, int, int]]:
        palette: Dict[str, Tuple[int, int, int]] = dict(DEFAULT_CLASS_PALETTE)
        palette.setdefault('default', (100, 200, 0))
        try:
            prefix_params = self.get_parameters_by_prefix('class_palette')
            for key, param in prefix_params.items():
                val = param.value
                if isinstance(val, (list, tuple)) and len(val) >= 3:
                    palette[str(key)] = (int(val[0]), int(val[1]), int(val[2]))
        except Exception:
            pass
        return palette

    # -----------------------------------------------------------------------------
    # Sensor Callbacks
    # -----------------------------------------------------------------------------
    def cb_scan(self, msg: LaserScan):
        scan_t = stamp_to_sec(msg.header.stamp)
        if scan_t <= 0.0:
            scan_t = self.get_clock().now().nanoseconds * 1e-9
        self.scan_buffer.append((scan_t, msg))

    def cb_camera_info(self, msg: CameraInfo):
        if len(msg.k) >= 9 and msg.k[0] > 0.0 and msg.k[4] > 0.0:
            width = int(msg.width) if msg.width > 0 else self.camera_width
            height = int(msg.height) if msg.height > 0 else self.camera_height
            self.intrinsics = CameraIntrinsics.from_camera_matrix(
                msg.k, width=width, height=height
            )
            self.camera_info_received = True

    def cb_map(self, msg: OccupancyGrid):
        self.latest_map = msg

    def cb_odom(self, msg: Odometry):
        self.current_angular_vel = float(msg.twist.twist.angular.z)
        self.latest_odom_xy = (
            float(msg.pose.pose.position.x),
            float(msg.pose.pose.position.y),
        )

    def cb_depth_image(self, msg: Image):
        try:
            h, w = int(msg.height), int(msg.width)
            if msg.encoding in ('32FC1', 'float32'):
                self.latest_depth_image = np.frombuffer(msg.data, dtype=np.float32).reshape((h, w)).copy()
            elif msg.encoding in ('16UC1', 'mono16'):
                self.latest_depth_image = np.frombuffer(msg.data, dtype=np.uint16).reshape((h, w)).copy()
        except Exception as e:
            self.get_logger().warn(f"Depth image parse warning: {e}", throttle_duration_sec=5.0)

    def cb_gas_level(self, msg: Float32):
        level = float(msg.data)
        if level < self.gas_threshold:
            return
        now_sec = self.get_clock().now().nanoseconds * 1e-9
        rx, ry, rz = self._lookup_robot_position()
        self.marker_manager.register_event(
            class_name='GAS_HAZARD',
            x=rx,
            y=ry,
            z=rz,
            conf=min(1.0, max(0.5, level)),
            stamp_sec=now_sec,
            merge_radius_m=self.event_dedup_radius_m,
        )
        self.publish_markers()

    def cb_audio_alert(self, msg: String):
        if not msg.data:
            return
        now_sec = self.get_clock().now().nanoseconds * 1e-9
        rx, ry, rz = self._lookup_robot_position()
        self.marker_manager.register_event(
            class_name='AUDIO_DISTRESS',
            x=rx,
            y=ry,
            z=rz,
            conf=1.0,
            stamp_sec=now_sec,
            merge_radius_m=self.event_dedup_radius_m,
        )
        self.publish_markers()

    # -----------------------------------------------------------------------------
    # Detection Objects Callback & 3D -> Map Projection Pipeline
    # -----------------------------------------------------------------------------
    def cb_detection_objects(self, msg: String):
        # Step 2 Motion Gating: Suspend permanent marker registration during fast turns (|omega| > 0.5 rad/s)
        if self.marker_manager.is_motion_gated(self.current_angular_vel):
            self.get_logger().debug(
                f"Motion gating active: |omega|={abs(self.current_angular_vel):.2f} > {self.max_angular_vel_rad_s} rad/s"
            )
            return

        try:
            payload = json.loads(msg.data)
        except Exception as e:
            self.get_logger().warn(f"Invalid /detection/objects JSON: {e}", throttle_duration_sec=5.0)
            return

        detections = payload.get('detections', [])
        if not detections:
            return

        stamp_dict = payload.get('stamp', {'sec': 0, 'nanosec': 0})
        det_stamp_sec = stamp_to_sec(stamp_dict)
        if det_stamp_sec <= 0.0:
            det_stamp_sec = self.get_clock().now().nanoseconds * 1e-9

        orig_width = int(payload.get('image_width', self.intrinsics.width))
        orig_height = int(payload.get('image_height', self.intrinsics.height))
        rotation_angle = int(payload.get('rotation_angle', 0))

        intrinsics = self.intrinsics
        if not self.camera_info_received and (
            orig_width != intrinsics.width or orig_height != intrinsics.height
        ):
            intrinsics = CameraIntrinsics.from_fov(
                orig_width, orig_height, self.camera_hfov_rad
            )

        # Match closest LaserScan within ±max_scan_sync_dt_sec using core ring-buffer matcher
        matched_scan: Optional[LaserScan] = find_closest_timed_item(
            list(self.scan_buffer), det_stamp_sec, max_delta_sec=self.max_scan_sync_dt_sec
        )

        # Lookup TF matrices at detection timestamp
        query_time = Time(
            seconds=int(stamp_dict.get('sec', 0)),
            nanoseconds=int(stamp_dict.get('nanosec', 0)),
        )
        t_map_cam, target_frame = self._lookup_map_from_cam(query_time)
        if t_map_cam is None:
            return
        self.active_target_frame = target_frame

        t_cam_scan = self._lookup_transform_matrix(
            self.camera_optical_frame, self.scan_frame, query_time, fallback=default_t_cam_scan()
        )
        t_base_cam = self._lookup_transform_matrix(
            self.base_frame, self.camera_optical_frame, query_time, fallback=default_t_base_cam()
        )

        ranges_np: Optional[np.ndarray] = None
        angle_min = 0.0
        angle_inc = 0.0
        if matched_scan is not None and len(matched_scan.ranges) > 0:
            ranges_np = np.asarray(matched_scan.ranges, dtype=np.float64)
            angle_min = float(matched_scan.angle_min)
            angle_inc = float(matched_scan.angle_increment)

        updated = False
        for det in detections:
            bbox_rot = det.get('bbox_xyxy', [0.0, 0.0, 0.0, 0.0])
            if len(bbox_rot) < 4:
                continue
            cls_id = int(det.get('class_id', 0))
            cls_name = str(det.get('class_name', 'object'))
            conf = float(det.get('conf', 0.5))

            # Unrotate bounding box from rotated YOLO frame back to original camera frame
            bbox_orig = unrotate_bbox_xyxy(
                (float(bbox_rot[0]), float(bbox_rot[1]), float(bbox_rot[2]), float(bbox_rot[3])),
                orig_width=orig_width,
                orig_height=orig_height,
                rotation_angle=rotation_angle,
            )

            # Estimate 3D position P_camera in camera_rgb_optical_frame
            depth_est: Optional[DepthEstimate] = self.depth_estimator.estimate(
                bbox_xyxy=bbox_orig,
                intrinsics=intrinsics,
                T_cam_scan=t_cam_scan,
                ranges=ranges_np,
                angle_min=angle_min,
                angle_increment=angle_inc,
                T_base_cam=t_base_cam,
                depth_image=self.latest_depth_image,
            )
            if depth_est is None or depth_est.point_cam is None:
                continue

            # Transform P_camera -> P_map = T_map_cam * P_camera (Paper Eq. 4-6)
            p_map = transform_point(depth_est.point_cam, t_map_cam)

            obs = SemanticObservation(
                class_id=cls_id,
                class_name=cls_name,
                x=float(p_map[0]),
                y=float(p_map[1]),
                z=float(p_map[2]),
                conf=conf,
                stamp_sec=det_stamp_sec,
                low_confidence=bool(depth_est.low_confidence),
                method=str(depth_est.method),
            )
            self.marker_manager.add_observation(
                obs, angular_vel_rad_s=self.current_angular_vel
            )
            updated = True

        if updated:
            self.publish_markers()

    # -----------------------------------------------------------------------------
    # TF2 Lookup Helpers
    # -----------------------------------------------------------------------------
    def _lookup_transform_matrix(
        self,
        target_frame: str,
        source_frame: str,
        query_time: Time,
        fallback: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        for t in (query_time, Time()):
            try:
                tf_msg = self.tf_buffer.lookup_transform(
                    target_frame, source_frame, t, timeout=Duration(seconds=0.02)
                )
                return transform_stamped_to_matrix(tf_msg)
            except Exception:
                continue
        return fallback

    def _lookup_map_from_cam(self, query_time: Time) -> Tuple[Optional[np.ndarray], str]:
        mat = self._lookup_transform_matrix(
            self.map_frame, self.camera_optical_frame, query_time, fallback=None
        )
        if mat is not None:
            return mat, self.map_frame

        if self.use_odom_fallback:
            mat_odom = self._lookup_transform_matrix(
                self.odom_frame, self.camera_optical_frame, query_time, fallback=None
            )
            if mat_odom is not None:
                return mat_odom, self.odom_frame

        return None, self.map_frame

    def _lookup_robot_position(self) -> Tuple[float, float, float]:
        for target in (self.map_frame, self.odom_frame):
            mat = self._lookup_transform_matrix(target, self.base_frame, Time(), fallback=None)
            if mat is not None:
                self.active_target_frame = target
                return float(mat[0, 3]), float(mat[1, 3]), float(mat[2, 3])
        return float(self.latest_odom_xy[0]), float(self.latest_odom_xy[1]), 0.0

    # -----------------------------------------------------------------------------
    # RViz2 MarkerArray Visualization
    # -----------------------------------------------------------------------------
    def _get_color_bgr(self, class_name: str) -> Tuple[int, int, int]:
        if class_name in self.class_palette:
            return self.class_palette[class_name]
        lower = class_name.lower()
        if lower in self.class_palette:
            return self.class_palette[lower]
        return self.class_palette.get('default', (100, 200, 0))

    def publish_markers(self):
        perm_markers: List[PermanentMarker] = list(self.marker_manager.perm_markers)
        arr = MarkerArray()
        now_msg = self.get_clock().now().to_msg()
        frame_id = self.active_target_frame or self.map_frame

        for idx, pm in enumerate(perm_markers):
            cls_name = str(pm.class_name)
            b, g, r = self._get_color_bgr(cls_name)
            mx = float(pm.x)
            my = float(pm.y)
            mz = float(pm.z)
            conf = float(pm.conf)
            hits = int(pm.hits)
            low_conf = bool(pm.low_confidence)
            marker_id = int(pm.marker_id if pm.marker_id is not None else idx + 1)

            # 1. Geometry Marker (Cylinder for hazards/persons, Cube for furniture/obstacles)
            m = Marker()
            m.header.stamp = now_msg
            m.header.frame_id = frame_id
            m.ns = 'semantic_objects'
            m.id = marker_id * 2
            if cls_name in ('person', 'GAS_HAZARD', 'AUDIO_DISTRESS'):
                m.type = Marker.CYLINDER
            else:
                m.type = Marker.CUBE
            m.action = Marker.ADD
            m.pose.position.x = mx
            m.pose.position.y = my
            m.pose.position.z = max(0.15, mz if mz > 0.0 else 0.15)
            m.pose.orientation.w = 1.0
            m.scale.x = 0.28
            m.scale.y = 0.28
            m.scale.z = 0.30
            m.color = ColorRGBA(
                r=float(r) / 255.0,
                g=float(g) / 255.0,
                b=float(b) / 255.0,
                a=0.55 if low_conf else 0.90,
            )
            m.lifetime = MsgDuration(sec=0, nanosec=0)
            arr.markers.append(m)

            # 2. TEXT_VIEW_FACING Label Marker
            txt = Marker()
            txt.header.stamp = now_msg
            txt.header.frame_id = frame_id
            txt.ns = 'semantic_labels'
            txt.id = marker_id * 2 + 1
            txt.type = Marker.TEXT_VIEW_FACING
            txt.action = Marker.ADD
            txt.pose.position.x = mx
            txt.pose.position.y = my
            txt.pose.position.z = m.pose.position.z + 0.28
            txt.pose.orientation.w = 1.0
            txt.scale.z = 0.16
            txt.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=0.95)
            tag = ' [LOW]' if low_conf else ''
            txt.text = f"{cls_name} ({conf:.2f}, n={hits}){tag}"
            txt.lifetime = MsgDuration(sec=0, nanosec=0)
            arr.markers.append(txt)

        self.pub_markers.publish(arr)
        self.publish_cached_objects(perm_markers, source_frame=frame_id)

    def publish_cached_objects(
        self,
        perm_markers: Optional[List[PermanentMarker]] = None,
        source_frame: Optional[str] = None,
    ):
        """
        Publish lightweight JSON cache of verified permanent semantic markers in the `odom`
        frame on `/semantic_map/cached_objects` so RPi 4 onboard `recovery_controller` can
        cache Ch 1 (low/indoor obstacles) and Ch 2 (gas/survivor/audio zones) across comm loss.
        """
        if perm_markers is None:
            perm_markers = list(self.marker_manager.perm_markers)
        src_frame = source_frame or self.active_target_frame or self.map_frame

        if src_frame == self.odom_frame:
            t_odom_src = np.eye(4, dtype=np.float64)
        else:
            t_odom_src = self._lookup_transform_matrix(
                self.odom_frame, src_frame, Time(), fallback=np.eye(4, dtype=np.float64)
            )

        objects_list: List[Dict[str, Any]] = []
        for idx, pm in enumerate(perm_markers):
            cls_name = str(pm.class_name)
            p_odom = transform_point((float(pm.x), float(pm.y), float(pm.z)), t_odom_src)
            is_ch2 = cls_name in ('person', 'GAS_HAZARD', 'AUDIO_DISTRESS') or cls_name.lower() in (
                'person', 'gas_hazard', 'audio_distress'
            )
            channel = 2 if is_ch2 else 1
            radius = self.ch2_hazard_radius_m if channel == 2 else self.ch1_obstacle_radius_m
            marker_id = int(pm.marker_id if pm.marker_id is not None else idx + 1)

            objects_list.append({
                'marker_id': marker_id,
                'class_id': int(pm.class_id),
                'class_name': cls_name,
                'x': round(float(p_odom[0]), 4),
                'y': round(float(p_odom[1]), 4),
                'z': round(float(p_odom[2]), 4),
                'radius': float(radius),
                'channel': int(channel),
                'conf': round(float(pm.conf), 4),
                'hits': int(pm.hits),
                'low_confidence': bool(pm.low_confidence),
            })

        msg = String()
        msg.data = json.dumps({
            'frame_id': self.odom_frame,
            'objects': objects_list,
        })
        self.pub_cached_objects.publish(msg)

    # -----------------------------------------------------------------------------
    # Map Saving Service (Algorithm 2) & Shutdown Export
    # -----------------------------------------------------------------------------
    def handle_save_map(self, request: Trigger.Request, response: Trigger.Response):
        ok, msg = self.save_semantic_map_artifacts()
        response.success = ok
        response.message = msg
        return response

    def save_semantic_map_artifacts(self, target_dir: Optional[str] = None) -> Tuple[bool, str]:
        out_dir = os.path.abspath(target_dir or self.output_dir)
        try:
            if self.latest_map is not None:
                width = int(self.latest_map.info.width)
                height = int(self.latest_map.info.height)
                resolution = float(self.latest_map.info.resolution)
                origin_x = float(self.latest_map.info.origin.position.x)
                origin_y = float(self.latest_map.info.origin.position.y)
                grid_data = np.asarray(self.latest_map.data, dtype=np.int16).reshape((height, width))
            else:
                width, height = 200, 200
                resolution = 0.05
                origin_x, origin_y = -5.0, -5.0
                grid_data = np.full((height, width), -1, dtype=np.int16)

            paths = save_semantic_map_bundle(
                output_dir=out_dir,
                occupancy_grid=grid_data,
                width=width,
                height=height,
                resolution=resolution,
                origin_x=origin_x,
                origin_y=origin_y,
                marker_manager=self.marker_manager,
                palette=self.class_palette,
            )

            msg = (
                f"Saved semantic map artifacts ({len(self.marker_manager.perm_markers)} markers) to {out_dir}: "
                + ", ".join(os.path.basename(p) for p in paths.values())
            )
            self.get_logger().info(msg)
            return True, msg
        except Exception as e:
            err_msg = f"Failed to save semantic map artifacts to {out_dir}: {e}"
            self.get_logger().error(err_msg)
            return False, err_msg

    def destroy_node(self):
        if self.save_on_shutdown and not self._saved_on_shutdown:
            self._saved_on_shutdown = True
            self.save_semantic_map_artifacts()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = SemanticMapNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("SemanticMapNode shutting down...")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
