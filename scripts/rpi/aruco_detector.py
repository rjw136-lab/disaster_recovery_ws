#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=================================================================================================
Project: Disaster Recovery TurtleBot3 Burger System
File: aruco_detector.py
Description:
  ROS 2 Humble 기반 ArUco 마커 인식, 3D 포즈 추정 (거리 및 상대 위치) 및 초고속 30 FPS 웹 스트리밍 노드.
  - Option B: Sony IMX219 (라즈베리파이 카메라 V2, 640x480) 센서 스펙 기반 이론적 카메라 내부 행렬 자동 적용
  - 마커 중심 3D 좌표축 (Red: X, Green: Y, Blue: Z) 및 실제 물리적 거리(cm) 실시간 계산/표시
  - 선택적 캘리브레이션 YAML 파일 로딩 지원
=================================================================================================
"""

import sys
import os
import time
import argparse
import threading
import socket
from http.server import BaseHTTPRequestHandler, HTTPServer
import socketserver
import numpy as np
import cv2
import yaml
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

ARUCO_DICTS = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
}

# 웹 페이지 실시간 상태 표시용 전역 설정
CURRENT_CONFIG = {
    "marker_size_m": 0.05,
    "calib_type": "Sony IMX219 Theoretical (fx=fy=530px)",
}


class StreamingBuffer:
    def __init__(self):
        self.frame_data = None
        self.condition = threading.Condition()

    def set_frame(self, frame_bytes):
        with self.condition:
            self.frame_data = frame_bytes
            self.condition.notify_all()

    def get_frame(self):
        with self.condition:
            self.condition.wait(timeout=1.0)
            return self.frame_data


video_buffer = StreamingBuffer()


class StreamingHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def setup(self):
        super().setup()
        # Nagle 알고리즘 비활성화로 네트워크 지연 최소화
        try:
            self.request.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except Exception:
            pass

    def do_GET(self):
        if self.path == "/" or self.path.startswith("/?"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Connection", "close")
            self.end_headers()

            marker_cm = CURRENT_CONFIG["marker_size_m"] * 100.0
            calib_name = CURRENT_CONFIG["calib_type"]

            html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>카메라 실시간 30 FPS 영상 & ArUco 3D 포즈 추정</title>
    <style>
        body {{
            background-color: #111;
            color: #eee;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            text-align: center;
            margin: 0;
            padding: 15px;
        }}
        h1 {{ margin-bottom: 5px; color: #4CAF50; font-size: 22px; }}
        p {{ color: #aaa; margin-top: 0; font-size: 14px; }}
        .video-container {{
            display: inline-block;
            background: #000;
            border-radius: 12px;
            box-shadow: 0 8px 30px rgba(0,0,0,0.8);
            padding: 6px;
            margin-top: 10px;
        }}
        img {{
            max-height: 75vh;
            max-width: 95vw;
            border-radius: 8px;
            display: block;
        }}
        .status-badge {{
            margin-top: 12px;
            display: inline-block;
            padding: 6px 18px;
            background: #222;
            border: 1px solid #444;
            border-radius: 20px;
            font-size: 13px;
            color: #ddd;
        }}
        .live-dot {{
            height: 10px;
            width: 10px;
            background-color: #4CAF50;
            border-radius: 50%;
            display: inline-block;
            margin-right: 6px;
        }}
        .axis-info {{
            margin-top: 8px;
            font-size: 12px;
            color: #888;
        }}
        .axis-x {{ color: #ff5555; font-weight: bold; }}
        .axis-y {{ color: #55ff55; font-weight: bold; }}
        .axis-z {{ color: #5599ff; font-weight: bold; }}
    </style>
</head>
<body>
    <h1>📷 재난구조로봇 실시간 카메라 & ArUco 3D 거리 측정</h1>
    <p>ArUco 마커를 카메라에 비추면 3D 좌표축과 실제 거리(cm)가 화면에 실시간 표시됩니다.</p>
    <div class="video-container">
        <img src="/stream.mjpg" alt="카메라 화면 연결 중..." />
    </div>
    <br>
    <div class="status-badge">
        <span class="live-dot"></span>
        <span>초고속 스트리밍 중 (~30 FPS) | 기준 마커 크기: <strong>{marker_cm:.1f} cm</strong> ({calib_name})</span>
    </div>
    <div class="axis-info">
        3D 좌표계 범례: <span class="axis-x">빨간색(X 축: 좌우)</span> | <span class="axis-y">초록색(Y 축: 상하)</span> | <span class="axis-z">파란색(Z 축: 전방 깊이)</span>
    </div>
</body>
</html>"""
            self.wfile.write(html.encode("utf-8"))
            self.wfile.flush()

        elif self.path == "/stream.mjpg":
            self.send_response(200)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, private, no-store, must-revalidate")
            self.send_header("Pragma", "no-cache")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while True:
                    data = video_buffer.get_frame()
                    if data is None:
                        continue
                    self.wfile.write(
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n"
                        b"Content-Length: " + str(len(data)).encode() + b"\r\n\r\n"
                        + data + b"\r\n"
                    )
                    self.wfile.flush()
            except Exception:
                pass
        else:
            self.send_error(404)
            self.end_headers()

    def log_message(self, format, *args):
        return


class ThreadedHTTPServer(socketserver.ThreadingMixIn, HTTPServer):
    allow_reuse_address = True
    daemon_threads = True


def start_web_server(port=8080):
    server = ThreadedHTTPServer(("0.0.0.0", port), StreamingHandler)
    server.serve_forever()


class ArucoDetectorNode(Node):
    def __init__(
        self,
        dict_name="DICT_4X4_50",
        image_topic="/camera/image_raw",
        rotation=270,
        marker_size=0.05,
        calib_file=None,
    ):
        super().__init__("aruco_detector_node")

        self.declare_parameter("dictionary", dict_name)
        self.declare_parameter("image_topic", image_topic)
        self.declare_parameter("output_topic", "/camera/aruco/image")
        self.declare_parameter("rotation", rotation)
        self.declare_parameter("marker_size", marker_size)
        self.declare_parameter("calib_file", calib_file or "")

        dict_param = self.get_parameter("dictionary").get_parameter_value().string_value
        img_topic = self.get_parameter("image_topic").get_parameter_value().string_value
        out_topic = self.get_parameter("output_topic").get_parameter_value().string_value
        self.rotation = self.get_parameter("rotation").get_parameter_value().integer_value
        self.marker_size = self.get_parameter("marker_size").get_parameter_value().double_value
        self.calib_file = self.get_parameter("calib_file").get_parameter_value().string_value

        if dict_param not in ARUCO_DICTS:
            self.get_logger().warn(
                f"Dictionary '{dict_param}' not recognized. Falling back to DICT_4X4_50."
            )
            dict_param = "DICT_4X4_50"

        self.dictionary_name = dict_param
        self.dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICTS[dict_param])
        self.parameters = cv2.aruco.DetectorParameters_create()

        # 카메라 내부 파라미터 캐시
        self.camera_matrix = None
        self.dist_coeffs = None
        self.last_img_shape = None
        self.calib_type = "Sony IMX219 Theoretical (fx=fy=530px)"

        self.bridge = CvBridge()
        self.image_pub = self.create_publisher(Image, out_topic, 10)
        self.image_sub = self.create_subscription(
            Image, img_topic, self.image_callback, qos_profile_sensor_data
        )

        self.last_log_time = 0.0
        self.marker_detected_once = False

        print("\n" + "=" * 65)
        print("  🚀 ArUco 마커 인식 & 3D 포즈(거리/상대위치) 추정 시스템 가동")
        print(f"  - 딕셔너리: {self.dictionary_name}")
        print(f"  - 회전 각도: {self.rotation}도 (세로 모드 보정)")
        print(f"  - 기준 마커 크기: {self.marker_size * 100.0:.1f} cm ({self.marker_size} m)")
        print(f"  - 카메라 토픽: {img_topic}")
        print("  - 웹 브라우저 실시간 스트리밍:")
        print("      👉 Tailscale:   http://100.91.141.24:8080")
        print("      👉 로컬 접속:   http://192.168.0.135:8080")
        print("=" * 65)
        print("📷 카메라 측정 진행 중입니다. 카메라 앞에 마커를 비춰주세요...\n")

    def setup_camera_intrinsics(self, img_w, img_h):
        """카메라 해상도 및 회전 상태에 맞추어 내부 파라미터 행렬(K)과 왜곡 계수(D)를 구성합니다."""
        # 1. 사용자가 지정한 YAML 캘리브레이션 파일이 있는 경우 로드 시도
        if self.calib_file and os.path.isfile(self.calib_file):
            try:
                with open(self.calib_file, "r") as f:
                    calib_data = yaml.safe_load(f)
                    if "camera_matrix" in calib_data and "distortion_coefficients" in calib_data:
                        cm_data = calib_data["camera_matrix"]["data"]
                        self.camera_matrix = np.array(cm_data, dtype=np.float64).reshape((3, 3))
                        dist_data = calib_data["distortion_coefficients"]["data"]
                        self.dist_coeffs = np.array(dist_data, dtype=np.float64).reshape((-1, 1))
                        self.calib_type = f"YAML File ({os.path.basename(self.calib_file)})"
                        CURRENT_CONFIG["calib_type"] = self.calib_type
                        self.get_logger().info(f"Loaded camera calibration from {self.calib_file}")
                        return
            except Exception as e:
                self.get_logger().warn(f"Failed to load calibration file {self.calib_file}: {e}")

        # 2. Option B: Sony IMX219 (라즈베리파이 카메라 V2, 640x480) 이론적 파라미터 모델
        # HFOV ~ 62.2 deg, VFOV ~ 48.8 deg
        # fx = fy ≈ 530.0 px (정방형 픽셀)
        # 이미지 회전 후에도 fx=fy 이므로 초점거리는 일정하며,
        # 광학 중심(cx, cy)은 회전된 이미지의 정중앙에 위치함.
        fx = 530.0
        fy = 530.0
        cx = img_w / 2.0
        cy = img_h / 2.0

        self.camera_matrix = np.array([
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)
        self.dist_coeffs = np.zeros((5, 1), dtype=np.float64)
        self.calib_type = "Sony IMX219 Theoretical (fx=fy=530px)"
        CURRENT_CONFIG["calib_type"] = self.calib_type

    def image_callback(self, msg: Image):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"Image conversion failed: {e}")
            return

        # 카메라 물리적 장착 각도에 맞춰 회전 보정
        if self.rotation == 90:
            cv_image = cv2.rotate(cv_image, cv2.ROTATE_90_CLOCKWISE)
        elif self.rotation == 180:
            cv_image = cv2.rotate(cv_image, cv2.ROTATE_180)
        elif self.rotation == 270:
            cv_image = cv2.rotate(cv_image, cv2.ROTATE_90_COUNTERCLOCKWISE)

        img_h, img_w = cv_image.shape[:2]
        if self.camera_matrix is None or self.last_img_shape != (img_w, img_h):
            self.setup_camera_intrinsics(img_w, img_h)
            self.last_img_shape = (img_w, img_h)

        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = cv2.aruco.detectMarkers(
            gray, self.dictionary, parameters=self.parameters
        )

        annotated_image = cv_image.copy()
        now = self.get_clock().now().nanoseconds / 1e9

        if ids is not None and len(ids) > 0:
            cv2.aruco.drawDetectedMarkers(annotated_image, corners, ids)

            # 3D 포즈 추정 (rvec: 회전 벡터, tvec: 평행이동 벡터 [X, Y, Z])
            try:
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                    corners, self.marker_size, self.camera_matrix, self.dist_coeffs
                )
            except Exception as e:
                rvecs, tvecs = None, None

            detected_summary = []
            for i, marker_id in enumerate(ids.flatten()):
                pts = corners[i][0]
                cx = int(np.mean(pts[:, 0]))
                cy = int(np.mean(pts[:, 1]))
                side_len = int(np.linalg.norm(pts[0] - pts[1]))

                dist_cm = None
                x_cm, y_cm, z_cm = 0.0, 0.0, 0.0

                if tvecs is not None:
                    tvec = tvecs[i][0]
                    rvec = rvecs[i][0]
                    x_cm = float(tvec[0] * 100.0)
                    y_cm = float(tvec[1] * 100.0)
                    z_cm = float(tvec[2] * 100.0)
                    dist_cm = float(np.linalg.norm(tvec) * 100.0)

                    # 3D 좌표축 그리기 (빨강: X, 초록: Y, 파랑: Z)
                    try:
                        cv2.drawFrameAxes(
                            annotated_image,
                            self.camera_matrix,
                            self.dist_coeffs,
                            rvec,
                            tvec,
                            self.marker_size * 0.55
                        )
                    except Exception:
                        pass

                detected_summary.append({
                    "id": int(marker_id),
                    "cx": cx,
                    "cy": cy,
                    "side_len": side_len,
                    "dist_cm": dist_cm,
                    "x_cm": x_cm,
                    "y_cm": y_cm,
                    "z_cm": z_cm
                })

                cv2.circle(annotated_image, (cx, cy), 5, (0, 0, 255), -1)

                # 화면에 거리 및 상대 위치 텍스트 오버레이
                if dist_cm is not None:
                    label_main = f"ID:{marker_id}  Dist:{dist_cm:.1f}cm"
                    label_sub = f"Z:{z_cm:.1f} (X:{x_cm:+.1f}, Y:{y_cm:+.1f})"
                    cv2.putText(
                        annotated_image,
                        label_main,
                        (cx - 65, cy - 22),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.52,
                        (0, 255, 0),
                        2,
                    )
                    cv2.putText(
                        annotated_image,
                        label_sub,
                        (cx - 85, cy - 6),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.40,
                        (0, 255, 255),
                        1,
                    )
                else:
                    cv2.putText(
                        annotated_image,
                        f"ID:{marker_id} ({cx},{cy})",
                        (cx - 50, cy - 15),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )

            # 상단 상태 알림 배너
            primary = detected_summary[0]
            cv2.rectangle(annotated_image, (10, 10), (460, 52), (0, 140, 0), -1)
            if primary["dist_cm"] is not None:
                banner_text = f"[3D POSE] Dist: {primary['dist_cm']:.1f}cm (ID #{primary['id']})"
            else:
                banner_text = f"[MEASUREMENT COMPLETE] ID #{primary['id']}"
            cv2.putText(
                annotated_image,
                banner_text,
                (20, 38),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
            )

            # 마커 감지 시 터미널 로그 1회 출력 및 스냅샷 저장
            if not self.marker_detected_once:
                self.marker_detected_once = True
                self.last_log_time = now
                print("\n" + "=" * 65)
                print(f"🎉 [3D 포즈 측정 완료] ArUco 마커가 성공적으로 감지되었습니다!")
                for d in detected_summary:
                    if d["dist_cm"] is not None:
                        print(f"   ▶ 마커 ID: {d['id']:<2} | 실제 거리: {d['dist_cm']:>5.1f} cm  (전방 Z: {d['z_cm']:>5.1f} cm, 좌우 X: {d['x_cm']:>+5.1f} cm, 상하 Y: {d['y_cm']:>+5.1f} cm)")
                    else:
                        print(f"   ▶ 마커 ID: {d['id']:<2} | 중심 좌표: (X={d['cx']}, Y={d['cy']}) | 크기: {d['side_len']}px")
                print(f"   ▶ 측정 모델:   {self.calib_type} (마커 기준 크기: {self.marker_size*100.0:.1f} cm)")
                print(f"   ▶ 저장된 사진: /home/lab/detected_result.jpg")
                print(f"   ▶ 실시간 화면: http://100.91.141.24:8080")
                print("=" * 65 + "\n")
                sys.stdout.flush()

                cv2.imwrite("/home/lab/detected_result.jpg", annotated_image)

        else:
            self.marker_detected_once = False
            cv2.putText(
                annotated_image,
                "SEARCHING FOR ARUCO MARKERS...",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 200, 255),
                2,
            )

        # 고속 MJPEG 스트리밍용 JPEG 인코딩 (압축률 65 -> 지연시간 최소화)
        ret, jpeg = cv2.imencode(".jpg", annotated_image, [cv2.IMWRITE_JPEG_QUALITY, 65])
        if ret:
            video_buffer.set_frame(jpeg.tobytes())

        # ROS 이미지 토픽 발행
        try:
            out_msg = self.bridge.cv2_to_imgmsg(annotated_image, encoding="bgr8")
            out_msg.header = msg.header
            self.image_pub.publish(out_msg)
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(
        description="ROS 2 Humble ArUco Marker Detector with 3D Pose Estimation & 30 FPS Web Streaming"
    )
    parser.add_argument(
        "--dict",
        type=str,
        default="DICT_4X4_50",
        help=f"ArUco dictionary name. Options: {', '.join(ARUCO_DICTS.keys())}",
    )
    parser.add_argument(
        "--topic",
        type=str,
        default="/camera/image_raw",
        help="Input image topic name",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Web streaming port (default: 8080)",
    )
    parser.add_argument(
        "--rotate",
        type=int,
        default=270,
        choices=[0, 90, 180, 270],
        help="Image rotation in degrees (default: 270 for upright vertical)",
    )
    parser.add_argument(
        "--marker-size",
        type=float,
        default=0.05,
        help="ArUco marker physical side length in meters (default: 0.05 = 5.0 cm)",
    )
    parser.add_argument(
        "--size-cm",
        type=float,
        default=None,
        help="ArUco marker physical side length in centimeters (e.g. 5.0)",
    )
    parser.add_argument(
        "--calib",
        type=str,
        default=None,
        help="Path to camera calibration YAML file (optional, defaults to IMX219 theoretical matrix)",
    )

    args, unknown = parser.parse_known_args()

    marker_size = args.marker_size
    if args.size_cm is not None:
        marker_size = args.size_cm / 100.0

    CURRENT_CONFIG["marker_size_m"] = marker_size

    web_thread = threading.Thread(target=start_web_server, args=(args.port,), daemon=True)
    web_thread.start()

    rclpy.init(args=unknown)
    node = ArucoDetectorNode(
        dict_name=args.dict,
        image_topic=args.topic,
        rotation=args.rotate,
        marker_size=marker_size,
        calib_file=args.calib,
    )

    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
