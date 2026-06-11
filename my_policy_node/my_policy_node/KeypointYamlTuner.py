# export AIC_KEYPOINT_CONFIG=$PWD/my_policy_node/my_policy_node/perception/keypoint_configs/aic_keypoint_config.yaml

from __future__ import annotations

import os
import math
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml

from aic_control_interfaces.msg import MotionUpdate, TrajectoryGenerationMode
from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    SendFeedbackCallback,
)
from aic_model_interfaces.msg import Observation
from aic_task_interfaces.msg import Task
from geometry_msgs.msg import Point, Pose, Quaternion, Twist, Vector3, Wrench
from rclpy.time import Time
from std_msgs.msg import Header
from tf2_ros import TransformException


class C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    DIM = "\033[2m"


class KeypointYamlTuner(Policy):
    """Interactive keypoint YAML tuner for AIC perception labels.

    Run with the normal AIC engine path:

        /entrypoint.sh ground_truth:=true start_aic_engine:=true gazebo_gui:=true

    Then run this policy:

        PYTHONPATH=$PWD/my_policy_node:$PYTHONPATH \
        pixi run ros2 run aic_model aic_model \
          --ros-args \
          -p use_sim_time:=true \
          -p policy:=my_policy_node.KeypointYamlTuner

    This policy does NOT insert the cable. It opens an OpenCV UI, projects the
    YAML-defined 3D keypoints onto the left/center/right wrist camera images
    using ground-truth TF + CameraInfo, and lets you tune the 3D coordinates.

    YAML path is controlled by:

        AIC_KEYPOINT_CONFIG=/absolute/path/to/aic_keypoint_config.yaml

    Default:

        ~/ws_aic/src/aic/my_policy_node/my_policy_node/perception/keypoint_configs/aic_keypoint_config.yaml

    Notes:
      - Coordinates are in meters.
      - SC is one module with two holes. The keypoint names use hole_a/hole_b,
        not port_0/port_1.
      - SFP uses one SFP port entrance frame as the reference for the whole NIC
        front keypoint set. By default that is sfp_port_0_link_entrance.
    """

    WINDOW = "AIC keypoint YAML tuner"
    DEFAULT_SCALE = 0.42
    TUNE_STEP_M = 0.0005       # 0.5 mm keypoint edit step
    MOVE_STEP_M = 0.005        # 5 mm TCP jog step
    TRACKBAR_RANGE = 2000      # -100.0 mm to +100.0 mm in 0.1 mm units
    TRACKBAR_CENTER = 1000
    TRACKBAR_MM_PER_TICK = 0.1

    FREE_SPACE_STIFFNESS = [70.0, 70.0, 70.0, 35.0, 35.0, 35.0]
    FREE_SPACE_DAMPING = [45.0, 45.0, 45.0, 18.0, 18.0, 18.0]

    def __init__(self, parent_node):
        super().__init__(parent_node)
        self.config_path = Path(
            os.environ.get(
                "AIC_KEYPOINT_CONFIG",
                str(Path.home() / "ws_aic/src/aic/my_policy_node/my_policy_node/perception/keypoint_configs/aic_keypoint_config.yaml"),
            )
        )
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        self.config = self._load_or_create_config(self.config_path)

        self.active_object_name = "auto"
        self.selected_idx = 0
        self.mode = "TUNE"  # TUNE or MOVE
        self.scale = float(self.config.get("ui", {}).get("display_scale", self.DEFAULT_SCALE))
        self._trackbar_busy = False
        self._last_help_t = 0.0
        self._last_save_t = 0.0
        self._last_frame_path: Optional[Path] = None

        self.get_logger().info(
            f"{C.CYAN}{C.BOLD}[INIT]{C.RESET} KeypointYamlTuner using config: {self.config_path}"
        )

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------

    def _default_config(self) -> dict:
        return {
            "version": 1,
            "units": "meters",
            "ui": {
                "display_scale": self.DEFAULT_SCALE,
                "draw_all_objects": False,
            },
            "objects": {
                "sc": {
                    "class_id": 0,
                    "description": "SC module keypoints. This is one SC port/module with two visible holes; do not treat the holes as separate ports.",
                    "reference_frame_template": "task_board/{target_module_name}/{port_name}_link_entrance",
                    "fallback_reference_frame_template": "task_board/{target_module_name}/{port_name}_link",
                    "keypoints": {
                        # These are intentionally rough placeholders. Tune them in the UI.
                        "outer_tl": [-0.030,  0.016, 0.000],
                        "outer_tr": [ 0.030,  0.016, 0.000],
                        "outer_br": [ 0.030, -0.016, 0.000],
                        "outer_bl": [-0.030, -0.016, 0.000],
                        "hole_a_left":  [-0.017,  0.000, 0.000],
                        "hole_a_top":   [-0.012,  0.005, 0.000],
                        "hole_a_right": [-0.007,  0.000, 0.000],
                        "hole_a_bottom":[-0.012, -0.005, 0.000],
                        "hole_a_center":[-0.012,  0.000, 0.000],
                        "hole_b_left":  [ 0.007,  0.000, 0.000],
                        "hole_b_top":   [ 0.012,  0.005, 0.000],
                        "hole_b_right": [ 0.017,  0.000, 0.000],
                        "hole_b_bottom":[ 0.012, -0.005, 0.000],
                        "hole_b_center":[ 0.012,  0.000, 0.000],
                    },
                },
                "sfp": {
                    "class_id": 0,
                    "description": "SFP/NIC-front keypoints. Reference is one SFP port frame on the target NIC card; port0 and port1 are both labeled in that same frame.",
                    "sfp_reference_port": "sfp_port_0",
                    "reference_frame_template": "task_board/{target_module_name}/{sfp_reference_port}_link_entrance",
                    "fallback_reference_frame_template": "task_board/{target_module_name}/{sfp_reference_port}_link",
                    "keypoints": {
                        # Rough placeholders relative to sfp_port_0 entrance/link. Tune them in the UI.
                        "nic_outer_tl": [-0.032,  0.018, 0.000],
                        "nic_outer_tr": [ 0.048,  0.018, 0.000],
                        "nic_outer_br": [ 0.048, -0.018, 0.000],
                        "nic_outer_bl": [-0.032, -0.018, 0.000],
                        "port0_tl": [-0.006,  0.004, 0.000],
                        "port0_tr": [ 0.006,  0.004, 0.000],
                        "port0_br": [ 0.006, -0.004, 0.000],
                        "port0_bl": [-0.006, -0.004, 0.000],
                        "port0_center": [0.000, 0.000, 0.000],
                        "port1_tl": [ 0.020,  0.004, 0.000],
                        "port1_tr": [ 0.032,  0.004, 0.000],
                        "port1_br": [ 0.032, -0.004, 0.000],
                        "port1_bl": [ 0.020, -0.004, 0.000],
                        "port1_center": [0.026, 0.000, 0.000],
                        "left_ref": [-0.022, -0.012, 0.000],
                        "right_ref": [0.042, -0.012, 0.000],
                    },
                },
            },
        }

    def _load_or_create_config(self, path: Path) -> dict:
        if not path.exists():
            cfg = self._default_config()
            path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
            return cfg
        with path.open("r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        default = self._default_config()
        # Shallow merge: keep user's objects, fill missing top-level keys.
        for k, v in default.items():
            loaded.setdefault(k, v)
        loaded.setdefault("objects", default["objects"])
        return loaded

    def _save_config(self) -> None:
        """Save the edited keypoint YAML.

        If the configured path is not writable because it was copied/created by
        root or mounted read-only, do not crash the action thread. Save a
        fallback copy in the user's home directory and print the exact path.
        """
        target_path = self.config_path

        try:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            with target_path.open("w", encoding="utf-8") as f:
                yaml.safe_dump(self.config, f, sort_keys=False)
        except PermissionError as ex:
            fallback_path = Path.home() / "aic_keypoint_config_saved.yaml"
            with fallback_path.open("w", encoding="utf-8") as f:
                yaml.safe_dump(self.config, f, sort_keys=False)

            self._last_save_t = time.time()
            self.get_logger().error(
                f"{C.RED}{C.BOLD}[SAVE]{C.RESET} permission denied for {target_path}: {ex}"
            )
            self.get_logger().info(
                f"{C.YELLOW}{C.BOLD}[SAVE]{C.RESET} wrote fallback copy to {fallback_path}. "
                "Fix ownership/permissions or copy this file back to your config path."
            )
            return

        self._last_save_t = time.time()
        self.get_logger().info(f"{C.GREEN}{C.BOLD}[SAVE]{C.RESET} wrote {target_path}")

    def _object_names(self) -> list[str]:
        return list(self.config.get("objects", {}).keys())

    def _active_object_for_task(self, task: Task) -> str:
        if self.active_object_name != "auto":
            return self.active_object_name
        if task.port_type.lower() == "sc":
            return "sc"
        if task.port_type.lower() == "sfp":
            return "sfp"
        # Fallback to first object in YAML.
        names = self._object_names()
        return names[0] if names else "sc"

    def _keypoint_items(self, object_name: str) -> list[tuple[str, list[float]]]:
        obj = self.config["objects"][object_name]
        return list(obj.get("keypoints", {}).items())

    def _selected_keypoint_name(self, object_name: str) -> str:
        items = self._keypoint_items(object_name)
        if not items:
            return ""
        self.selected_idx = int(np.clip(self.selected_idx, 0, len(items) - 1))
        return items[self.selected_idx][0]

    def _selected_xyz(self, object_name: str) -> np.ndarray:
        name = self._selected_keypoint_name(object_name)
        return np.asarray(self.config["objects"][object_name]["keypoints"][name], dtype=float)

    def _set_selected_xyz(self, object_name: str, xyz: np.ndarray) -> None:
        name = self._selected_keypoint_name(object_name)
        self.config["objects"][object_name]["keypoints"][name] = [float(x) for x in xyz]

    def _format_frame_template(self, template: str, task: Task, object_name: str) -> str:
        obj = self.config["objects"][object_name]
        return template.format(
            target_module_name=task.target_module_name,
            port_name=task.port_name,
            port_type=task.port_type,
            plug_name=task.plug_name,
            cable_name=task.cable_name,
            sfp_reference_port=obj.get("sfp_reference_port", "sfp_port_0"),
        )

    def _reference_frame(self, task: Task, object_name: str) -> str:
        obj = self.config["objects"][object_name]
        return self._format_frame_template(obj["reference_frame_template"], task, object_name)

    def _fallback_reference_frame(self, task: Task, object_name: str) -> str:
        obj = self.config["objects"][object_name]
        return self._format_frame_template(obj["fallback_reference_frame_template"], task, object_name)

    # ------------------------------------------------------------------
    # TF / transform helpers
    # ------------------------------------------------------------------

    def lookup_transform(self, target_frame: str, source_frame: str):
        return self._parent_node._tf_buffer.lookup_transform(target_frame, source_frame, Time())

    def _quat_xyzw_to_R(self, q) -> np.ndarray:
        x, y, z, w = float(q.x), float(q.y), float(q.z), float(q.w)
        n = math.sqrt(x * x + y * y + z * z + w * w)
        if n < 1e-12:
            return np.eye(3, dtype=float)
        x, y, z, w = x / n, y / n, z / n, w / n
        xx, yy, zz = x * x, y * y, z * z
        xy, xz, yz = x * y, x * z, y * z
        wx, wy, wz = w * x, w * y, w * z
        return np.array(
            [
                [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
                [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
                [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
            ],
            dtype=float,
        )

    def _transform_msg_to_mat(self, tf_msg) -> np.ndarray:
        T = np.eye(4, dtype=float)
        T[:3, :3] = self._quat_xyzw_to_R(tf_msg.rotation)
        T[:3, 3] = [float(tf_msg.translation.x), float(tf_msg.translation.y), float(tf_msg.translation.z)]
        return T

    def _pose_from_T(self, T: np.ndarray) -> Pose:
        # We only need this for jogging: keep orientation from existing TF msg where possible.
        # This simple matrix-to-quat is robust enough for debug motion.
        R = T[:3, :3]
        tr = float(np.trace(R))
        if tr > 0.0:
            s = math.sqrt(tr + 1.0) * 2.0
            qw = 0.25 * s
            qx = (R[2, 1] - R[1, 2]) / s
            qy = (R[0, 2] - R[2, 0]) / s
            qz = (R[1, 0] - R[0, 1]) / s
        else:
            i = int(np.argmax(np.diag(R)))
            if i == 0:
                s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
                qw = (R[2, 1] - R[1, 2]) / s
                qx = 0.25 * s
                qy = (R[0, 1] + R[1, 0]) / s
                qz = (R[0, 2] + R[2, 0]) / s
            elif i == 1:
                s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
                qw = (R[0, 2] - R[2, 0]) / s
                qx = (R[0, 1] + R[1, 0]) / s
                qy = 0.25 * s
                qz = (R[1, 2] + R[2, 1]) / s
            else:
                s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
                qw = (R[1, 0] - R[0, 1]) / s
                qx = (R[0, 2] + R[2, 0]) / s
                qy = (R[1, 2] + R[2, 1]) / s
                qz = 0.25 * s
        q = np.array([qx, qy, qz, qw], dtype=float)
        q = q / max(np.linalg.norm(q), 1e-12)
        return Pose(
            position=Point(x=float(T[0, 3]), y=float(T[1, 3]), z=float(T[2, 3])),
            orientation=Quaternion(x=float(q[0]), y=float(q[1]), z=float(q[2]), w=float(q[3])),
        )

    def _base_T_frame(self, frame: str) -> np.ndarray:
        tf = self.lookup_transform("base_link", frame).transform
        return self._transform_msg_to_mat(tf)

    def _resolve_reference_transform(self, task: Task, object_name: str) -> tuple[str, np.ndarray]:
        primary = self._reference_frame(task, object_name)
        try:
            return primary, self._base_T_frame(primary)
        except TransformException:
            fallback = self._fallback_reference_frame(task, object_name)
            return fallback, self._base_T_frame(fallback)

    # ------------------------------------------------------------------
    # Image / projection
    # ------------------------------------------------------------------

    def _image_msg_to_bgr(self, msg) -> np.ndarray:
        h, w = int(msg.height), int(msg.width)
        enc = str(msg.encoding).lower()
        data = np.frombuffer(msg.data, dtype=np.uint8)
        step = int(msg.step)

        def reshape_channels(ch: int) -> np.ndarray:
            row = data.reshape(h, step)
            pix = row[:, : w * ch].reshape(h, w, ch)
            return pix.copy()

        if enc in ("bgr8", "8uc3"):
            return reshape_channels(3)
        if enc == "rgb8":
            return cv2.cvtColor(reshape_channels(3), cv2.COLOR_RGB2BGR)
        if enc == "bgra8":
            return cv2.cvtColor(reshape_channels(4), cv2.COLOR_BGRA2BGR)
        if enc == "rgba8":
            return cv2.cvtColor(reshape_channels(4), cv2.COLOR_RGBA2BGR)
        if enc in ("mono8", "8uc1"):
            return cv2.cvtColor(data.reshape(h, step)[:, :w].copy(), cv2.COLOR_GRAY2BGR)
        # Last-resort fallback.
        try:
            return reshape_channels(3)
        except Exception:
            return np.zeros((h, w, 3), dtype=np.uint8)

    def _camera_entries(self, obs: Observation) -> list[tuple[str, object, object]]:
        return [
            ("left", obs.left_image, obs.left_camera_info),
            ("center", obs.center_image, obs.center_camera_info),
            ("right", obs.right_image, obs.right_camera_info),
        ]

    def _project_object(self, object_name: str, task: Task, obs: Observation):
        ref_frame, T_base_ref = self._resolve_reference_transform(task, object_name)
        items = self._keypoint_items(object_name)
        if not items:
            return ref_frame, []

        names = [name for name, _ in items]
        object_points = np.asarray([xyz for _, xyz in items], dtype=np.float64)

        result = []
        for cam_name, image_msg, info in self._camera_entries(obs):
            img = self._image_msg_to_bgr(image_msg)
            H, W = img.shape[:2]
            cam_frame = str(info.header.frame_id)
            if not cam_frame:
                result.append((cam_name, img, [], names, ref_frame, "NO_CAMERA_FRAME"))
                continue
            try:
                T_base_cam = self._base_T_frame(cam_frame)
                T_cam_ref = np.linalg.inv(T_base_cam) @ T_base_ref
                R = T_cam_ref[:3, :3]
                t = T_cam_ref[:3, 3].reshape(3, 1)
                rvec, _ = cv2.Rodrigues(R)
                K = np.asarray(info.k, dtype=np.float64).reshape(3, 3)
                D = np.asarray(info.d, dtype=np.float64).reshape(-1, 1) if len(info.d) else None
                pixels, _ = cv2.projectPoints(object_points, rvec, t, K, D)
                pixels = pixels.reshape(-1, 2)
                pts_cam = (R @ object_points.T + t).T
                visibility = []
                for (u, v), pc in zip(pixels, pts_cam):
                    visible = bool(pc[2] > 0.0 and 0 <= u < W and 0 <= v < H)
                    visibility.append((float(u), float(v), visible, float(pc[2])))
                result.append((cam_name, img, visibility, names, ref_frame, cam_frame))
            except Exception as ex:
                result.append((cam_name, img, [], names, ref_frame, f"ERR: {ex}"))
        return ref_frame, result

    def _draw_panel(self, cam_name: str, img: np.ndarray, points, names, ref_frame: str, camera_frame: str, object_name: str) -> np.ndarray:
        out = img.copy()
        H, W = out.shape[:2]
        for i, p in enumerate(points):
            u, v, visible, z = p
            if not (math.isfinite(u) and math.isfinite(v)):
                continue
            center = (int(round(u)), int(round(v)))
            if visible:
                color = (0, 220, 255) if i != self.selected_idx else (0, 0, 255)
                radius = 5 if i != self.selected_idx else 8
                cv2.circle(out, center, radius, color, -1, lineType=cv2.LINE_AA)
                cv2.putText(out, str(i), (center[0] + 6, center[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
            else:
                # Draw off-image/depth-bad projections only if they are near image bounds.
                if -100 <= u <= W + 100 and -100 <= v <= H + 100:
                    cv2.circle(out, center, 4, (120, 120, 120), 1, lineType=cv2.LINE_AA)

        header = f"{cam_name} | obj={object_name} | ref={ref_frame}"
        cv2.rectangle(out, (0, 0), (W, 70), (0, 0, 0), -1)
        cv2.putText(out, header[:120], (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(out, f"cam={camera_frame}", (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1, cv2.LINE_AA)
        return out

    def _render(self, task: Task, obs: Observation, object_name: str) -> np.ndarray:
        ref_frame, panels = self._project_object(object_name, task, obs)
        rendered = []
        for cam_name, img, points, names, ref, cam_frame in panels:
            drawn = self._draw_panel(cam_name, img, points, names, ref, cam_frame, object_name)
            if self.scale != 1.0:
                drawn = cv2.resize(drawn, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
            rendered.append(drawn)

        if not rendered:
            return np.zeros((480, 640, 3), dtype=np.uint8)

        # Make heights equal before horizontal stacking.
        min_h = min(im.shape[0] for im in rendered)
        rendered = [im[:min_h, :, :] for im in rendered]
        canvas = np.hstack(rendered)

        # Bottom info bar.
        info_h = 140
        bar = np.zeros((info_h, canvas.shape[1], 3), dtype=np.uint8)
        selected_name = self._selected_keypoint_name(object_name)
        xyz = self._selected_xyz(object_name)
        controls_1 = "m mode | n/p keypoint | h/l X | j/k Y | u/i Z | s save | c capture | o object | ESC quit"
        controls_2 = "MOVE mode: w/x base X, a/d base Y, r/f base Z, +/- move step"
        status = f"mode={self.mode} obj={object_name} idx={self.selected_idx} name={selected_name} xyz=({xyz[0]:+.4f},{xyz[1]:+.4f},{xyz[2]:+.4f}) m"
        cv2.putText(bar, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(bar, controls_1, (10, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA)
        cv2.putText(bar, controls_2, (10, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 220, 255), 1, cv2.LINE_AA)
        cv2.putText(bar, f"config: {self.config_path}", (10, 124), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)
        return np.vstack([canvas, bar])

    # ------------------------------------------------------------------
    # OpenCV UI
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        cv2.namedWindow(self.WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.WINDOW, 1800, 760)
        cv2.createTrackbar("X 0.1mm", self.WINDOW, self.TRACKBAR_CENTER, self.TRACKBAR_RANGE, self._on_trackbar)
        cv2.createTrackbar("Y 0.1mm", self.WINDOW, self.TRACKBAR_CENTER, self.TRACKBAR_RANGE, self._on_trackbar)
        cv2.createTrackbar("Z 0.1mm", self.WINDOW, self.TRACKBAR_CENTER, self.TRACKBAR_RANGE, self._on_trackbar)

    def _xyz_to_trackbar(self, x_m: float) -> int:
        mm = float(x_m) * 1000.0
        return int(np.clip(round(mm / self.TRACKBAR_MM_PER_TICK) + self.TRACKBAR_CENTER, 0, self.TRACKBAR_RANGE))

    def _trackbar_to_m(self, value: int) -> float:
        mm = (int(value) - self.TRACKBAR_CENTER) * self.TRACKBAR_MM_PER_TICK
        return mm / 1000.0

    def _sync_trackbars_from_point(self, object_name: str) -> None:
        xyz = self._selected_xyz(object_name)
        self._trackbar_busy = True
        cv2.setTrackbarPos("X 0.1mm", self.WINDOW, self._xyz_to_trackbar(xyz[0]))
        cv2.setTrackbarPos("Y 0.1mm", self.WINDOW, self._xyz_to_trackbar(xyz[1]))
        cv2.setTrackbarPos("Z 0.1mm", self.WINDOW, self._xyz_to_trackbar(xyz[2]))
        self._trackbar_busy = False

    def _on_trackbar(self, _value: int) -> None:
        if self._trackbar_busy:
            return
        # The active object is resolved inside the callback loop; this callback
        # is intentionally no-op because it does not know the current task.
        pass

    def _apply_trackbars_to_point(self, object_name: str) -> None:
        if self.mode != "TUNE":
            return
        x = self._trackbar_to_m(cv2.getTrackbarPos("X 0.1mm", self.WINDOW))
        y = self._trackbar_to_m(cv2.getTrackbarPos("Y 0.1mm", self.WINDOW))
        z = self._trackbar_to_m(cv2.getTrackbarPos("Z 0.1mm", self.WINDOW))
        self._set_selected_xyz(object_name, np.array([x, y, z], dtype=float))

    def _cycle_object(self, task: Task) -> None:
        names = ["auto"] + self._object_names()
        try:
            idx = names.index(self.active_object_name)
        except ValueError:
            idx = 0
        self.active_object_name = names[(idx + 1) % len(names)]
        resolved = self._active_object_for_task(task)
        self.selected_idx = 0
        self._sync_trackbars_from_point(resolved)
        self.get_logger().info(f"{C.CYAN}[UI]{C.RESET} active object = {self.active_object_name} -> {resolved}")

    def _edit_selected(self, object_name: str, delta: np.ndarray) -> None:
        xyz = self._selected_xyz(object_name)
        xyz = xyz + delta
        self._set_selected_xyz(object_name, xyz)
        self._sync_trackbars_from_point(object_name)

    def _capture_overlay(self, canvas: np.ndarray) -> None:
        out_dir = self.config_path.parent / "overlays"
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        path = out_dir / f"overlay_{stamp}.png"
        cv2.imwrite(str(path), canvas)
        self._last_frame_path = path
        self.get_logger().info(f"{C.GREEN}[CAPTURE]{C.RESET} wrote {path}")

    def _print_help(self) -> None:
        self.get_logger().info(
            "\n"
            f"{C.CYAN}{C.BOLD}KeypointYamlTuner controls{C.RESET}\n"
            "  m           toggle TUNE/MOVE mode\n"
            "  n / p       next / previous keypoint\n"
            "  h / l       selected keypoint X -/+ 0.5 mm\n"
            "  j / k       selected keypoint Y -/+ 0.5 mm\n"
            "  u / i       selected keypoint Z -/+ 0.5 mm\n"
            "  trackbars   direct X/Y/Z coordinate edit in 0.1 mm units\n"
            "  o           cycle auto/sc/sfp object config\n"
            "  s           save YAML\n"
            "  c           save current overlay image\n"
            "  MOVE mode: w/x base X, a/d base Y, r/f base Z\n"
            "  MOVE mode: +/- adjust TCP jog step\n"
            "  ESC         quit tuner and return False to engine\n"
        )

    # ------------------------------------------------------------------
    # Robot jog commands
    # ------------------------------------------------------------------

    def diag36(self, values: list[float]) -> list[float]:
        return np.diag(values).flatten().astype(float).tolist()

    def _make_pose_cmd(self, pose: Pose) -> MotionUpdate:
        msg = MotionUpdate()
        msg.header = Header(frame_id="base_link", stamp=self.get_clock().now().to_msg())
        msg.pose = pose
        msg.velocity = Twist(linear=Vector3(), angular=Vector3())
        msg.target_stiffness = self.diag36(self.FREE_SPACE_STIFFNESS)
        msg.target_damping = self.diag36(self.FREE_SPACE_DAMPING)
        msg.feedforward_wrench_at_tip = Wrench(force=Vector3(), torque=Vector3())
        msg.wrench_feedback_gains_at_tip = [0.2, 0.2, 0.2, 0.0, 0.0, 0.0]
        msg.trajectory_generation_mode = TrajectoryGenerationMode(mode=TrajectoryGenerationMode.MODE_POSITION)
        return msg

    def _tcp_pose_with_delta(self, delta_base: np.ndarray) -> Pose:
        T = self._base_T_frame("gripper/tcp")
        T[:3, 3] += delta_base.astype(float)
        return self._pose_from_T(T)

    def _send_tcp_jog(self, move_robot: MoveRobotCallback, delta_base: np.ndarray) -> None:
        pose = self._tcp_pose_with_delta(delta_base)
        try:
            move_robot(motion_update=self._make_pose_cmd(pose))
            self.get_logger().info(
                f"{C.BLUE}[MOVE]{C.RESET} TCP jog base delta=({delta_base[0]:+.4f},{delta_base[1]:+.4f},{delta_base[2]:+.4f})"
            )
        except Exception as ex:
            self.get_logger().info(f"{C.RED}[MOVE]{C.RESET} jog failed: {ex}")

    # ------------------------------------------------------------------
    # Policy entrypoint
    # ------------------------------------------------------------------

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> bool:
        self.get_logger().info(f"{C.MAGENTA}{C.BOLD}[TASK]{C.RESET} {task}")
        self._print_help()
        self._setup_ui()

        object_name = self._active_object_for_task(task)
        self.selected_idx = 0
        self._sync_trackbars_from_point(object_name)
        send_feedback("Keypoint YAML tuner active. Use OpenCV window; ESC to quit.")

        while True:
            obs = get_observation()
            if obs is None:
                self.sleep_for(0.05)
                continue

            object_name = self._active_object_for_task(task)
            self._apply_trackbars_to_point(object_name)

            try:
                canvas = self._render(task, obs, object_name)
            except Exception as ex:
                canvas = np.zeros((480, 900, 3), dtype=np.uint8)
                cv2.putText(canvas, f"Render/TF error: {ex}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            cv2.imshow(self.WINDOW, canvas)
            key = cv2.waitKey(30) & 0xFF

            if key == 27:  # ESC
                self._save_config()
                cv2.destroyWindow(self.WINDOW)
                send_feedback("Keypoint YAML tuner closed")
                return False

            if key == ord("?"):
                self._print_help()

            elif key == ord("m"):
                self.mode = "MOVE" if self.mode == "TUNE" else "TUNE"
                self.get_logger().info(f"{C.CYAN}[UI]{C.RESET} mode={self.mode}")

            elif key == ord("o"):
                self._cycle_object(task)

            elif key == ord("n"):
                self.selected_idx = (self.selected_idx + 1) % max(1, len(self._keypoint_items(object_name)))
                self._sync_trackbars_from_point(object_name)

            elif key == ord("p"):
                self.selected_idx = (self.selected_idx - 1) % max(1, len(self._keypoint_items(object_name)))
                self._sync_trackbars_from_point(object_name)

            elif key == ord("s"):
                self._save_config()

            elif key == ord("c"):
                self._capture_overlay(canvas)

            elif self.mode == "TUNE":
                if key == ord("h"):
                    self._edit_selected(object_name, np.array([-self.TUNE_STEP_M, 0.0, 0.0]))
                elif key == ord("l"):
                    self._edit_selected(object_name, np.array([self.TUNE_STEP_M, 0.0, 0.0]))
                elif key == ord("j"):
                    self._edit_selected(object_name, np.array([0.0, -self.TUNE_STEP_M, 0.0]))
                elif key == ord("k"):
                    self._edit_selected(object_name, np.array([0.0, self.TUNE_STEP_M, 0.0]))
                elif key == ord("u"):
                    self._edit_selected(object_name, np.array([0.0, 0.0, -self.TUNE_STEP_M]))
                elif key == ord("i"):
                    self._edit_selected(object_name, np.array([0.0, 0.0, self.TUNE_STEP_M]))

            elif self.mode == "MOVE":
                if key in (ord("+"), ord("=")):
                    self.MOVE_STEP_M = min(0.050, self.MOVE_STEP_M + 0.001)
                    self.get_logger().info(f"{C.BLUE}[MOVE]{C.RESET} step={self.MOVE_STEP_M:.4f} m")
                elif key in (ord("-"), ord("_")):
                    self.MOVE_STEP_M = max(0.001, self.MOVE_STEP_M - 0.001)
                    self.get_logger().info(f"{C.BLUE}[MOVE]{C.RESET} step={self.MOVE_STEP_M:.4f} m")
                elif key == ord("w"):
                    self._send_tcp_jog(move_robot, np.array([self.MOVE_STEP_M, 0.0, 0.0]))
                elif key == ord("x"):
                    self._send_tcp_jog(move_robot, np.array([-self.MOVE_STEP_M, 0.0, 0.0]))
                elif key == ord("a"):
                    self._send_tcp_jog(move_robot, np.array([0.0, self.MOVE_STEP_M, 0.0]))
                elif key == ord("d"):
                    self._send_tcp_jog(move_robot, np.array([0.0, -self.MOVE_STEP_M, 0.0]))
                elif key == ord("r"):
                    self._send_tcp_jog(move_robot, np.array([0.0, 0.0, self.MOVE_STEP_M]))
                elif key == ord("f"):
                    self._send_tcp_jog(move_robot, np.array([0.0, 0.0, -self.MOVE_STEP_M]))

            # Keep the action alive and the UI responsive.
            self.sleep_for(0.01)