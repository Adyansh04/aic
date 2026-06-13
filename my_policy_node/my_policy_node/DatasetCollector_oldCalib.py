# DatasetCollector manual entrance-frame velocity calibration v7.
# Purpose:
#   - No generated viewpoint candidates.
#   - No per-camera enable/disable flags.
#   - You manually teleoperate the wrist camera using velocity commands in the
#     active entrance/reference frame.
#   - The selected wrist camera always looks at the entrance/reference frame.
#   - Press SAVE to store the current camera offset relative to that entrance frame.
#   - Later dataset collection can reuse these stored offsets across trials.

from __future__ import annotations

import json
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


class DatasetCollector(Policy):
    """Manual safe-view calibration for AIC keypoint dataset generation.

    Run the world with ground truth and the AIC engine:

        /entrypoint.sh ground_truth:=true start_aic_engine:=true \
          aic_engine_config_file:=/path/to/aic_dataset_debug10.yaml

    Run this policy:

        cd ~/ws_aic/src/aic
        PYTHONPATH=$PWD/my_policy_node:$PYTHONPATH \
        pixi run ros2 run aic_model aic_model \
          --ros-args -p use_sim_time:=true -p policy:=my_policy_node.DatasetCollector

    This policy does not insert the cable and does not generate random poses.
    It shows the three wrist camera overlays and lets you manually teleoperate
    the selected motion camera in x/y/z of the target entrance/reference frame.

    It stores camera offsets in:
        calibrated_manual_viewpoints_sfp.yaml
        calibrated_manual_viewpoints_sc.yaml

    Coordinate convention for saved viewpoints:
        cam_offset_ref = inv(T_base_reference) * p_base_motion_camera

    During future collection, each saved cam_offset_ref can be replayed relative
    to the new trial's entrance/reference frame.
    """

    # ------------------------------------------------------------------
    # EDIT SETTINGS HERE
    # ------------------------------------------------------------------

    KEYPOINT_CONFIG_PATH = "my_policy_node/my_policy_node/perception/keypoint_configs/aic_keypoint_config.yaml"
    VIEWPOINT_OUTPUT_DIR = "my_policy_node/my_policy_node/perception/keypoint_configs"
    SFP_VIEWPOINT_FILE = "calibrated_manual_viewpoints_sfp.yaml"
    SC_VIEWPOINT_FILE = "calibrated_manual_viewpoints_sc.yaml"

    # Save this many accepted poses per object type. After this count, this
    # policy returns True so the engine can advance.
    TARGET_VIEWPOINTS_PER_OBJECT = 40

    # Which wrist camera pose is teleoperated and stored. The other two cameras
    # are shown only for visual inspection.
    MOTION_CAMERA_NAME = "center"  # left / center / right

    # Teleop loop rate and speeds. Velocities are in the reference/entrance frame.
    CONTROL_DT = 0.05  # 20 Hz
    SLOW_LINEAR_SPEED = 0.010  # m/s
    FAST_LINEAR_SPEED = 0.025  # m/s
    START_FAST = False

    # Image/UI speed. Lower width = faster UI.
    PREVIEW_CAMERA_WIDTH = 390
    PREVIEW_REFRESH_EVERY_N_LOOPS = 2
    WINDOW_NAME = "AIC manual viewpoint calibration"

    # Projection/display settings.
    BBOX_MARGIN_PX = 10.0
    MIN_VISIBLE_KEYPOINTS = 4

    # Controller gains for manual free-space motion. Keep these soft-ish.
    FREE_SPACE_STIFFNESS = [40.0, 40.0, 40.0, 24.0, 24.0, 24.0]
    FREE_SPACE_DAMPING = [38.0, 38.0, 38.0, 18.0, 18.0, 18.0]

    # Try MODE_VELOCITY if available. If your local AIC message does not expose
    # this enum, the code falls back to MODE_POSITION while still populating the
    # velocity field and continuously integrating the target pose.
    PREFER_VELOCITY_MODE = True

    # Button geometry.
    BUTTON_H = 42
    BUTTON_MARGIN = 8

    def __init__(self, parent_node):
        super().__init__(parent_node)
        self.config_path = Path(self.KEYPOINT_CONFIG_PATH).expanduser()
        self.view_out_dir = Path(self.VIEWPOINT_OUTPUT_DIR).expanduser()
        self.view_out_dir.mkdir(parents=True, exist_ok=True)

        self.config = self._load_config(self.config_path)

        self._active_velocity_ref = np.zeros(3, dtype=float)
        self._fast = bool(self.START_FAST)
        self._running = True
        self._save_requested = False
        self._save_png_requested = False
        self._quit_requested = False
        self._last_preview: Optional[np.ndarray] = None
        self._last_preview_meta: dict = {}
        self._buttons: dict[str, tuple[int, int, int, int]] = {}
        self._mouse_down_button: Optional[str] = None
        self._loop_count = 0

        self.get_logger().info(
            f"{C.CYAN}{C.BOLD}[INIT]{C.RESET} Manual velocity calibration; config={self.config_path}"
        )
        self.get_logger().info(
            f"{C.CYAN}[INIT]{C.RESET} motion_camera={self.MOTION_CAMERA_NAME} "
            f"target={self.TARGET_VIEWPOINTS_PER_OBJECT} per object"
        )

    # ------------------------------------------------------------------
    # Config and object helpers
    # ------------------------------------------------------------------

    def _load_config(self, path: Path) -> dict:
        if not path.exists():
            raise FileNotFoundError(f"KEYPOINT_CONFIG_PATH not found: {path}")
        with path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        cfg.setdefault("training", {})
        cfg["training"].setdefault("exclude_keypoints", {})
        cfg["training"].setdefault("export_keypoints", {})
        cfg.setdefault("objects", {})
        return cfg

    def _object_name_for_task(self, task: Task) -> str:
        pt = str(task.port_type).lower()
        if pt == "sc":
            return "sc"
        if pt == "sfp":
            return "sfp"
        raise ValueError(f"Unsupported task.port_type={task.port_type!r}; expected sc or sfp")

    def _viewpoint_file(self, object_name: str) -> Path:
        if object_name == "sfp":
            return self.view_out_dir / self.SFP_VIEWPOINT_FILE
        if object_name == "sc":
            return self.view_out_dir / self.SC_VIEWPOINT_FILE
        raise ValueError(object_name)

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
        return self._format_frame_template(
            obj.get("fallback_reference_frame_template", obj["reference_frame_template"]),
            task,
            object_name,
        )

    def _all_keypoint_names(self, object_name: str) -> list[str]:
        return list(self.config["objects"][object_name].get("keypoints", {}).keys())

    def _export_keypoint_names(self, object_name: str) -> list[str]:
        training = self.config.get("training", {})
        explicit = training.get("export_keypoints", {}).get(object_name)
        if explicit:
            return [name for name in explicit if name in self.config["objects"][object_name]["keypoints"]]
        exclude = set(training.get("exclude_keypoints", {}).get(object_name, []))
        return [name for name in self._all_keypoint_names(object_name) if name not in exclude]

    def _object_points(self, object_name: str, names: list[str]) -> np.ndarray:
        kp = self.config["objects"][object_name]["keypoints"]
        return np.asarray([kp[name] for name in names], dtype=np.float64)

    def _load_saved_viewpoints(self, object_name: str) -> list[dict]:
        path = self._viewpoint_file(object_name)
        if not path.exists():
            return []
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            return list(data.get("viewpoints", []))
        except Exception as ex:
            self.get_logger().warn(f"{C.YELLOW}[YAML]{C.RESET} failed to load {path}: {ex}")
            return []

    def _write_saved_viewpoints(self, object_name: str, viewpoints: list[dict]) -> None:
        path = self._viewpoint_file(object_name)
        data = {
            "version": 1,
            "object_name": object_name,
            "units": "meters",
            "motion_camera_name": self.MOTION_CAMERA_NAME,
            "description": (
                "Manual calibrated camera offsets relative to the active entrance/reference frame. "
                "Each cam_offset_ref is [x,y,z] in that reference frame."
            ),
            "viewpoints": viewpoints,
        }
        path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
        self.get_logger().info(
            f"{C.GREEN}[YAML]{C.RESET} saved {len(viewpoints)} {object_name} viewpoints -> {path}"
        )

    # ------------------------------------------------------------------
    # TF and transform helpers
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

    def _mat_to_pose(self, T: np.ndarray) -> Pose:
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
                s = math.sqrt(max(1e-12, 1.0 + R[0, 0] - R[1, 1] - R[2, 2])) * 2.0
                qw = (R[2, 1] - R[1, 2]) / s
                qx = 0.25 * s
                qy = (R[0, 1] + R[1, 0]) / s
                qz = (R[0, 2] + R[2, 0]) / s
            elif i == 1:
                s = math.sqrt(max(1e-12, 1.0 + R[1, 1] - R[0, 0] - R[2, 2])) * 2.0
                qw = (R[0, 2] - R[2, 0]) / s
                qx = (R[0, 1] + R[1, 0]) / s
                qy = 0.25 * s
                qz = (R[1, 2] + R[2, 1]) / s
            else:
                s = math.sqrt(max(1e-12, 1.0 + R[2, 2] - R[0, 0] - R[1, 1])) * 2.0
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
    # Camera / projection helpers
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
        return reshape_channels(3)

    def _camera_entries(self, obs: Observation) -> list[tuple[str, object, object]]:
        return [
            ("left", obs.left_image, obs.left_camera_info),
            ("center", obs.center_image, obs.center_camera_info),
            ("right", obs.right_image, obs.right_camera_info),
        ]

    def _camera_info_for_name(self, obs: Observation, camera_name: str):
        name = str(camera_name).lower()
        if name == "left":
            return obs.left_camera_info
        if name == "center":
            return obs.center_camera_info
        if name == "right":
            return obs.right_camera_info
        raise ValueError(f"Unknown camera name {camera_name!r}")

    def _motion_camera_frame(self, obs: Observation) -> str:
        info = self._camera_info_for_name(obs, self.MOTION_CAMERA_NAME)
        frame = str(info.header.frame_id)
        if not frame:
            raise RuntimeError(f"{self.MOTION_CAMERA_NAME} CameraInfo has empty frame_id")
        return frame

    def _project_points(
        self,
        object_points: np.ndarray,
        T_base_ref: np.ndarray,
        info,
        image_shape: tuple[int, int],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, np.ndarray]:
        cam_frame = str(info.header.frame_id)
        if not cam_frame:
            raise RuntimeError("CameraInfo header.frame_id is empty")
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
        H, W = image_shape[:2]
        visible = np.array(
            [bool(pc[2] > 0.0 and 0 <= uv[0] < W and 0 <= uv[1] < H) for uv, pc in zip(pixels, pts_cam)],
            dtype=bool,
        )
        return pixels, pts_cam, visible, cam_frame, T_base_cam

    # ------------------------------------------------------------------
    # Motion helpers
    # ------------------------------------------------------------------

    def _diag36(self, values: list[float]) -> list[float]:
        return np.diag(values).flatten().astype(float).tolist()

    def _look_at_camera_rotation(self, p_cam: np.ndarray, p_target: np.ndarray, preferred_up: np.ndarray) -> np.ndarray:
        """Return R_base_cam for a ROS optical camera frame.

        Optical frame convention: +Z looks forward, +X right, +Y down.
        We aim +Z at p_target and keep the image approximately upright.
        """
        z_fwd = p_target - p_cam
        z_fwd = z_fwd / max(float(np.linalg.norm(z_fwd)), 1e-9)

        up = np.asarray(preferred_up, dtype=float)
        up = up / max(float(np.linalg.norm(up)), 1e-9)
        y_down = -up
        y_down = y_down - float(np.dot(y_down, z_fwd)) * z_fwd
        if float(np.linalg.norm(y_down)) < 1e-6:
            y_down = np.array([0.0, 1.0, 0.0], dtype=float)
            y_down = y_down - float(np.dot(y_down, z_fwd)) * z_fwd
        y_down = y_down / max(float(np.linalg.norm(y_down)), 1e-9)

        x_right = np.cross(y_down, z_fwd)
        x_right = x_right / max(float(np.linalg.norm(x_right)), 1e-9)
        y_down = np.cross(z_fwd, x_right)
        y_down = y_down / max(float(np.linalg.norm(y_down)), 1e-9)
        return np.column_stack([x_right, y_down, z_fwd])

    def _motion_mode(self):
        if self.PREFER_VELOCITY_MODE and hasattr(TrajectoryGenerationMode, "MODE_VELOCITY"):
            return TrajectoryGenerationMode(mode=TrajectoryGenerationMode.MODE_VELOCITY)
        return TrajectoryGenerationMode(mode=TrajectoryGenerationMode.MODE_POSITION)

    def _make_motion_cmd(self, pose: Pose, v_base: np.ndarray) -> MotionUpdate:
        msg = MotionUpdate()
        msg.header = Header(frame_id="base_link", stamp=self.get_clock().now().to_msg())
        msg.pose = pose
        msg.velocity = Twist(
            linear=Vector3(x=float(v_base[0]), y=float(v_base[1]), z=float(v_base[2])),
            angular=Vector3(x=0.0, y=0.0, z=0.0),
        )
        msg.target_stiffness = self._diag36(self.FREE_SPACE_STIFFNESS)
        msg.target_damping = self._diag36(self.FREE_SPACE_DAMPING)
        msg.feedforward_wrench_at_tip = Wrench(force=Vector3(), torque=Vector3())
        msg.wrench_feedback_gains_at_tip = [0.2, 0.2, 0.2, 0.0, 0.0, 0.0]
        msg.trajectory_generation_mode = self._motion_mode()
        return msg

    def _current_camera_offset_ref(self, T_base_ref: np.ndarray, T_base_cam: np.ndarray) -> np.ndarray:
        p_ref = np.linalg.inv(T_base_ref) @ np.array([T_base_cam[0, 3], T_base_cam[1, 3], T_base_cam[2, 3], 1.0])
        return p_ref[:3].astype(float)

    def _send_manual_velocity_command(
        self,
        obs: Observation,
        task: Task,
        object_name: str,
        move_robot: MoveRobotCallback,
    ) -> tuple[np.ndarray, str, np.ndarray]:
        """Send one teleop control tick.

        The velocity vector is expressed in the reference frame. We compute a
        desired center-camera pose that integrates this velocity while always
        looking at the target reference origin. Then convert desired camera pose
        to desired TCP pose with the current rigid TCP->camera transform.
        """
        ref_frame, T_base_ref = self._resolve_reference_transform(task, object_name)
        cam_frame = self._motion_camera_frame(obs)
        T_base_cam_now = self._base_T_frame(cam_frame)
        T_base_tcp_now = self._base_T_frame("gripper/tcp")
        T_tcp_cam = np.linalg.inv(T_base_tcp_now) @ T_base_cam_now

        offset_ref_now = self._current_camera_offset_ref(T_base_ref, T_base_cam_now)
        speed = self.FAST_LINEAR_SPEED if self._fast else self.SLOW_LINEAR_SPEED
        v_ref = self._active_velocity_ref.astype(float) * speed
        offset_ref_des = offset_ref_now + v_ref * self.CONTROL_DT

        p_cam_des = (T_base_ref @ np.array([offset_ref_des[0], offset_ref_des[1], offset_ref_des[2], 1.0]))[:3]
        p_target = T_base_ref[:3, 3]
        preferred_up = T_base_ref[:3, 1]
        R_base_cam_des = self._look_at_camera_rotation(p_cam_des, p_target, preferred_up)

        T_base_cam_des = np.eye(4, dtype=float)
        T_base_cam_des[:3, :3] = R_base_cam_des
        T_base_cam_des[:3, 3] = p_cam_des
        T_base_tcp_des = T_base_cam_des @ np.linalg.inv(T_tcp_cam)

        v_base = T_base_ref[:3, :3] @ v_ref
        cmd = self._make_motion_cmd(self._mat_to_pose(T_base_tcp_des), v_base)
        try:
            move_robot(motion_update=cmd)
        except Exception as ex:
            self.get_logger().warn(f"{C.YELLOW}[MOVE]{C.RESET} move_robot failed: {ex}")

        return offset_ref_now, ref_frame, T_base_ref

    # ------------------------------------------------------------------
    # UI drawing and callbacks
    # ------------------------------------------------------------------

    def _resize_keep_aspect(self, img: np.ndarray, width: int) -> np.ndarray:
        h, w = img.shape[:2]
        if w <= 0 or h <= 0:
            return img
        scale = width / float(w)
        return cv2.resize(img, (width, int(round(h * scale))), interpolation=cv2.INTER_AREA)

    def _draw_keypoints_on_image(
        self,
        img: np.ndarray,
        object_name: str,
        T_base_ref: np.ndarray,
        cam_info,
    ) -> tuple[np.ndarray, int, int]:
        all_names = self._all_keypoint_names(object_name)
        export_names = self._export_keypoint_names(object_name)
        export_set = set(export_names)
        pts = self._object_points(object_name, all_names)
        try:
            pixels, _pts_cam, visible, _cam_frame, _T_base_cam = self._project_points(pts, T_base_ref, cam_info, img.shape)
        except Exception:
            return img.copy(), 0, len(export_names)

        out = img.copy()
        export_visible = 0
        for i, (name, uv, vis) in enumerate(zip(all_names, pixels, visible)):
            if name in export_set and vis:
                export_visible += 1
            u, v = float(uv[0]), float(uv[1])
            if not math.isfinite(u) or not math.isfinite(v):
                continue
            if not (-100 <= u <= img.shape[1] + 100 and -100 <= v <= img.shape[0] + 100):
                continue
            xy = (int(round(u)), int(round(v)))
            if name in export_set:
                color = (0, 255, 255) if vis else (90, 90, 90)
                cv2.circle(out, xy, 5, color, -1 if vis else 1, lineType=cv2.LINE_AA)
            else:
                color = (255, 0, 255)
                cv2.drawMarker(out, xy, color, markerType=cv2.MARKER_CROSS, markerSize=12, thickness=2)
            cv2.putText(out, str(i), (xy[0] + 6, xy[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
        return out, export_visible, len(export_names)

    def _button(self, img: np.ndarray, key: str, text: str, x: int, y: int, w: int, h: int, active: bool = False) -> int:
        color = (60, 160, 70) if active else (65, 65, 65)
        if key in {"save", "quit", "stop"}:
            color = {"save": (45, 130, 210), "quit": (50, 50, 180), "stop": (50, 50, 210)}[key]
        cv2.rectangle(img, (x, y), (x + w, y + h), color, -1)
        cv2.rectangle(img, (x, y), (x + w, y + h), (230, 230, 230), 1)
        cv2.putText(img, text, (x + 8, y + int(h * 0.65)), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
        self._buttons[key] = (x, y, x + w, y + h)
        return x + w + self.BUTTON_MARGIN

    def _compose_preview(
        self,
        obs: Observation,
        task: Task,
        object_name: str,
        ref_frame: str,
        T_base_ref: np.ndarray,
        offset_ref: np.ndarray,
        saved_count: int,
    ) -> np.ndarray:
        cams = []
        vis_lines = []
        for cam_name, image_msg, info in self._camera_entries(obs):
            img = self._image_msg_to_bgr(image_msg)
            over, nvis, ntotal = self._draw_keypoints_on_image(img, object_name, T_base_ref, info)
            over = self._resize_keep_aspect(over, self.PREVIEW_CAMERA_WIDTH)
            cv2.putText(over, f"{cam_name}: {nvis}/{ntotal}", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
            cams.append(over)
            vis_lines.append(f"{cam_name}={nvis}/{ntotal}")

        # Pad to common height and stack horizontally.
        max_h = max(c.shape[0] for c in cams)
        padded = []
        for c in cams:
            if c.shape[0] < max_h:
                pad = np.zeros((max_h - c.shape[0], c.shape[1], 3), dtype=np.uint8)
                c = np.vstack([c, pad])
            padded.append(c)
        top = np.hstack(padded)
        W = top.shape[1]
        panel_h = 172
        panel = np.zeros((panel_h, W, 3), dtype=np.uint8)
        self._buttons.clear()

        status1 = (
            f"{object_name.upper()} saved {saved_count}/{self.TARGET_VIEWPOINTS_PER_OBJECT} | "
            f"ref={ref_frame} | {'FAST' if self._fast else 'SLOW'} | "
            f"offset_ref=[{offset_ref[0]:+.4f}, {offset_ref[1]:+.4f}, {offset_ref[2]:+.4f}] m"
        )
        status2 = "Velocity is in entrance/reference frame. Camera always looks at entrance. Hold buttons or keys to move."
        status3 = "Keys: H/L=X-/X+, J/K=Y-/Y+, U/I=Z-/Z+, F=fast, SPACE=stop, S=save, C=png, Q=quit"
        cv2.putText(panel, status1[:180], (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(panel, status2, (12, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (210, 210, 210), 1, cv2.LINE_AA)
        cv2.putText(panel, status3, (12, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (210, 210, 210), 1, cv2.LINE_AA)
        cv2.putText(panel, "Visibility: " + " | ".join(vis_lines), (12, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (180, 220, 255), 1, cv2.LINE_AA)

        x = 12
        y = 118
        bw = 62
        bh = self.BUTTON_H
        x = self._button(panel, "x-", "X-", x, y, bw, bh, self._active_velocity_ref[0] < 0)
        x = self._button(panel, "x+", "X+", x, y, bw, bh, self._active_velocity_ref[0] > 0)
        x = self._button(panel, "y-", "Y-", x, y, bw, bh, self._active_velocity_ref[1] < 0)
        x = self._button(panel, "y+", "Y+", x, y, bw, bh, self._active_velocity_ref[1] > 0)
        x = self._button(panel, "z-", "Z-", x, y, bw, bh, self._active_velocity_ref[2] < 0)
        x = self._button(panel, "z+", "Z+", x, y, bw, bh, self._active_velocity_ref[2] > 0)
        x = self._button(panel, "stop", "STOP", x, y, 78, bh)
        x = self._button(panel, "fast", "FAST", x, y, 74, bh, self._fast)
        x = self._button(panel, "save", "SAVE", x, y, 78, bh)
        x = self._button(panel, "png", "PNG", x, y, 68, bh)
        x = self._button(panel, "quit", "QUIT", x, y, 74, bh)

        return np.vstack([top, panel])

    def _set_velocity_for_button(self, button: Optional[str]) -> None:
        self._active_velocity_ref[:] = 0.0
        if button == "x-":
            self._active_velocity_ref[0] = -1.0
        elif button == "x+":
            self._active_velocity_ref[0] = 1.0
        elif button == "y-":
            self._active_velocity_ref[1] = -1.0
        elif button == "y+":
            self._active_velocity_ref[1] = 1.0
        elif button == "z-":
            self._active_velocity_ref[2] = -1.0
        elif button == "z+":
            self._active_velocity_ref[2] = 1.0

    def _handle_button_click(self, button: str, is_down: bool) -> None:
        if button in {"x-", "x+", "y-", "y+", "z-", "z+"}:
            self._mouse_down_button = button if is_down else None
            self._set_velocity_for_button(self._mouse_down_button)
            return
        if not is_down:
            return
        if button == "stop":
            self._mouse_down_button = None
            self._active_velocity_ref[:] = 0.0
        elif button == "fast":
            self._fast = not self._fast
        elif button == "save":
            self._save_requested = True
        elif button == "png":
            self._save_png_requested = True
        elif button == "quit":
            self._quit_requested = True

    def _mouse_cb(self, event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN:
            for key, (x1, y1, x2, y2) in self._buttons.items():
                if x1 <= x <= x2 and y1 <= y <= y2:
                    self._handle_button_click(key, True)
                    break
        elif event == cv2.EVENT_LBUTTONUP:
            if self._mouse_down_button is not None:
                self._handle_button_click(self._mouse_down_button, False)
            else:
                self._active_velocity_ref[:] = 0.0
        elif event == cv2.EVENT_MOUSEMOVE:
            # If the user drags off a velocity button while holding, keep moving.
            pass

    def _handle_key(self, key: int) -> None:
        if key < 0:
            return
        k = key & 0xFF
        # For keyboard teleop we send one velocity pulse. Key repeat from OS will
        # make this feel continuous if the key is held.
        self._active_velocity_ref[:] = 0.0
        if k in (ord("h"), ord("H")):
            self._active_velocity_ref[0] = -1.0
        elif k in (ord("l"), ord("L")):
            self._active_velocity_ref[0] = 1.0
        elif k in (ord("j"), ord("J")):
            self._active_velocity_ref[1] = -1.0
        elif k in (ord("k"), ord("K")):
            self._active_velocity_ref[1] = 1.0
        elif k in (ord("u"), ord("U")):
            self._active_velocity_ref[2] = -1.0
        elif k in (ord("i"), ord("I")):
            self._active_velocity_ref[2] = 1.0
        elif k in (ord("f"), ord("F")):
            self._fast = not self._fast
        elif k == ord(" "):
            self._active_velocity_ref[:] = 0.0
            self._mouse_down_button = None
        elif k in (ord("s"), ord("S")):
            self._save_requested = True
        elif k in (ord("c"), ord("C")):
            self._save_png_requested = True
        elif k in (ord("q"), ord("Q"), 27):
            self._quit_requested = True

    # ------------------------------------------------------------------
    # Saving accepted viewpoint
    # ------------------------------------------------------------------

    def _save_current_viewpoint(
        self,
        object_name: str,
        task: Task,
        ref_frame: str,
        offset_ref: np.ndarray,
        viewpoints: list[dict],
    ) -> list[dict]:
        name = f"{object_name}_manual_{len(viewpoints):03d}"
        entry = {
            "name": name,
            "cam_offset_ref": [float(offset_ref[0]), float(offset_ref[1]), float(offset_ref[2])],
            "motion_camera_name": self.MOTION_CAMERA_NAME,
            "source": "manual_velocity_teleop",
            "reference_frame_during_calibration": ref_frame,
            "task_during_calibration": {
                "id": str(task.id),
                "port_type": str(task.port_type),
                "port_name": str(task.port_name),
                "target_module_name": str(task.target_module_name),
            },
            "timestamp_unix": float(time.time()),
        }
        viewpoints.append(entry)
        self._write_saved_viewpoints(object_name, viewpoints)
        self.get_logger().info(
            f"{C.GREEN}{C.BOLD}[SAVE]{C.RESET} accepted {name}: "
            f"cam_offset_ref={[round(float(x), 5) for x in offset_ref]}"
        )
        return viewpoints

    def _save_preview_png(self, object_name: str, count: int) -> None:
        if self._last_preview is None:
            return
        out_dir = self.view_out_dir / "manual_calibration_previews"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{object_name}_preview_{count:03d}_{int(time.time())}.png"
        cv2.imwrite(str(path), self._last_preview)
        self.get_logger().info(f"{C.GREEN}[PNG]{C.RESET} saved preview {path}")

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
        object_name = self._object_name_for_task(task)
        viewpoints = self._load_saved_viewpoints(object_name)
        send_feedback(f"Manual calibration for {object_name}; saved {len(viewpoints)}/{self.TARGET_VIEWPOINTS_PER_OBJECT}")
        self.get_logger().info(
            f"{C.MAGENTA}{C.BOLD}[TASK]{C.RESET} object={object_name} saved={len(viewpoints)}/{self.TARGET_VIEWPOINTS_PER_OBJECT} task={task}"
        )

        if len(viewpoints) >= self.TARGET_VIEWPOINTS_PER_OBJECT:
            self.get_logger().info(f"{C.GREEN}[DONE]{C.RESET} already have enough {object_name} viewpoints")
            return True

        cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.WINDOW_NAME, self._mouse_cb)

        self._running = True
        self._quit_requested = False
        self._save_requested = False
        self._save_png_requested = False
        self._active_velocity_ref[:] = 0.0
        self._mouse_down_button = None
        self._loop_count = 0

        last_offset = np.zeros(3, dtype=float)
        last_ref_frame = ""
        last_T_base_ref = np.eye(4, dtype=float)

        while self._running and not self._quit_requested:
            obs = get_observation()
            if obs is None:
                self.sleep_for(self.CONTROL_DT)
                continue

            try:
                last_offset, last_ref_frame, last_T_base_ref = self._send_manual_velocity_command(
                    obs, task, object_name, move_robot
                )
            except Exception as ex:
                self.get_logger().warn(f"{C.YELLOW}[LOOP]{C.RESET} motion tick failed: {ex}")
                self.sleep_for(self.CONTROL_DT)
                continue

            # Preview periodically, or immediately if user requested something.
            if (
                self._loop_count % self.PREVIEW_REFRESH_EVERY_N_LOOPS == 0
                or self._save_requested
                or self._save_png_requested
            ):
                try:
                    preview = self._compose_preview(
                        obs,
                        task,
                        object_name,
                        last_ref_frame,
                        last_T_base_ref,
                        last_offset,
                        len(viewpoints),
                    )
                    self._last_preview = preview
                    cv2.imshow(self.WINDOW_NAME, preview)
                except Exception as ex:
                    self.get_logger().warn(f"{C.YELLOW}[UI]{C.RESET} preview failed: {ex}")

            key = cv2.waitKey(1)
            self._handle_key(key)

            if self._save_png_requested:
                self._save_png_requested = False
                self._save_preview_png(object_name, len(viewpoints))

            if self._save_requested:
                self._save_requested = False
                viewpoints = self._save_current_viewpoint(
                    object_name,
                    task,
                    last_ref_frame,
                    last_offset,
                    viewpoints,
                )
                send_feedback(
                    f"Saved {object_name} viewpoint {len(viewpoints)}/{self.TARGET_VIEWPOINTS_PER_OBJECT}"
                )
                if len(viewpoints) >= self.TARGET_VIEWPOINTS_PER_OBJECT:
                    self.get_logger().info(
                        f"{C.GREEN}{C.BOLD}[DONE]{C.RESET} reached target {self.TARGET_VIEWPOINTS_PER_OBJECT} for {object_name}"
                    )
                    break

            self._loop_count += 1
            self.sleep_for(self.CONTROL_DT)

        self._active_velocity_ref[:] = 0.0
        self._mouse_down_button = None
        self._write_saved_viewpoints(object_name, viewpoints)
        try:
            cv2.destroyWindow(self.WINDOW_NAME)
        except Exception:
            pass
        return True