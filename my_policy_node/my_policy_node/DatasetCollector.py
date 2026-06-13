# DatasetCollector replay manual calibrated viewpoints v8 for AIC keypoint dataset generation.
from __future__ import annotations

import json
import math
import os
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
    """AIC automatic YOLO-pose dataset collector.

    Run the world with ground-truth enabled and the engine enabled:

        /entrypoint.sh ground_truth:=true start_aic_engine:=true \
          aic_engine_config_file:=/path/to/dataset_engine_config.yaml

    Then edit the settings near the top of this file and run:

        PYTHONPATH=$PWD/my_policy_node:$PYTHONPATH \
        pixi run ros2 run aic_model aic_model \
          --ros-args -p use_sim_time:=true -p policy:=my_policy_node.DatasetCollector

    It receives each engine task, replays your manually calibrated camera
    viewpoints around the target, and saves:
      - raw left/center/right images
      - YOLO-pose labels
      - overlay images for inspection
      - metadata JSON with frames/transforms

    It does NOT attempt insertion.
    """

    # ------------------------------------------------------------------
    # EDIT THESE SETTINGS HERE. No environment variables are required.
    # ------------------------------------------------------------------

    # Paths. Run the policy from ~/ws_aic/src/aic. Relative paths are resolved
    # relative to the current working directory.
    KEYPOINT_CONFIG_PATH = "my_policy_node/my_policy_node/perception/keypoint_configs/aic_keypoint_config.yaml"

    # Manual calibrated viewpoint YAMLs from DatasetCollector_manual_velocity_calib_v7.py.
    VIEWPOINT_DIR = "my_policy_node/my_policy_node/perception/keypoint_configs"
    SFP_VIEWPOINT_FILE = "calibrated_manual_viewpoints_sfp.yaml"
    SC_VIEWPOINT_FILE = "calibrated_manual_viewpoints_sc.yaml"

    # Dataset output. Change RUN_NAME each time you generate a new dataset.
    DATASET_ROOT = "datasets/aic_keypoint_manual40"
    RUN_NAME = "manual40_trials40_v1"

    # Collection behavior.
    DRY_RUN = False                 # True = do not move robot, only save current camera pose
    FRAMES_PER_VIEW = 1             # save N frames at each calibrated viewpoint
    MOVE_DT = 0.05                  # seconds between MotionUpdate commands
    MOVE_STEPS = 95                 # command duration per viewpoint; 95*0.05 = 4.75 sec
    SETTLE_STEPS = 18               # wait after motion before saving; 18*0.05 = 0.9 sec

    # Label filtering.
    MIN_VISIBLE_KEYPOINTS = 4
    BBOX_MARGIN_PX = 10.0

    # IMPORTANT: manual calibration already handled occlusion/visibility. Do not
    # run the old automatic gripper/plug occlusion filter because it was too
    # conservative for this scene and skipped good images.
    OCCLUSION_FILTER_ENABLED = False
    TOOL_OCCLUSION_MARGIN_PX = 0.0
    MIN_PLUG_DISTANCE_FROM_REFERENCE_M = 0.0

    # This should match the camera you teleoperated during calibration.
    MOTION_CAMERA_NAME = "center"

    FREE_SPACE_STIFFNESS = [45.0, 45.0, 45.0, 28.0, 28.0, 28.0]
    FREE_SPACE_DAMPING = [40.0, 40.0, 40.0, 18.0, 18.0, 18.0]

    def __init__(self, parent_node):
        super().__init__(parent_node)
        self.config_path = Path(self.KEYPOINT_CONFIG_PATH).expanduser()
        self.dataset_root = Path(self.DATASET_ROOT).expanduser()
        self.run_name = self.RUN_NAME or time.strftime("run_%Y%m%d_%H%M%S")
        self.out_root = self.dataset_root / self.run_name

        self.config = self._load_config(self.config_path)
        self._trial_index = 0
        self._saved = 0
        self._skipped = 0

        self.viewpoint_dir = Path(self.VIEWPOINT_DIR).expanduser()
        self._ensure_dirs()
        self.get_logger().info(
            f"{C.CYAN}{C.BOLD}[INIT]{C.RESET} DatasetCollector config={self.config_path} out={self.out_root}"
        )
        self.get_logger().info(
            f"{C.CYAN}[INIT]{C.RESET} DRY_RUN={self.DRY_RUN} manual_viewpoints_dir={self.viewpoint_dir} "
            f"frames/view={self.FRAMES_PER_VIEW} motion_camera={self.MOTION_CAMERA_NAME} "
            f"occlusion_filter={self.OCCLUSION_FILTER_ENABLED}"
        )

    # ------------------------------------------------------------------
    # Config and dataset layout
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

    def _ensure_dirs(self) -> None:
        for obj_name in self.config.get("objects", {}).keys():
            for split in ["train", "val"]:
                for sub in ["images", "labels", "overlays", "metadata"]:
                    (self.out_root / obj_name / sub / split).mkdir(parents=True, exist_ok=True)

    def _object_name_for_task(self, task: Task) -> str:
        pt = str(task.port_type).lower()
        if pt == "sc":
            return "sc"
        if pt == "sfp":
            return "sfp"
        raise ValueError(f"Unsupported task.port_type={task.port_type!r}; expected sc or sfp")

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
        return self._format_frame_template(obj.get("fallback_reference_frame_template", obj["reference_frame_template"]), task, object_name)

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

    def _write_dataset_yaml(self, object_name: str) -> None:
        export_names = self._export_keypoint_names(object_name)
        display_name = "sc_module" if object_name == "sc" else "sfp_nic_front"
        data = {
            "path": str((self.out_root / object_name).resolve()),
            "train": "images/train",
            "val": "images/val",
            "kpt_shape": [len(export_names), 3],
            "names": {0: display_name},
            "keypoint_names": export_names,
        }
        with (self.out_root / object_name / f"{object_name}_pose.yaml").open("w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=False)

    def _trial_split(self) -> str:
        # Split by trial, not by individual image. Every 5th trial goes to val.
        return "val" if (self._trial_index % 5 == 0) else "train"

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
    # Image and projection
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

    def _camera_info_for_name(self, obs: Observation, camera_name: str):
        camera_name = str(camera_name).lower()
        if camera_name == "left":
            return obs.left_camera_info
        if camera_name == "center":
            return obs.center_camera_info
        if camera_name == "right":
            return obs.right_camera_info
        raise ValueError(f"Unknown camera name {camera_name!r}; expected left/center/right")

    def _frame_name_for_motion_camera(self, get_observation: GetObservationCallback) -> str:
        obs = get_observation()
        if obs is None:
            raise RuntimeError("No observation available for camera frame lookup")
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

    def _project_base_point_to_image(self, p_base: np.ndarray, info, image_shape: tuple[int, int]) -> tuple[Optional[np.ndarray], bool]:
        """Project a base-frame 3D point into an image. Returns (uv, in_front_and_inside)."""
        cam_frame = str(info.header.frame_id)
        if not cam_frame:
            return None, False
        T_base_cam = self._base_T_frame(cam_frame)
        T_cam_base = np.linalg.inv(T_base_cam)
        p_h = np.array([float(p_base[0]), float(p_base[1]), float(p_base[2]), 1.0], dtype=float)
        p_cam = T_cam_base @ p_h
        if p_cam[2] <= 1e-6:
            return None, False
        K = np.asarray(info.k, dtype=np.float64).reshape(3, 3)
        uv_h = K @ p_cam[:3]
        uv = np.array([uv_h[0] / uv_h[2], uv_h[1] / uv_h[2]], dtype=float)
        H, W = image_shape[:2]
        inside = bool(0 <= uv[0] < W and 0 <= uv[1] < H)
        return uv, inside

    def _bbox_from_visible_pixels(self, pixels: np.ndarray, visible: np.ndarray, image_shape: tuple[int, int], margin_px: float) -> Optional[tuple[float, float, float, float]]:
        if int(np.count_nonzero(visible)) < self.MIN_VISIBLE_KEYPOINTS:
            return None
        H, W = image_shape[:2]
        vis_pixels = pixels[visible]
        xmin = max(0.0, float(np.min(vis_pixels[:, 0]) - margin_px))
        xmax = min(float(W - 1), float(np.max(vis_pixels[:, 0]) + margin_px))
        ymin = max(0.0, float(np.min(vis_pixels[:, 1]) - margin_px))
        ymax = min(float(H - 1), float(np.max(vis_pixels[:, 1]) + margin_px))
        if xmax <= xmin or ymax <= ymin:
            return None
        return xmin, ymin, xmax, ymax

    def _tool_likely_occludes_target(self, task: Task, info, image_shape: tuple[int, int], bbox: tuple[float, float, float, float], T_base_ref: np.ndarray) -> tuple[bool, str]:
        """Conservative approximate occlusion filter.

        We cannot perfectly verify visibility from RGB alone. This filter catches
        common bad cases where the held plug/TCP is close to the target in 3D or
        projects into the target label box. This prevents many geometrically
        projected-but-actually-occluded labels.
        """
        if not self.OCCLUSION_FILTER_ENABLED:
            return False, "disabled"

        xmin, ymin, xmax, ymax = bbox
        xmin -= self.TOOL_OCCLUSION_MARGIN_PX
        ymin -= self.TOOL_OCCLUSION_MARGIN_PX
        xmax += self.TOOL_OCCLUSION_MARGIN_PX
        ymax += self.TOOL_OCCLUSION_MARGIN_PX

        p_ref = T_base_ref[:3, 3]
        try:
            T_base_plug = self._base_T_frame(f"{task.cable_name}/{task.plug_name}_link")
            p_plug = T_base_plug[:3, 3]
            plug_dist = float(np.linalg.norm(p_plug - p_ref))
            if plug_dist < self.MIN_PLUG_DISTANCE_FROM_REFERENCE_M:
                return True, f"plug_close_3d={plug_dist:.3f}m"
        except Exception:
            pass

        for frame in ["gripper/tcp", f"{task.cable_name}/{task.plug_name}_link"]:
            try:
                T_base_frame = self._base_T_frame(frame)
                uv, inside = self._project_base_point_to_image(T_base_frame[:3, 3], info, image_shape)
                if uv is not None and inside:
                    if xmin <= uv[0] <= xmax and ymin <= uv[1] <= ymax:
                        return True, f"{frame}_projects_near_label_box"
            except Exception:
                continue

        return False, "clear"

    def _make_yolo_label(
        self,
        class_id: int,
        pixels: np.ndarray,
        visible: np.ndarray,
        image_shape: tuple[int, int],
    ) -> Optional[str]:
        H, W = image_shape[:2]
        bbox = self._bbox_from_visible_pixels(pixels, visible, image_shape, self.BBOX_MARGIN_PX)
        if bbox is None:
            return None
        xmin, ymin, xmax, ymax = bbox

        xc = ((xmin + xmax) / 2.0) / W
        yc = ((ymin + ymax) / 2.0) / H
        bw = (xmax - xmin) / W
        bh = (ymax - ymin) / H
        vals = [str(int(class_id)), f"{xc:.8f}", f"{yc:.8f}", f"{bw:.8f}", f"{bh:.8f}"]

        for (u, v), is_vis in zip(pixels, visible):
            if is_vis:
                vals += [f"{u / W:.8f}", f"{v / H:.8f}", "2"]
            else:
                vals += ["0.00000000", "0.00000000", "0"]
        return " ".join(vals) + "\n"

    def _draw_overlay(
        self,
        img: np.ndarray,
        all_names: list[str],
        all_pixels: np.ndarray,
        all_visible: np.ndarray,
        export_names: list[str],
        ref_frame: str,
        cam_name: str,
        task: Task,
    ) -> np.ndarray:
        out = img.copy()
        H, W = out.shape[:2]
        export_set = set(export_names)
        for i, (name, uv, is_vis) in enumerate(zip(all_names, all_pixels, all_visible)):
            u, v = float(uv[0]), float(uv[1])
            if not math.isfinite(u) or not math.isfinite(v):
                continue
            if not (-100 <= u <= W + 100 and -100 <= v <= H + 100):
                continue
            xy = (int(round(u)), int(round(v)))
            if name in export_set:
                color = (0, 255, 255) if is_vis else (120, 120, 120)
                radius = 5
            else:
                # Derived/debug points like centers: draw magenta cross, not a filled training point.
                color = (255, 0, 255)
                radius = 6
            if name in export_set:
                cv2.circle(out, xy, radius, color, -1 if is_vis else 1, lineType=cv2.LINE_AA)
            else:
                cv2.drawMarker(out, xy, color, markerType=cv2.MARKER_CROSS, markerSize=12, thickness=2)
            cv2.putText(out, str(i), (xy[0] + 6, xy[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)

        cv2.rectangle(out, (0, 0), (W, 78), (0, 0, 0), -1)
        header = f"{cam_name} task={task.id} {task.port_type}/{task.port_name} ref={ref_frame}"
        cv2.putText(out, header[:130], (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(out, "yellow=corners used for YOLO | magenta=center/debug excluded", (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (210, 210, 210), 1, cv2.LINE_AA)
        return out

    # ------------------------------------------------------------------
    # Robot movement for viewpoints
    # ------------------------------------------------------------------

    def _viewpoint_file_for_object(self, object_name: str) -> Path:
        if object_name == "sfp":
            return self.viewpoint_dir / self.SFP_VIEWPOINT_FILE
        if object_name == "sc":
            return self.viewpoint_dir / self.SC_VIEWPOINT_FILE
        raise ValueError(f"Unsupported object_name={object_name!r}")

    def _load_viewpoints_for_object(self, object_name: str) -> list[dict]:
        path = self._viewpoint_file_for_object(object_name)
        if not path.exists():
            raise FileNotFoundError(
                f"Missing calibrated viewpoint file for {object_name}: {path}. "
                "Run the manual calibration policy first and save 40 viewpoints."
            )
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        viewpoints = data.get("viewpoints", data if isinstance(data, list) else [])
        clean: list[dict] = []
        for i, view in enumerate(viewpoints):
            if not isinstance(view, dict):
                continue
            if "cam_offset_ref" not in view:
                continue
            v = dict(view)
            v.setdefault("name", f"{object_name}_manual_{i:03d}")
            v.setdefault("motion_camera_name", self.MOTION_CAMERA_NAME)
            clean.append(v)
        if not clean:
            raise RuntimeError(f"No valid viewpoints with cam_offset_ref found in {path}")
        return clean

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

    def _look_at_camera_rotation(self, p_cam: np.ndarray, p_target: np.ndarray, preferred_up: np.ndarray) -> np.ndarray:
        """Return R_base_cam for a ROS optical camera frame.

        Optical frame convention: +Z looks forward, +X right, +Y down.
        We aim +Z at p_target and keep the image roughly upright using
        preferred_up when possible.
        """
        z_fwd = p_target - p_cam
        z_norm = float(np.linalg.norm(z_fwd))
        if z_norm < 1e-9:
            z_fwd = np.array([0.0, 0.0, 1.0], dtype=float)
        else:
            z_fwd = z_fwd / z_norm

        up = np.asarray(preferred_up, dtype=float)
        up = up / max(float(np.linalg.norm(up)), 1e-9)
        # Optical +Y points down, so use negative world/ref up as the first guess.
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

        R = np.column_stack([x_right, y_down, z_fwd])
        return R

    def _desired_tcp_pose_for_viewpoint(
        self,
        task: Task,
        object_name: str,
        view: dict,
        get_observation: GetObservationCallback,
    ) -> Pose:
        ref_frame, T_base_ref = self._resolve_reference_transform(task, object_name)
        cam_frame = self._frame_name_for_motion_camera(get_observation)
        T_base_cam_now = self._base_T_frame(cam_frame)
        T_base_tcp_now = self._base_T_frame("gripper/tcp")
        T_tcp_cam = np.linalg.inv(T_base_tcp_now) @ T_base_cam_now

        offset_ref = np.asarray(view.get("cam_offset_ref", [0.0, 0.0, 0.25]), dtype=float)
        p_target = T_base_ref[:3, 3].copy()
        p_cam_des = (T_base_ref @ np.array([offset_ref[0], offset_ref[1], offset_ref[2], 1.0], dtype=float))[:3]

        # Prefer the reference frame's local Y as image-up reference. This keeps
        # orientation stable across board yaw while still looking at the target.
        preferred_up = T_base_ref[:3, 1]
        R_base_cam_des = self._look_at_camera_rotation(p_cam_des, p_target, preferred_up)

        T_base_cam_des = np.eye(4, dtype=float)
        T_base_cam_des[:3, :3] = R_base_cam_des
        T_base_cam_des[:3, 3] = p_cam_des

        # Convert desired camera pose to desired TCP pose using the current rigid
        # TCP->camera transform.
        T_base_tcp_des = T_base_cam_des @ np.linalg.inv(T_tcp_cam)
        return self._mat_to_pose(T_base_tcp_des)

    def _move_to_viewpoint(
        self,
        task: Task,
        object_name: str,
        view: dict,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
    ) -> None:
        if self.DRY_RUN:
            return
        pose = self._desired_tcp_pose_for_viewpoint(task, object_name, view, get_observation)
        cmd = self._make_pose_cmd(pose)
        self.get_logger().info(
            f"{C.BLUE}[MOVE]{C.RESET} view={view.get('name')} camera={self.MOTION_CAMERA_NAME} "
            f"cam_offset_ref={view.get('cam_offset_ref')}"
        )
        for _ in range(self.MOVE_STEPS):
            try:
                move_robot(motion_update=cmd)
            except Exception as ex:
                self.get_logger().error(f"{C.RED}[MOVE]{C.RESET} move_robot failed: {ex}")
                break
            self.sleep_for(self.MOVE_DT)
        for _ in range(self.SETTLE_STEPS):
            self.sleep_for(self.MOVE_DT)

    # ------------------------------------------------------------------
    # Saving
    # ------------------------------------------------------------------

    def _sample_prefix(self, task: Task, split: str, view_name: str, cam_name: str, frame_idx: int) -> str:
        safe_task = str(task.id).replace("/", "_").replace(" ", "_")
        return f"trial{self._trial_index:03d}_{safe_task}_{view_name}_{cam_name}_{frame_idx:02d}"

    def _save_one_camera(
        self,
        object_name: str,
        split: str,
        task: Task,
        view: dict,
        frame_idx: int,
        cam_name: str,
        image_msg,
        info,
        ref_frame: str,
        T_base_ref: np.ndarray,
    ) -> bool:
        img = self._image_msg_to_bgr(image_msg)
        H, W = img.shape[:2]
        all_names = self._all_keypoint_names(object_name)
        export_names = self._export_keypoint_names(object_name)
        class_id = int(self.config["objects"][object_name].get("class_id", 0))

        all_points = self._object_points(object_name, all_names)
        export_points = self._object_points(object_name, export_names)
        try:
            all_pixels, all_pts_cam, all_visible, cam_frame, T_base_cam = self._project_points(all_points, T_base_ref, info, img.shape)
            export_pixels, export_pts_cam, export_visible, _, _ = self._project_points(export_points, T_base_ref, info, img.shape)
        except Exception as ex:
            self.get_logger().warn(f"{C.YELLOW}[SKIP]{C.RESET} projection failed {cam_name}: {ex}")
            self._skipped += 1
            return False

        bbox_for_filter = self._bbox_from_visible_pixels(export_pixels, export_visible, img.shape, self.BBOX_MARGIN_PX)
        if bbox_for_filter is None:
            self.get_logger().info(
                f"{C.YELLOW}[SKIP]{C.RESET} {object_name}/{cam_name} visible={int(np.count_nonzero(export_visible))}/{len(export_names)}"
            )
            self._skipped += 1
            return False

        occluded, occ_reason = self._tool_likely_occludes_target(task, info, img.shape, bbox_for_filter, T_base_ref)
        if occluded:
            self.get_logger().info(
                f"{C.YELLOW}[SKIP]{C.RESET} {object_name}/{cam_name} likely occluded: {occ_reason}"
            )
            self._skipped += 1
            return False

        label = self._make_yolo_label(class_id, export_pixels, export_visible, img.shape)
        if label is None:
            self.get_logger().info(
                f"{C.YELLOW}[SKIP]{C.RESET} {object_name}/{cam_name} visible={int(np.count_nonzero(export_visible))}/{len(export_names)}"
            )
            self._skipped += 1
            return False

        prefix = self._sample_prefix(task, split, str(view.get("name", "view")), cam_name, frame_idx)
        root = self.out_root / object_name
        img_path = root / "images" / split / f"{prefix}.png"
        label_path = root / "labels" / split / f"{prefix}.txt"
        overlay_path = root / "overlays" / split / f"{prefix}_overlay.png"
        meta_path = root / "metadata" / split / f"{prefix}.json"

        overlay = self._draw_overlay(img, all_names, all_pixels, all_visible, export_names, ref_frame, cam_name, task)
        cv2.imwrite(str(img_path), img)
        label_path.write_text(label, encoding="utf-8")
        cv2.imwrite(str(overlay_path), overlay)
        metadata = {
            "object_name": object_name,
            "split": split,
            "trial_index": self._trial_index,
            "task": {
                "id": task.id,
                "cable_type": task.cable_type,
                "cable_name": task.cable_name,
                "plug_type": task.plug_type,
                "plug_name": task.plug_name,
                "port_type": task.port_type,
                "port_name": task.port_name,
                "target_module_name": task.target_module_name,
                "time_limit": int(task.time_limit),
            },
            "view": view,
            "camera": cam_name,
            "camera_frame": cam_frame,
            "reference_frame": ref_frame,
            "image": str(img_path),
            "label": str(label_path),
            "overlay": str(overlay_path),
            "exported_keypoints": export_names,
            "all_keypoints": all_names,
            "export_visible": [bool(x) for x in export_visible.tolist()],
            "T_base_ref": T_base_ref.tolist(),
            "T_base_cam": T_base_cam.tolist(),
            "occlusion_filter_enabled": bool(self.OCCLUSION_FILTER_ENABLED),
            "tool_occlusion_margin_px": float(self.TOOL_OCCLUSION_MARGIN_PX),
            "min_plug_distance_from_reference_m": float(self.MIN_PLUG_DISTANCE_FROM_REFERENCE_M),
        }
        meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        self._saved += 1
        return True

    def _collect_at_current_pose(self, task: Task, object_name: str, split: str, view: dict, frame_idx: int, get_observation: GetObservationCallback) -> int:
        obs = get_observation()
        if obs is None:
            return 0
        try:
            ref_frame, T_base_ref = self._resolve_reference_transform(task, object_name)
        except Exception as ex:
            self.get_logger().error(f"{C.RED}[TF]{C.RESET} cannot resolve reference frame: {ex}")
            return 0

        count = 0
        for cam_name, image_msg, info in self._camera_entries(obs):
            if self._save_one_camera(object_name, split, task, view, frame_idx, cam_name, image_msg, info, ref_frame, T_base_ref):
                count += 1
        return count

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
        self._trial_index += 1
        split = self._trial_split()
        object_name = self._object_name_for_task(task)
        self._write_dataset_yaml(object_name)

        self.get_logger().info(
            f"{C.MAGENTA}{C.BOLD}[TASK]{C.RESET} trial={self._trial_index} split={split} object={object_name} task={task}"
        )
        send_feedback(f"DatasetCollector collecting {object_name} trial {self._trial_index} split={split}")

        export_names = self._export_keypoint_names(object_name)
        self.get_logger().info(f"{C.CYAN}[KEYPOINTS]{C.RESET} exported {len(export_names)}: {export_names}")

        try:
            viewpoints = self._load_viewpoints_for_object(object_name)
        except Exception as ex:
            self.get_logger().error(f"{C.RED}[VIEWPOINTS]{C.RESET} {ex}")
            send_feedback(f"DatasetCollector missing viewpoints for {object_name}: {ex}")
            return False

        self.get_logger().info(
            f"{C.CYAN}[VIEWPOINTS]{C.RESET} loaded {len(viewpoints)} manual {object_name} viewpoints "
            f"from {self._viewpoint_file_for_object(object_name)}"
        )

        trial_saved_before = self._saved
        for view in viewpoints:
            try:
                self._move_to_viewpoint(task, object_name, view, get_observation, move_robot)
            except Exception as ex:
                self.get_logger().warn(f"{C.YELLOW}[MOVE]{C.RESET} view move failed, collecting anyway: {ex}")

            for frame_idx in range(self.FRAMES_PER_VIEW):
                saved_here = self._collect_at_current_pose(task, object_name, split, view, frame_idx, get_observation)
                self.get_logger().info(
                    f"{C.GREEN if saved_here else C.YELLOW}[SAVE]{C.RESET} view={view.get('name')} frame={frame_idx} saved_cameras={saved_here} total={self._saved} skipped={self._skipped}"
                )
                self.sleep_for(0.05)

        self._write_dataset_yaml(object_name)
        delta = self._saved - trial_saved_before
        self.get_logger().info(
            f"{C.GREEN}{C.BOLD}[DONE]{C.RESET} trial={self._trial_index} saved={delta}; total_saved={self._saved}; skipped={self._skipped}"
        )
        send_feedback(f"DatasetCollector done trial {self._trial_index}; saved {delta}")
        # Return True so the engine advances to the next dataset trial.
        return True