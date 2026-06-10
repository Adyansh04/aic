from __future__ import annotations

import math
from typing import Optional

import numpy as np

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
from rclpy.duration import Duration
from rclpy.time import Time
from std_msgs.msg import Header
from tf2_ros import TransformException
from transforms3d._gohlketransforms import (
    quaternion_multiply,
    quaternion_slerp,
)


QuaternionTuple = tuple[float, float, float, float]


class C:
    """ANSI color helpers for readable ROS terminal logs."""

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


class Phase1GroundTruthLogged(Policy):
    """Known-working Phase1GroundTruth with instrumentation only.

    This policy intentionally preserves the original open-loop geometric logic:
      - target frame remains task_board/<module>/<port>_link
      - approach and descent z offsets are unchanged
      - force is never used to make decisions
      - entrance frame is only logged, not used for control
    """

    # Motion timing. Kept exactly from the working Phase1GroundTruth baseline.
    APPROACH_STEPS = 100
    APPROACH_DT = 0.05

    DESCENT_DT = 0.05
    START_Z_OFFSET = 0.1
    END_Z_OFFSET = -0.018
    DESCENT_STEP = -0.0005

    STABILIZE_SECONDS = 3.0

    # Controller gains. Kept exactly from the working Phase1GroundTruth baseline.
    FREE_SPACE_STIFFNESS = [90.0, 90.0, 90.0, 50.0, 50.0, 50.0]
    FREE_SPACE_DAMPING = [50.0, 50.0, 50.0, 20.0, 20.0, 20.0]

    INSERT_STIFFNESS = [70.0, 70.0, 70.0, 45.0, 45.0, 45.0]
    INSERT_DAMPING = [45.0, 45.0, 45.0, 18.0, 18.0, 18.0]

    # Kept exactly from the working Phase1GroundTruth baseline.
    XY_INTEGRATOR_LIMIT = 0.05
    XY_INTEGRATOR_GAIN = 0.15

    # Logging only.
    SAMPLE_LOG_PERIOD = 0.35
    GEOM_LOG_PERIOD = 0.75

    SOFT_FORCE_EVENT_N = 1.0
    BLOCKED_FORCE_EVENT_N = 3.0
    HARD_FORCE_EVENT_N = 7.0

    ENTRANCE_PLANE_EPS_M = 0.0
    INSERTION_DEPTH_EVENT_M = 0.005

    def __init__(self, parent_node):
        self._task: Optional[Task] = None
        self._xy_integrator = np.zeros(2, dtype=float)

        self._last_log_t: dict[str, float] = {}
        self._event_keys: set[str] = set()

        self._force_bias = np.zeros(3, dtype=float)
        self._force_bias_ready = False

        super().__init__(parent_node)
        self.log_event("INIT", "Phase1GroundTruthLogged initialized; control logic unchanged", C.CYAN)

    # -------------------------------------------------------------------------
    # Logging helpers.
    # -------------------------------------------------------------------------

    def _now_sec(self) -> float:
        return self.time_now().nanoseconds * 1e-9

    def log_event(self, tag: str, msg: str, color: str = C.WHITE) -> None:
        self.get_logger().info(f"{color}{C.BOLD}[{tag}]{C.RESET} {color}{msg}{C.RESET}")

    def log_throttle(
        self,
        key: str,
        tag: str,
        msg: str,
        color: str = C.WHITE,
        period: float = 0.75,
    ) -> None:
        now = self._now_sec()
        last = self._last_log_t.get(key, -1e9)
        if now - last >= period:
            self._last_log_t[key] = now
            self.log_event(tag, msg, color)

    def log_once(self, key: str, tag: str, msg: str, color: str = C.WHITE) -> None:
        if key not in self._event_keys:
            self._event_keys.add(key)
            self.log_event(tag, msg, color)

    # -------------------------------------------------------------------------
    # Quaternion and vector helpers.
    # Convention in this file:
    #     internal quaternion tuple = (w, x, y, z)
    #     geometry_msgs/Quaternion = (x, y, z, w)
    # -------------------------------------------------------------------------

    def _normalize_q(self, q: QuaternionTuple) -> QuaternionTuple:
        arr = np.array(q, dtype=float)
        norm = np.linalg.norm(arr)
        if norm < 1e-12:
            return (1.0, 0.0, 0.0, 0.0)
        arr = arr / norm
        return (float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3]))

    def _inverse_q(self, q: QuaternionTuple) -> QuaternionTuple:
        q = self._normalize_q(q)
        return (q[0], -q[1], -q[2], -q[3])

    def _multiply_q(self, q1: QuaternionTuple, q2: QuaternionTuple) -> QuaternionTuple:
        q = quaternion_multiply(q1, q2)
        return self._normalize_q((float(q[0]), float(q[1]), float(q[2]), float(q[3])))

    def _rotate_vec(self, q: QuaternionTuple, v: np.ndarray) -> np.ndarray:
        """Rotate 3D vector v by quaternion q."""
        q = self._normalize_q(q)
        vq = (0.0, float(v[0]), float(v[1]), float(v[2]))
        rotated = quaternion_multiply(
            quaternion_multiply(q, vq),
            self._inverse_q(q),
        )
        return np.array([rotated[1], rotated[2], rotated[3]], dtype=float)

    def _q_from_tf_rotation(self, rot) -> QuaternionTuple:
        return self._normalize_q(
            (
                float(rot.w),
                float(rot.x),
                float(rot.y),
                float(rot.z),
            )
        )

    def _q_to_msg(self, q: QuaternionTuple) -> Quaternion:
        q = self._normalize_q(q)
        return Quaternion(x=q[1], y=q[2], z=q[3], w=q[0])

    def _p_from_tf_translation(self, trans) -> np.ndarray:
        return np.array(
            [
                float(trans.x),
                float(trans.y),
                float(trans.z),
            ],
            dtype=float,
        )

    def _quat_angle_error(self, q_target: QuaternionTuple, q_current: QuaternionTuple) -> float:
        """Smallest orientation error angle in radians."""
        q_err = self._multiply_q(q_target, self._inverse_q(q_current))
        return float(2.0 * math.atan2(np.linalg.norm(q_err[1:]), abs(q_err[0])))

    # -------------------------------------------------------------------------
    # TF helpers.
    # -------------------------------------------------------------------------

    def _lookup_transform(self, target_frame: str, source_frame: str):
        return self._parent_node._tf_buffer.lookup_transform(
            target_frame,
            source_frame,
            Time(),
        )

    def _wait_for_tf(
        self,
        target_frame: str,
        source_frame: str,
        timeout_sec: float = 10.0,
    ) -> bool:
        start = self.time_now()
        timeout = Duration(seconds=timeout_sec)
        attempt = 0

        while (self.time_now() - start) < timeout:
            try:
                self._lookup_transform(target_frame, source_frame)
                self.log_event("TF", f"ready: {source_frame} -> {target_frame}", C.GREEN)
                return True
            except TransformException:
                if attempt % 20 == 0:
                    self.log_throttle(
                        key=f"wait_tf_{source_frame}",
                        tag="TF",
                        msg=(
                            f"Waiting for TF '{source_frame}' -> '{target_frame}'. "
                            "Did you launch eval with ground_truth:=true?"
                        ),
                        color=C.YELLOW,
                        period=2.0,
                    )
                attempt += 1
                self.sleep_for(0.1)

        self.log_event(
            "ERROR",
            f"TF '{source_frame}' -> '{target_frame}' not available after {timeout_sec:.1f}s",
            C.RED,
        )
        return False

    def _try_lookup_transform(self, target_frame: str, source_frame: str):
        try:
            return self._lookup_transform(target_frame, source_frame).transform
        except TransformException:
            return None

    # -------------------------------------------------------------------------
    # Controller command helpers.
    # -------------------------------------------------------------------------

    def _diag36(self, values: list[float]) -> list[float]:
        return np.diag(values).flatten().astype(float).tolist()

    def _make_pose_motion_update(
        self,
        pose: Pose,
        frame_id: str = "base_link",
        stiffness: Optional[list[float]] = None,
        damping: Optional[list[float]] = None,
        wrench_feedback: Optional[list[float]] = None,
    ) -> MotionUpdate:
        if stiffness is None:
            stiffness = self.FREE_SPACE_STIFFNESS
        if damping is None:
            damping = self.FREE_SPACE_DAMPING
        if wrench_feedback is None:
            wrench_feedback = [0.5, 0.5, 0.5, 0.0, 0.0, 0.0]

        msg = MotionUpdate()
        msg.header = Header(
            frame_id=frame_id,
            stamp=self.get_clock().now().to_msg(),
        )
        msg.pose = pose
        msg.velocity = Twist(
            linear=Vector3(x=0.0, y=0.0, z=0.0),
            angular=Vector3(x=0.0, y=0.0, z=0.0),
        )
        msg.target_stiffness = self._diag36(stiffness)
        msg.target_damping = self._diag36(damping)
        msg.feedforward_wrench_at_tip = Wrench(
            force=Vector3(x=0.0, y=0.0, z=0.0),
            torque=Vector3(x=0.0, y=0.0, z=0.0),
        )
        msg.wrench_feedback_gains_at_tip = wrench_feedback
        msg.trajectory_generation_mode = TrajectoryGenerationMode(
            mode=TrajectoryGenerationMode.MODE_POSITION,
        )
        return msg

    def _send_pose(
        self,
        move_robot: MoveRobotCallback,
        pose: Pose,
        stiffness: Optional[list[float]] = None,
        damping: Optional[list[float]] = None,
    ) -> bool:
        cmd = self._make_pose_motion_update(
            pose=pose,
            frame_id="base_link",
            stiffness=stiffness,
            damping=damping,
        )
        return bool(move_robot(motion_update=cmd))

    # -------------------------------------------------------------------------
    # Observation helpers.
    # -------------------------------------------------------------------------

    def _force_vec_raw(self, observation: Optional[Observation]) -> np.ndarray:
        if observation is None:
            return np.zeros(3, dtype=float)
        f = observation.wrist_wrench.wrench.force
        return np.array([float(f.x), float(f.y), float(f.z)], dtype=float)

    def _force_vec_delta(self, observation: Optional[Observation]) -> np.ndarray:
        raw = self._force_vec_raw(observation)
        if not self._force_bias_ready:
            return raw
        return raw - self._force_bias

    def _force_norm_raw(self, observation: Optional[Observation]) -> float:
        return float(np.linalg.norm(self._force_vec_raw(observation)))

    def _force_norm_delta(self, observation: Optional[Observation]) -> float:
        return float(np.linalg.norm(self._force_vec_delta(observation)))

    def _tcp_error_norm(self, observation: Optional[Observation]) -> float:
        if observation is None:
            return 0.0
        try:
            e = np.asarray(observation.controller_state.tcp_error[:3], dtype=float)
            return float(np.linalg.norm(e))
        except Exception:
            return 0.0

    def _set_force_bias_from_observation(
        self,
        get_observation: GetObservationCallback,
        label: str,
    ) -> None:
        observation = get_observation()
        self._force_bias = self._force_vec_raw(observation)
        self._force_bias_ready = observation is not None
        if self._force_bias_ready:
            self.log_event(
                "BIAS",
                (
                    f"{label}: force bias set to "
                    f"({self._force_bias[0]:+.3f}, {self._force_bias[1]:+.3f}, {self._force_bias[2]:+.3f}) N"
                ),
                C.GREEN,
            )
        else:
            self.log_event("BIAS", f"{label}: no observation available; F_delta will equal raw force", C.YELLOW)

    # -------------------------------------------------------------------------
    # Geometry core.
    # -------------------------------------------------------------------------

    def _calc_tcp_pose_for_plug_alignment(
        self,
        port_transform,
        z_offset: float,
        slerp_fraction: float = 1.0,
        position_fraction: float = 1.0,
        reset_xy_integrator: bool = False,
    ) -> Pose:
        """Return TCP pose that makes current plug link align with port link.

        Logic intentionally unchanged from the known-working Phase1GroundTruth.
        """

        if self._task is None:
            raise RuntimeError("Task not set")

        plug_frame = f"{self._task.cable_name}/{self._task.plug_name}_link"

        plug_tf = self._lookup_transform("base_link", plug_frame).transform
        tcp_tf = self._lookup_transform("base_link", "gripper/tcp").transform

        p_port = self._p_from_tf_translation(port_transform.translation)
        q_port = self._q_from_tf_rotation(port_transform.rotation)

        p_plug = self._p_from_tf_translation(plug_tf.translation)
        q_plug = self._q_from_tf_rotation(plug_tf.rotation)

        p_tcp = self._p_from_tf_translation(tcp_tf.translation)
        q_tcp = self._q_from_tf_rotation(tcp_tf.rotation)

        xy_error = p_port[:2] - p_plug[:2]

        if reset_xy_integrator:
            self._xy_integrator[:] = 0.0
        else:
            self._xy_integrator += xy_error
            self._xy_integrator = np.clip(
                self._xy_integrator,
                -self.XY_INTEGRATOR_LIMIT,
                self.XY_INTEGRATOR_LIMIT,
            )

        corrected_xy = p_port[:2] + self.XY_INTEGRATOR_GAIN * self._xy_integrator

        p_plug_desired = np.array(
            [
                corrected_xy[0],
                corrected_xy[1],
                p_port[2] + z_offset,
            ],
            dtype=float,
        )
        q_plug_desired = q_port

        q_delta = self._multiply_q(q_plug_desired, self._inverse_q(q_plug))

        p_tcp_offset_from_plug = p_tcp - p_plug
        p_tcp_desired = p_plug_desired + self._rotate_vec(q_delta, p_tcp_offset_from_plug)
        q_tcp_desired = self._multiply_q(q_delta, q_tcp)

        position_fraction = float(np.clip(position_fraction, 0.0, 1.0))
        slerp_fraction = float(np.clip(slerp_fraction, 0.0, 1.0))

        p_tcp_blend = (
            position_fraction * p_tcp_desired
            + (1.0 - position_fraction) * p_tcp
        )

        q_tcp_blend = quaternion_slerp(q_tcp, q_tcp_desired, slerp_fraction)
        q_tcp_blend = self._normalize_q(
            (
                float(q_tcp_blend[0]),
                float(q_tcp_blend[1]),
                float(q_tcp_blend[2]),
                float(q_tcp_blend[3]),
            )
        )

        self.log_throttle(
            key="phase1_geometry",
            tag="GEOM",
            msg=(
                f"z_offset={z_offset:+.4f} "
                f"inner_xy_err=({xy_error[0]:+.4f},{xy_error[1]:+.4f}) "
                f"xy_int=({self._xy_integrator[0]:+.4f},{self._xy_integrator[1]:+.4f})"
            ),
            color=C.DIM + C.CYAN,
            period=self.GEOM_LOG_PERIOD,
        )

        return Pose(
            position=Point(
                x=float(p_tcp_blend[0]),
                y=float(p_tcp_blend[1]),
                z=float(p_tcp_blend[2]),
            ),
            orientation=self._q_to_msg(q_tcp_blend),
        )

    # -------------------------------------------------------------------------
    # Instrumentation.
    # -------------------------------------------------------------------------

    def _compute_metrics(
        self,
        port_tf,
        entrance_tf,
        observation: Optional[Observation],
    ) -> dict[str, float]:
        if self._task is None:
            raise RuntimeError("Task not set")

        plug_frame = f"{self._task.cable_name}/{self._task.plug_name}_link"
        plug_tf = self._lookup_transform("base_link", plug_frame).transform

        p_inner = self._p_from_tf_translation(port_tf.translation)
        q_inner = self._q_from_tf_rotation(port_tf.rotation)

        p_plug = self._p_from_tf_translation(plug_tf.translation)
        q_plug = self._q_from_tf_rotation(plug_tf.rotation)

        inner_xy_err = p_inner[:2] - p_plug[:2]
        inner_z = float(p_plug[2] - p_inner[2])
        inner_ang = self._quat_angle_error(q_inner, q_plug)

        metrics = {
            "rawF": self._force_norm_raw(observation),
            "dF": self._force_norm_delta(observation),
            "tcp_err": self._tcp_error_norm(observation),
            "inner_xy": float(np.linalg.norm(inner_xy_err)),
            "inner_dx": float(inner_xy_err[0]),
            "inner_dy": float(inner_xy_err[1]),
            "inner_z": inner_z,
            "inner_ang": inner_ang,
            "ent_xy": float("nan"),
            "ent_dx": float("nan"),
            "ent_dy": float("nan"),
            "ent_z": float("nan"),
            "ent_axis": float("nan"),
            "ent_lat": float("nan"),
            "ent_ang": float("nan"),
        }

        if entrance_tf is not None:
            p_ent = self._p_from_tf_translation(entrance_tf.translation)
            q_ent = self._q_from_tf_rotation(entrance_tf.rotation)

            ent_xy_err = p_ent[:2] - p_plug[:2]
            ent_z = float(p_plug[2] - p_ent[2])
            ent_ang = self._quat_angle_error(q_ent, q_plug)

            axis_vec = p_inner - p_ent
            axis_norm = float(np.linalg.norm(axis_vec))
            if axis_norm > 1e-9:
                axis = axis_vec / axis_norm
                v = p_plug - p_ent
                ent_axis = float(np.dot(v, axis))
                lateral_vec = v - ent_axis * axis
                ent_lat = float(np.linalg.norm(lateral_vec))
            else:
                ent_axis = float("nan")
                ent_lat = float("nan")

            metrics.update(
                {
                    "ent_xy": float(np.linalg.norm(ent_xy_err)),
                    "ent_dx": float(ent_xy_err[0]),
                    "ent_dy": float(ent_xy_err[1]),
                    "ent_z": ent_z,
                    "ent_axis": ent_axis,
                    "ent_lat": ent_lat,
                    "ent_ang": ent_ang,
                }
            )

        return metrics

    def _log_metrics(
        self,
        phase: str,
        z_offset: float,
        port_tf,
        entrance_tf,
        get_observation: GetObservationCallback,
        force_now: bool = False,
    ) -> None:
        observation = get_observation()

        try:
            m = self._compute_metrics(port_tf, entrance_tf, observation)
        except TransformException as ex:
            self.log_throttle(
                key=f"metrics_tf_{phase}",
                tag="TF",
                msg=f"{phase}: TF failed during metric logging: {ex}",
                color=C.YELLOW,
                period=1.0,
            )
            return

        color = C.BLUE
        if m["dF"] >= self.HARD_FORCE_EVENT_N:
            color = C.RED
        elif m["dF"] >= self.BLOCKED_FORCE_EVENT_N:
            color = C.YELLOW
        elif m["dF"] >= self.SOFT_FORCE_EVENT_N:
            color = C.MAGENTA

        # Event logs.
        if m["dF"] >= self.SOFT_FORCE_EVENT_N:
            self.log_once(
                f"{phase}_soft_force",
                "EVENT",
                f"{phase}: F_delta crossed {self.SOFT_FORCE_EVENT_N:.1f}N at z={z_offset:+.4f}, ent_axis={m['ent_axis']:+.4f}, inner_z={m['inner_z']:+.4f}",
                C.MAGENTA,
            )
        if m["dF"] >= self.BLOCKED_FORCE_EVENT_N:
            self.log_once(
                f"{phase}_blocked_force",
                "EVENT",
                f"{phase}: F_delta crossed {self.BLOCKED_FORCE_EVENT_N:.1f}N at z={z_offset:+.4f}, ent_axis={m['ent_axis']:+.4f}, inner_z={m['inner_z']:+.4f}",
                C.YELLOW,
            )
        if m["dF"] >= self.HARD_FORCE_EVENT_N:
            self.log_once(
                f"{phase}_hard_force",
                "EVENT",
                f"{phase}: F_delta crossed {self.HARD_FORCE_EVENT_N:.1f}N at z={z_offset:+.4f}, ent_axis={m['ent_axis']:+.4f}, inner_z={m['inner_z']:+.4f}",
                C.RED,
            )

        if not math.isnan(m["ent_axis"]):
            if m["ent_axis"] >= self.ENTRANCE_PLANE_EPS_M:
                self.log_once(
                    f"{phase}_entrance_plane",
                    "EVENT",
                    f"{phase}: plug crossed entrance plane at z={z_offset:+.4f}, ent_axis={m['ent_axis']:+.4f}, ent_lat={m['ent_lat']:+.4f}",
                    C.GREEN,
                )
            if m["ent_axis"] >= self.INSERTION_DEPTH_EVENT_M:
                self.log_once(
                    f"{phase}_inside_5mm",
                    "EVENT",
                    f"{phase}: plug >= {self.INSERTION_DEPTH_EVENT_M*1000:.1f}mm past entrance at z={z_offset:+.4f}, ent_axis={m['ent_axis']:+.4f}, ent_lat={m['ent_lat']:+.4f}",
                    C.GREEN,
                )

        msg = (
            f"phase={phase} z_offset={z_offset:+.4f} "
            f"F_raw={m['rawF']:.3f}N F_delta={m['dF']:.3f}N tcp_err={m['tcp_err']:.4f} "
            f"inner_xy={m['inner_xy']:.4f} inner_dx={m['inner_dx']:+.4f} inner_dy={m['inner_dy']:+.4f} "
            f"inner_z={m['inner_z']:+.4f} inner_ang={m['inner_ang']:.3f}rad "
            f"entrance_xy={m['ent_xy']:.4f} entrance_dx={m['ent_dx']:+.4f} entrance_dy={m['ent_dy']:+.4f} "
            f"entrance_z={m['ent_z']:+.4f} entrance_axis={m['ent_axis']:+.4f} entrance_lat={m['ent_lat']:.4f} "
            f"entrance_ang={m['ent_ang']:.3f}rad"
        )

        if force_now:
            self.log_event("SAMPLE", msg, color)
        else:
            self.log_throttle(
                key=f"sample_{phase}",
                tag="SAMPLE",
                msg=msg,
                color=color,
                period=self.SAMPLE_LOG_PERIOD,
            )

    # -------------------------------------------------------------------------
    # Main AIC callback.
    # -------------------------------------------------------------------------

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> bool:
        self._task = task
        self._xy_integrator[:] = 0.0
        self._last_log_t.clear()
        self._event_keys.clear()
        self._force_bias[:] = 0.0
        self._force_bias_ready = False

        self.log_event("TASK", f"Phase1GroundTruthLogged.insert_cable() task: {task}", C.CYAN)
        send_feedback("Phase 1 ground-truth logged baseline starting")

        port_frame = f"task_board/{task.target_module_name}/{task.port_name}_link"
        entrance_frame = f"task_board/{task.target_module_name}/{task.port_name}_link_entrance"
        plug_frame = f"{task.cable_name}/{task.plug_name}_link"

        self.log_event("FRAME", f"target inner port frame used for CONTROL: {port_frame}", C.BLUE)
        self.log_event("FRAME", f"entrance frame used for LOGGING ONLY:      {entrance_frame}", C.BLUE)
        self.log_event("FRAME", f"plug frame:                              {plug_frame}", C.BLUE)

        # These frames require ground_truth:=true. Entrance is optional for logging.
        for frame in [port_frame, plug_frame, "gripper/tcp"]:
            if not self._wait_for_tf("base_link", frame, timeout_sec=10.0):
                send_feedback(f"Missing TF frame: {frame}")
                return False

        entrance_available = self._wait_for_tf("base_link", entrance_frame, timeout_sec=2.0)
        if not entrance_available:
            self.log_event("FRAME", "entrance frame unavailable; entrance metrics will be NaN", C.YELLOW)

        try:
            port_tf = self._lookup_transform("base_link", port_frame).transform
        except TransformException as ex:
            self.log_event("ERROR", f"Could not look up port TF: {ex}", C.RED)
            return False

        entrance_tf = self._try_lookup_transform("base_link", entrance_frame) if entrance_available else None

        self.log_event(
            "MODE",
            "Control logic is unchanged: inner port frame target + open-loop z descent; all new data is logging only.",
            C.MAGENTA,
        )

        # ---------------------------------------------------------------------
        # Phase A: smooth move from current pose to a high pre-insertion pose.
        # ---------------------------------------------------------------------
        send_feedback("Moving to high pre-insertion pose")

        for i in range(self.APPROACH_STEPS + 1):
            frac = i / float(self.APPROACH_STEPS)

            try:
                pose = self._calc_tcp_pose_for_plug_alignment(
                    port_transform=port_tf,
                    z_offset=self.START_Z_OFFSET,
                    slerp_fraction=frac,
                    position_fraction=frac,
                    reset_xy_integrator=True,
                )
                self._send_pose(
                    move_robot=move_robot,
                    pose=pose,
                    stiffness=self.FREE_SPACE_STIFFNESS,
                    damping=self.FREE_SPACE_DAMPING,
                )
            except TransformException as ex:
                self.get_logger().warn(f"TF failed during approach: {ex}")

            self._log_metrics(
                phase="approach",
                z_offset=self.START_Z_OFFSET,
                port_tf=port_tf,
                entrance_tf=entrance_tf,
                get_observation=get_observation,
            )
            self.sleep_for(self.APPROACH_DT)

        self._log_metrics(
            phase="after_approach",
            z_offset=self.START_Z_OFFSET,
            port_tf=port_tf,
            entrance_tf=entrance_tf,
            get_observation=get_observation,
            force_now=True,
        )

        # Bias only affects logged F_delta, not control.
        self._set_force_bias_from_observation(get_observation, label="after_approach")

        # ---------------------------------------------------------------------
        # Phase B: slow geometric descent.
        #
        # This is intentionally the known-working behavior. We only add rich logs.
        # ---------------------------------------------------------------------
        send_feedback("Starting slow ground-truth descent")

        z_offset = self.START_Z_OFFSET

        while z_offset > self.END_Z_OFFSET:
            z_offset += self.DESCENT_STEP

            try:
                pose = self._calc_tcp_pose_for_plug_alignment(
                    port_transform=port_tf,
                    z_offset=z_offset,
                    slerp_fraction=1.0,
                    position_fraction=1.0,
                    reset_xy_integrator=False,
                )
                self._send_pose(
                    move_robot=move_robot,
                    pose=pose,
                    stiffness=self.INSERT_STIFFNESS,
                    damping=self.INSERT_DAMPING,
                )
            except TransformException as ex:
                self.get_logger().warn(f"TF failed during descent: {ex}")

            self._log_metrics(
                phase="descent",
                z_offset=z_offset,
                port_tf=port_tf,
                entrance_tf=entrance_tf,
                get_observation=get_observation,
            )

            self.sleep_for(self.DESCENT_DT)

        # ---------------------------------------------------------------------
        # Phase C: hold briefly so Gazebo/contact settles.
        # ---------------------------------------------------------------------
        send_feedback("Holding final pose for stabilization")
        self.log_event("HOLD", "Waiting for connector/contact to stabilize...", C.CYAN)

        hold_start = self._now_sec()
        while self._now_sec() - hold_start < self.STABILIZE_SECONDS:
            self._log_metrics(
                phase="hold",
                z_offset=z_offset,
                port_tf=port_tf,
                entrance_tf=entrance_tf,
                get_observation=get_observation,
            )
            self.sleep_for(self.DESCENT_DT)

        self._log_metrics(
            phase="final",
            z_offset=z_offset,
            port_tf=port_tf,
            entrance_tf=entrance_tf,
            get_observation=get_observation,
            force_now=True,
        )

        self.log_event("DONE", "Phase1GroundTruthLogged.insert_cable() complete", C.GREEN)
        send_feedback("Phase 1 ground-truth logged baseline complete")
        self.sleep_for(2.0)
        return True