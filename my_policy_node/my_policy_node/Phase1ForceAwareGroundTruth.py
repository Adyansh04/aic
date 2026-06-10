
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
from transforms3d._gohlketransforms import quaternion_multiply, quaternion_slerp


QuaternionTuple = tuple[float, float, float, float]


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


class Phase1ForceAwareGroundTruth(Policy):
    """Geometry-first, entrance-aware Phase 1 policy.

    This version intentionally keeps the successful Phase1GroundTruth motion
    logic: align the plug to the *inner* port TF and run a slow geometric
    descent. It uses the entrance TF for diagnostics, entry latching, and safety
    gating only.

    Why this version exists:
      - Your Phase1GroundTruthLogged policy inserted all 3 trials.
      - The logs showed the successful path is mostly geometry driven.
      - Search/retreat logic was pulling the plug out after it had entered.
      - Force should be a safety net, not the main planner.

    Required launch:
        /entrypoint.sh ground_truth:=true start_aic_engine:=true

    Later perception phase:
      - Replace get_inner_port_tf()/get_entrance_tf() with PnP estimates.
      - Keep the same measured entrance_axis / entrance_lat logic.
    """

    # ------------------------------------------------------------------
    # Motion timing and target depth.
    # ------------------------------------------------------------------
    # The logged successful policy was using +0.1000 m as the start standoff.
    START_Z_OFFSET = 0.10
    END_Z_OFFSET = -0.018
    DESCENT_STEP = -0.0005

    APPROACH_STEPS = 100
    APPROACH_DT = 0.05
    DESCENT_DT = 0.05
    STABILIZE_SECONDS = 3.0

    # ------------------------------------------------------------------
    # Entrance-frame thresholds learned from your successful logs.
    # ------------------------------------------------------------------
    # In the successful SFP trials, entrance-axis crossing occurred around
    # z_offset ~= +0.0445. For SC, crossing happened around +0.015.
    # So we DO NOT use fixed z_offset for entry detection.
    ENTRY_LATCH_AXIS_M = 0.0000          # plug crosses entrance plane
    ENTRY_CONFIRMED_AXIS_M = 0.0050      # >= 5 mm past entrance
    ENTRY_MAX_LATERAL_M = 0.0025         # successful trials were ~0.0003-0.0010 m
    ALIGN_PAUSE_LATERAL_M = 0.0035       # if near entrance and worse than this, pause/realign
    ALIGN_TARGET_LATERAL_M = 0.0015
    ALIGN_TARGET_ANGLE_RAD = 0.035       # ~2 deg
    ALIGN_STABLE_COUNT = 6
    ALIGN_MAX_STEPS = 80

    # If force rises before the plug has crossed the entrance and lateral error
    # is also bad, stop descending briefly and re-align. Do not spiral by default.
    PRE_ENTRY_WARN_FORCE_N = 2.5
    PRE_ENTRY_HARD_FORCE_N = 6.0

    # Once inside, do not retreat unless something is truly wrong.
    # Successful final F_delta in your logs was roughly 5.2-5.9 N and tcp_err
    # around 0.017-0.018, so this must be higher than that.
    POST_ENTRY_WARN_FORCE_N = 7.0
    POST_ENTRY_HARD_FORCE_N = 12.0

    # Software force tare.
    FORCE_BIAS_SAMPLES = 25

    # ------------------------------------------------------------------
    # Controller gains.
    # ------------------------------------------------------------------
    FREE_SPACE_STIFFNESS = [90.0, 90.0, 90.0, 50.0, 50.0, 50.0]
    FREE_SPACE_DAMPING = [50.0, 50.0, 50.0, 20.0, 20.0, 20.0]

    INSERT_STIFFNESS = [70.0, 70.0, 70.0, 45.0, 45.0, 45.0]
    INSERT_DAMPING = [45.0, 45.0, 45.0, 18.0, 18.0, 18.0]

    # Keep the original working correction. Your successful logs showed it
    # helped converge the entrance lateral error to sub-mm values during descent.
    XY_INTEGRATOR_LIMIT = 0.05
    XY_INTEGRATOR_GAIN = 0.15

    def __init__(self, parent_node):
        self._task: Optional[Task] = None
        self._xy_integrator = np.zeros(2, dtype=float)

        self._force_bias = np.zeros(3, dtype=float)
        self._force_bias_ready = False

        self._entry_latched = False
        self._entry_confirmed = False
        self._entry_latch_z: Optional[float] = None
        self._entry_latch_axis: Optional[float] = None

        self._last_log_t: dict[str, float] = {}
        self._event_flags: set[str] = set()

        super().__init__(parent_node)
        self.log_event("INIT", "Phase1ForceAwareGroundTruth geometry-first policy initialized", C.CYAN)

    # ------------------------------------------------------------------
    # Logging helpers.
    # ------------------------------------------------------------------

    def now_sec(self) -> float:
        return self.time_now().nanoseconds * 1e-9

    def log_event(self, tag: str, msg: str, color: str = C.WHITE) -> None:
        self.get_logger().info(f"{color}{C.BOLD}[{tag}]{C.RESET} {color}{msg}{C.RESET}")

    def log_throttle(
        self,
        key: str,
        tag: str,
        msg: str,
        color: str = C.WHITE,
        period: float = 0.7,
    ) -> None:
        now = self.now_sec()
        last = self._last_log_t.get(key, -1e9)
        if now - last >= period:
            self._last_log_t[key] = now
            self.log_event(tag, msg, color)

    def log_once(self, key: str, tag: str, msg: str, color: str = C.WHITE) -> None:
        if key not in self._event_flags:
            self._event_flags.add(key)
            self.log_event(tag, msg, color)

    # ------------------------------------------------------------------
    # Quaternion and vector helpers.
    # Internal quaternion convention: (w, x, y, z)
    # geometry_msgs Quaternion convention: (x, y, z, w)
    # ------------------------------------------------------------------

    def normalize_q(self, q: QuaternionTuple) -> QuaternionTuple:
        arr = np.asarray(q, dtype=float)
        norm = np.linalg.norm(arr)
        if norm < 1e-12:
            return (1.0, 0.0, 0.0, 0.0)
        arr = arr / norm
        return (float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3]))

    def inverse_q(self, q: QuaternionTuple) -> QuaternionTuple:
        q = self.normalize_q(q)
        return (q[0], -q[1], -q[2], -q[3])

    def multiply_q(self, q1: QuaternionTuple, q2: QuaternionTuple) -> QuaternionTuple:
        q = quaternion_multiply(q1, q2)
        return self.normalize_q((float(q[0]), float(q[1]), float(q[2]), float(q[3])))

    def rotate_vec(self, q: QuaternionTuple, v: np.ndarray) -> np.ndarray:
        q = self.normalize_q(q)
        vq = (0.0, float(v[0]), float(v[1]), float(v[2]))
        rotated = quaternion_multiply(
            quaternion_multiply(q, vq),
            self.inverse_q(q),
        )
        return np.array([rotated[1], rotated[2], rotated[3]], dtype=float)

    def q_from_msg(self, rot) -> QuaternionTuple:
        return self.normalize_q((float(rot.w), float(rot.x), float(rot.y), float(rot.z)))

    def q_to_msg(self, q: QuaternionTuple) -> Quaternion:
        q = self.normalize_q(q)
        return Quaternion(x=q[1], y=q[2], z=q[3], w=q[0])

    def p_from_msg(self, trans) -> np.ndarray:
        return np.array([float(trans.x), float(trans.y), float(trans.z)], dtype=float)

    def quat_angle_error(self, q_target: QuaternionTuple, q_current: QuaternionTuple) -> float:
        q_err = self.multiply_q(q_target, self.inverse_q(q_current))
        w = float(np.clip(abs(q_err[0]), 0.0, 1.0))
        return float(2.0 * math.acos(w))

    # ------------------------------------------------------------------
    # TF helpers.
    # ------------------------------------------------------------------

    def inner_port_frame(self) -> str:
        if self._task is None:
            raise RuntimeError("Task not set")
        return f"task_board/{self._task.target_module_name}/{self._task.port_name}_link"

    def entrance_frame(self) -> str:
        return f"{self.inner_port_frame()}_entrance"

    def plug_frame(self) -> str:
        if self._task is None:
            raise RuntimeError("Task not set")
        return f"{self._task.cable_name}/{self._task.plug_name}_link"

    def lookup_transform(self, target_frame: str, source_frame: str):
        return self._parent_node._tf_buffer.lookup_transform(target_frame, source_frame, Time())

    def wait_for_tf(self, target_frame: str, source_frame: str, timeout_sec: float = 10.0) -> bool:
        start = self.time_now()
        timeout = Duration(seconds=timeout_sec)
        attempt = 0

        while (self.time_now() - start) < timeout:
            try:
                self.lookup_transform(target_frame, source_frame)
                self.log_event("TF", f"ready: {source_frame} → {target_frame}", C.GREEN)
                return True
            except TransformException:
                if attempt % 20 == 0:
                    self.log_throttle(
                        key=f"wait_tf_{source_frame}",
                        tag="TF",
                        msg=f"waiting for {source_frame} → {target_frame}; launch with ground_truth:=true",
                        color=C.YELLOW,
                        period=2.0,
                    )
                attempt += 1
                self.sleep_for(0.1)

        self.log_event(
            "ERROR",
            f"TF unavailable after {timeout_sec:.1f}s: {source_frame} → {target_frame}",
            C.RED,
        )
        return False

    def get_inner_port_tf(self):
        return self.lookup_transform("base_link", self.inner_port_frame()).transform

    def get_entrance_tf(self):
        return self.lookup_transform("base_link", self.entrance_frame()).transform

    def get_plug_tf(self):
        return self.lookup_transform("base_link", self.plug_frame()).transform

    def get_tcp_tf(self):
        return self.lookup_transform("base_link", "gripper/tcp").transform

    # ------------------------------------------------------------------
    # Controller command helpers.
    # ------------------------------------------------------------------

    def diag36(self, values: list[float]) -> list[float]:
        return np.diag(values).flatten().astype(float).tolist()

    def make_pose_motion_update(
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
        msg.header = Header(frame_id=frame_id, stamp=self.get_clock().now().to_msg())
        msg.pose = pose
        msg.velocity = Twist(
            linear=Vector3(x=0.0, y=0.0, z=0.0),
            angular=Vector3(x=0.0, y=0.0, z=0.0),
        )
        msg.target_stiffness = self.diag36(stiffness)
        msg.target_damping = self.diag36(damping)
        msg.feedforward_wrench_at_tip = Wrench(
            force=Vector3(x=0.0, y=0.0, z=0.0),
            torque=Vector3(x=0.0, y=0.0, z=0.0),
        )
        msg.wrench_feedback_gains_at_tip = wrench_feedback
        msg.trajectory_generation_mode = TrajectoryGenerationMode(
            mode=TrajectoryGenerationMode.MODE_POSITION,
        )
        return msg

    def send_pose(
        self,
        move_robot: MoveRobotCallback,
        pose: Pose,
        stiffness: Optional[list[float]] = None,
        damping: Optional[list[float]] = None,
    ) -> bool:
        try:
            move_robot(
                motion_update=self.make_pose_motion_update(
                    pose=pose,
                    frame_id="base_link",
                    stiffness=stiffness,
                    damping=damping,
                )
            )
            return True
        except Exception as ex:
            self.log_event("MOVE", f"move_robot exception: {ex}", C.RED)
            return False

    # ------------------------------------------------------------------
    # Force / observation helpers.
    # ------------------------------------------------------------------

    def get_force_raw(self, obs: Optional[Observation]) -> np.ndarray:
        if obs is None:
            return np.zeros(3, dtype=float)
        f = obs.wrist_wrench.wrench.force
        return np.array([float(f.x), float(f.y), float(f.z)], dtype=float)

    def get_force_delta(self, obs: Optional[Observation]) -> np.ndarray:
        raw = self.get_force_raw(obs)
        if not self._force_bias_ready:
            return raw
        return raw - self._force_bias

    def force_raw_norm(self, obs: Optional[Observation]) -> float:
        return float(np.linalg.norm(self.get_force_raw(obs)))

    def force_delta_norm(self, obs: Optional[Observation]) -> float:
        return float(np.linalg.norm(self.get_force_delta(obs)))

    def tcp_error_norm(self, obs: Optional[Observation]) -> float:
        if obs is None:
            return 0.0
        try:
            e = np.asarray(obs.controller_state.tcp_error[:3], dtype=float)
            return float(np.linalg.norm(e))
        except Exception:
            return 0.0

    def calibrate_force_bias(self, get_observation: GetObservationCallback) -> None:
        forces = []
        self.log_event("BIAS", f"calibrating force bias from {self.FORCE_BIAS_SAMPLES} samples", C.CYAN)
        for _ in range(self.FORCE_BIAS_SAMPLES):
            obs = get_observation()
            if obs is not None:
                forces.append(self.get_force_raw(obs))
            self.sleep_for(self.DESCENT_DT)

        if not forces:
            self._force_bias[:] = 0.0
            self._force_bias_ready = False
            self.log_event("BIAS", "could not read observations; using raw force as F_delta", C.YELLOW)
            return

        self._force_bias = np.mean(np.stack(forces, axis=0), axis=0)
        self._force_bias_ready = True
        self.log_event(
            "BIAS",
            (
                f"force bias = ({self._force_bias[0]:+.3f}, "
                f"{self._force_bias[1]:+.3f}, {self._force_bias[2]:+.3f}) N"
            ),
            C.GREEN,
        )

    # ------------------------------------------------------------------
    # Geometry.
    # ------------------------------------------------------------------

    def calc_tcp_pose_for_plug_alignment(
        self,
        inner_port_tf,
        z_offset: float,
        slerp_fraction: float = 1.0,
        position_fraction: float = 1.0,
        reset_xy_integrator: bool = False,
    ) -> Pose:
        """Compute TCP pose that aligns current plug link to inner port link.

        This is intentionally the same rigid transform logic as the successful
        Phase1GroundTruth baseline. It maps the current plug pose to the desired
        plug pose and applies the same rigid delta to the TCP:

            q_delta = q_port_desired * inv(q_plug_current)
            p_tcp_des = p_plug_des + R(q_delta) * (p_tcp_current - p_plug_current)
            q_tcp_des = q_delta * q_tcp_current
        """

        plug_tf = self.get_plug_tf()
        tcp_tf = self.get_tcp_tf()

        p_port = self.p_from_msg(inner_port_tf.translation)
        q_port = self.q_from_msg(inner_port_tf.rotation)

        p_plug = self.p_from_msg(plug_tf.translation)
        q_plug = self.q_from_msg(plug_tf.rotation)

        p_tcp = self.p_from_msg(tcp_tf.translation)
        q_tcp = self.q_from_msg(tcp_tf.rotation)

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

        q_delta = self.multiply_q(q_plug_desired, self.inverse_q(q_plug))

        p_tcp_offset_from_plug = p_tcp - p_plug
        p_tcp_desired = p_plug_desired + self.rotate_vec(q_delta, p_tcp_offset_from_plug)
        q_tcp_desired = self.multiply_q(q_delta, q_tcp)

        position_fraction = float(np.clip(position_fraction, 0.0, 1.0))
        slerp_fraction = float(np.clip(slerp_fraction, 0.0, 1.0))

        p_tcp_blend = position_fraction * p_tcp_desired + (1.0 - position_fraction) * p_tcp
        q_tcp_blend = quaternion_slerp(q_tcp, q_tcp_desired, slerp_fraction)
        q_tcp_blend = self.normalize_q(
            (
                float(q_tcp_blend[0]),
                float(q_tcp_blend[1]),
                float(q_tcp_blend[2]),
                float(q_tcp_blend[3]),
            )
        )

        self.log_throttle(
            key="geom",
            tag="GEOM",
            msg=(
                f"z_offset={z_offset:+.4f} "
                f"inner_xy_err=({xy_error[0]:+.4f},{xy_error[1]:+.4f}) "
                f"xy_int=({self._xy_integrator[0]:+.4f},{self._xy_integrator[1]:+.4f})"
            ),
            color=C.DIM + C.CYAN,
            period=0.8,
        )

        return Pose(
            position=Point(x=float(p_tcp_blend[0]), y=float(p_tcp_blend[1]), z=float(p_tcp_blend[2])),
            orientation=self.q_to_msg(q_tcp_blend),
        )

    def entrance_metrics(self, obs: Optional[Observation], z_offset: float) -> dict[str, float]:
        """Compute measured plug-vs-entrance metrics.

        entrance_axis:
            negative = plug is before/outside entrance
            zero     = plug frame has reached entrance plane
            positive = plug frame has moved inward past entrance

        entrance_lat:
            lateral distance from insertion axis through entrance
        """

        inner_tf = self.get_inner_port_tf()
        entrance_tf = self.get_entrance_tf()
        plug_tf = self.get_plug_tf()

        p_inner = self.p_from_msg(inner_tf.translation)
        q_inner = self.q_from_msg(inner_tf.rotation)

        p_entrance = self.p_from_msg(entrance_tf.translation)
        q_entrance = self.q_from_msg(entrance_tf.rotation)

        p_plug = self.p_from_msg(plug_tf.translation)
        q_plug = self.q_from_msg(plug_tf.rotation)

        axis = p_inner - p_entrance
        axis_len = float(np.linalg.norm(axis))
        if axis_len < 1e-8:
            # Fallback: use inner port local z-axis if entrance and inner frame
            # are accidentally co-located.
            axis = self.rotate_vec(q_entrance, np.array([0.0, 0.0, 1.0]))
            axis_len = float(np.linalg.norm(axis))
        insertion_axis = axis / max(axis_len, 1e-8)

        v = p_plug - p_entrance
        entrance_axis = float(np.dot(v, insertion_axis))
        lateral_vec = v - entrance_axis * insertion_axis
        entrance_lat = float(np.linalg.norm(lateral_vec))

        inner_xy_vec = p_inner[:2] - p_plug[:2]
        entrance_xy_vec = p_entrance[:2] - p_plug[:2]

        return {
            "z_offset": float(z_offset),
            "F_raw": self.force_raw_norm(obs),
            "F_delta": self.force_delta_norm(obs),
            "tcp_err": self.tcp_error_norm(obs),
            "axis_len": axis_len,
            "entrance_axis": entrance_axis,
            "entrance_lat": entrance_lat,
            "entrance_dx": float(entrance_xy_vec[0]),
            "entrance_dy": float(entrance_xy_vec[1]),
            "entrance_xy": float(np.linalg.norm(entrance_xy_vec)),
            "inner_dx": float(inner_xy_vec[0]),
            "inner_dy": float(inner_xy_vec[1]),
            "inner_xy": float(np.linalg.norm(inner_xy_vec)),
            "inner_z_base": float(p_plug[2] - p_inner[2]),
            "entrance_z_base": float(p_plug[2] - p_entrance[2]),
            "inner_ang": self.quat_angle_error(q_inner, q_plug),
            "entrance_ang": self.quat_angle_error(q_entrance, q_plug),
        }

    def log_metrics(self, phase: str, m: dict[str, float], period: float = 0.55) -> None:
        color = C.BLUE
        if m["F_delta"] >= self.POST_ENTRY_HARD_FORCE_N:
            color = C.RED
        elif m["F_delta"] >= self.PRE_ENTRY_WARN_FORCE_N:
            color = C.YELLOW
        elif self._entry_latched:
            color = C.GREEN

        self.log_throttle(
            key=f"metrics_{phase}",
            tag="METRIC",
            msg=(
                f"phase={phase} z={m['z_offset']:+.4f} "
                f"axis={m['entrance_axis']:+.4f} lat={m['entrance_lat']:.4f} "
                f"inner_xy={m['inner_xy']:.4f} inner_z={m['inner_z_base']:+.4f} "
                f"ang={m['inner_ang']:.3f} "
                f"F_delta={m['F_delta']:.2f}N F_raw={m['F_raw']:.2f}N "
                f"tcp_err={m['tcp_err']:.4f} "
                f"entry={self._entry_latched}/{self._entry_confirmed}"
            ),
            color=color,
            period=period,
        )

    def update_entry_latches(self, z_offset: float, m: dict[str, float]) -> None:
        if not self._entry_latched:
            if m["entrance_axis"] >= self.ENTRY_LATCH_AXIS_M and m["entrance_lat"] <= self.ENTRY_MAX_LATERAL_M:
                self._entry_latched = True
                self._entry_latch_z = z_offset
                self._entry_latch_axis = m["entrance_axis"]
                self.log_event(
                    "ENTRY",
                    (
                        f"latched at z={z_offset:+.4f}: "
                        f"axis={m['entrance_axis']:+.4f}, lat={m['entrance_lat']:.4f}, "
                        f"F_delta={m['F_delta']:.2f}N"
                    ),
                    C.GREEN,
                )
            elif m["entrance_axis"] >= self.ENTRY_LATCH_AXIS_M and m["entrance_lat"] > self.ENTRY_MAX_LATERAL_M:
                self.log_throttle(
                    key="entry_blocked_lat",
                    tag="ALIGN",
                    msg=(
                        f"at entrance but lateral too large: "
                        f"axis={m['entrance_axis']:+.4f}, lat={m['entrance_lat']:.4f}; "
                        f"target <= {self.ENTRY_MAX_LATERAL_M:.4f}"
                    ),
                    color=C.YELLOW,
                    period=0.5,
                )

        if self._entry_latched and not self._entry_confirmed:
            if m["entrance_axis"] >= self.ENTRY_CONFIRMED_AXIS_M:
                self._entry_confirmed = True
                self.log_event(
                    "ENTRY",
                    (
                        f"confirmed >=5mm past entrance: "
                        f"axis={m['entrance_axis']:+.4f}, lat={m['entrance_lat']:.4f}, "
                        f"z={z_offset:+.4f}"
                    ),
                    C.GREEN,
                )

    def align_at_current_z(
        self,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        z_offset: float,
        label: str,
        max_steps: Optional[int] = None,
    ) -> bool:
        """Pause forward progress and repeatedly command the same aligned pose.

        This is NOT a spiral. It is just a measured TF convergence wait, using
        the same inner-port target as the successful baseline.
        """

        if max_steps is None:
            max_steps = self.ALIGN_MAX_STEPS

        stable = 0
        self.log_event("ALIGN", f"{label}: holding z={z_offset:+.4f} for TF convergence", C.CYAN)

        for _ in range(max_steps):
            obs = get_observation()
            m = self.entrance_metrics(obs, z_offset)
            ok_lat = m["entrance_lat"] <= self.ALIGN_TARGET_LATERAL_M
            ok_ang = m["inner_ang"] <= self.ALIGN_TARGET_ANGLE_RAD

            if ok_lat and ok_ang:
                stable += 1
            else:
                stable = 0

            self.log_throttle(
                key=f"align_{label}",
                tag="ALIGN",
                msg=(
                    f"{label}: axis={m['entrance_axis']:+.4f} "
                    f"lat={m['entrance_lat']:.4f} ang={m['inner_ang']:.3f} "
                    f"F_delta={m['F_delta']:.2f}N stable={stable}/{self.ALIGN_STABLE_COUNT}"
                ),
                color=C.CYAN if stable < self.ALIGN_STABLE_COUNT else C.GREEN,
                period=0.35,
            )

            if stable >= self.ALIGN_STABLE_COUNT:
                self.log_event(
                    "ALIGN",
                    f"{label}: aligned lat={m['entrance_lat']:.4f}, ang={m['inner_ang']:.3f}",
                    C.GREEN,
                )
                return True

            try:
                inner_tf = self.get_inner_port_tf()
                pose = self.calc_tcp_pose_for_plug_alignment(
                    inner_port_tf=inner_tf,
                    z_offset=z_offset,
                    slerp_fraction=1.0,
                    position_fraction=1.0,
                    reset_xy_integrator=False,
                )
                self.send_pose(move_robot, pose, self.INSERT_STIFFNESS, self.INSERT_DAMPING)
            except TransformException as ex:
                self.log_throttle("tf_align", "TF", f"TF failed during alignment: {ex}", C.YELLOW, period=1.0)

            self.sleep_for(self.DESCENT_DT)

        obs = get_observation()
        m = self.entrance_metrics(obs, z_offset)
        self.log_event(
            "ALIGN",
            f"{label}: timeout lat={m['entrance_lat']:.4f}, ang={m['inner_ang']:.3f}",
            C.YELLOW,
        )
        return m["entrance_lat"] <= 2.0 * self.ALIGN_TARGET_LATERAL_M

    # ------------------------------------------------------------------
    # Main policy callback.
    # ------------------------------------------------------------------

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> bool:
        self._task = task
        self._xy_integrator[:] = 0.0
        self._force_bias[:] = 0.0
        self._force_bias_ready = False
        self._entry_latched = False
        self._entry_confirmed = False
        self._entry_latch_z = None
        self._entry_latch_axis = None
        self._last_log_t.clear()
        self._event_flags.clear()

        self.log_event("TASK", f"{task}", C.CYAN)
        send_feedback("Phase1ForceAwareGroundTruth geometry-first insertion starting")

        inner_frame = self.inner_port_frame()
        entrance_frame = self.entrance_frame()
        plug_frame = self.plug_frame()
        tcp_frame = "gripper/tcp"

        self.log_event("FRAME", f"inner control frame: {inner_frame}", C.BLUE)
        self.log_event("FRAME", f"entrance metric frame: {entrance_frame}", C.BLUE)
        self.log_event("FRAME", f"plug frame: {plug_frame}", C.BLUE)
        self.log_event("FRAME", f"tcp frame: {tcp_frame}", C.BLUE)

        for frame in [inner_frame, entrance_frame, plug_frame, tcp_frame]:
            if not self.wait_for_tf("base_link", frame, timeout_sec=10.0):
                send_feedback(f"Missing TF frame: {frame}")
                return False

        # ------------------------------------------------------------------
        # Phase A: same smooth approach as working baseline.
        # ------------------------------------------------------------------
        send_feedback("Moving to high pre-insertion pose")
        self.log_event("STATE", "APPROACH", C.MAGENTA)

        for i in range(self.APPROACH_STEPS + 1):
            frac = i / float(self.APPROACH_STEPS)

            try:
                inner_tf = self.get_inner_port_tf()
                pose = self.calc_tcp_pose_for_plug_alignment(
                    inner_port_tf=inner_tf,
                    z_offset=self.START_Z_OFFSET,
                    slerp_fraction=frac,
                    position_fraction=frac,
                    reset_xy_integrator=True,
                )
                self.send_pose(
                    move_robot=move_robot,
                    pose=pose,
                    stiffness=self.FREE_SPACE_STIFFNESS,
                    damping=self.FREE_SPACE_DAMPING,
                )

                obs = get_observation()
                m = self.entrance_metrics(obs, self.START_Z_OFFSET)
                self.log_metrics("approach", m, period=0.8)
            except TransformException as ex:
                self.log_throttle("tf_approach", "TF", f"TF failed during approach: {ex}", C.YELLOW, period=1.0)

            self.sleep_for(self.APPROACH_DT)

        # The successful logs set force bias after approach. Do the same.
        self.align_at_current_z(
            get_observation=get_observation,
            move_robot=move_robot,
            z_offset=self.START_Z_OFFSET,
            label="after_approach",
            max_steps=30,
        )
        self.calibrate_force_bias(get_observation)

        # ------------------------------------------------------------------
        # Phase B: geometry-first descent.
        # ------------------------------------------------------------------
        send_feedback("Starting entrance-aware geometric descent")
        self.log_event("STATE", "DESCENT", C.MAGENTA)

        z_offset = self.START_Z_OFFSET
        pause_realign_cooldown = 0

        while z_offset > self.END_Z_OFFSET:
            z_offset += self.DESCENT_STEP

            try:
                obs = get_observation()
                m = self.entrance_metrics(obs, z_offset)
                self.update_entry_latches(z_offset, m)

                # If we are close to the entrance but not latched and lateral
                # alignment is poor, pause descent and let TF alignment converge.
                near_entrance_window = -0.004 <= m["entrance_axis"] <= 0.010
                if (
                    not self._entry_latched
                    and near_entrance_window
                    and m["entrance_lat"] > self.ALIGN_PAUSE_LATERAL_M
                    and pause_realign_cooldown <= 0
                ):
                    self.log_event(
                        "ALIGN",
                        (
                            f"near entrance with high lateral error; pausing: "
                            f"axis={m['entrance_axis']:+.4f}, lat={m['entrance_lat']:.4f}"
                        ),
                        C.YELLOW,
                    )
                    self.align_at_current_z(get_observation, move_robot, z_offset, label="near_entrance")
                    pause_realign_cooldown = 30

                # Pre-entry force is safety only. Do not search. Re-align if
                # force rises while lateral error is also bad.
                if not self._entry_latched:
                    if m["F_delta"] >= self.PRE_ENTRY_HARD_FORCE_N:
                        self.log_event(
                            "SAFETY",
                            (
                                f"pre-entry hard force: F_delta={m['F_delta']:.2f}N "
                                f"axis={m['entrance_axis']:+.4f} lat={m['entrance_lat']:.4f}; "
                                "holding/re-aligning instead of pushing/searching"
                            ),
                            C.RED,
                        )
                        self.align_at_current_z(get_observation, move_robot, z_offset, label="pre_entry_force")
                    elif m["F_delta"] >= self.PRE_ENTRY_WARN_FORCE_N and m["entrance_lat"] > self.ALIGN_PAUSE_LATERAL_M:
                        self.log_throttle(
                            "pre_entry_force_warn",
                            "SAFETY",
                            (
                                f"pre-entry force + lateral error: F_delta={m['F_delta']:.2f}N "
                                f"lat={m['entrance_lat']:.4f}; pausing/re-aligning"
                            ),
                            C.YELLOW,
                            period=1.0,
                        )
                        self.align_at_current_z(get_observation, move_robot, z_offset, label="pre_entry_warn", max_steps=25)

                # After entry is latched, never go back to search/retreat just
                # because force rises to the normal successful insertion range.
                if self._entry_latched:
                    if m["F_delta"] >= self.POST_ENTRY_HARD_FORCE_N:
                        self.log_event(
                            "SAFETY",
                            (
                                f"post-entry hard force: F_delta={m['F_delta']:.2f}N "
                                f"axis={m['entrance_axis']:+.4f}; stopping descent and holding"
                            ),
                            C.RED,
                        )
                        break
                    elif m["F_delta"] >= self.POST_ENTRY_WARN_FORCE_N:
                        self.log_throttle(
                            "post_entry_force_warn",
                            "SAFETY",
                            (
                                f"post-entry force high but continuing cautiously: "
                                f"F_delta={m['F_delta']:.2f}N axis={m['entrance_axis']:+.4f}"
                            ),
                            C.YELLOW,
                            period=1.0,
                        )

                inner_tf = self.get_inner_port_tf()
                pose = self.calc_tcp_pose_for_plug_alignment(
                    inner_port_tf=inner_tf,
                    z_offset=z_offset,
                    slerp_fraction=1.0,
                    position_fraction=1.0,
                    reset_xy_integrator=False,
                )
                self.send_pose(
                    move_robot=move_robot,
                    pose=pose,
                    stiffness=self.INSERT_STIFFNESS,
                    damping=self.INSERT_DAMPING,
                )

                # Events + throttled metric log.
                if m["entrance_axis"] >= 0.0:
                    self.log_once(
                        "crossed_entrance",
                        "EVENT",
                        (
                            f"plug crossed entrance plane at z={z_offset:+.4f}, "
                            f"axis={m['entrance_axis']:+.4f}, lat={m['entrance_lat']:.4f}"
                        ),
                        C.GREEN,
                    )
                if m["entrance_axis"] >= self.ENTRY_CONFIRMED_AXIS_M:
                    self.log_once(
                        "confirmed_5mm",
                        "EVENT",
                        (
                            f"plug >=5mm past entrance at z={z_offset:+.4f}, "
                            f"axis={m['entrance_axis']:+.4f}, lat={m['entrance_lat']:.4f}"
                        ),
                        C.GREEN,
                    )

                self.log_metrics("descent", m, period=0.55)

            except TransformException as ex:
                self.log_throttle("tf_descent", "TF", f"TF failed during descent: {ex}", C.YELLOW, period=1.0)

            pause_realign_cooldown -= 1
            self.sleep_for(self.DESCENT_DT)

        # ------------------------------------------------------------------
        # Phase C: hold final pose. If entry was latched, this should return
        # success; do not run search/retreat after insertion.
        # ------------------------------------------------------------------
        send_feedback("Holding final pose for stabilization")
        self.log_event("STATE", "HOLD", C.MAGENTA)

        hold_start = self.now_sec()
        while self.now_sec() - hold_start < self.STABILIZE_SECONDS:
            try:
                obs = get_observation()
                m = self.entrance_metrics(obs, self.END_Z_OFFSET)
                self.update_entry_latches(self.END_Z_OFFSET, m)
                self.log_metrics("hold", m, period=0.55)
            except TransformException as ex:
                self.log_throttle("tf_hold", "TF", f"TF failed during hold: {ex}", C.YELLOW, period=1.0)

            self.sleep_for(self.DESCENT_DT)

        try:
            obs = get_observation()
            m = self.entrance_metrics(obs, self.END_Z_OFFSET)
            self.log_event(
                "FINAL",
                (
                    f"axis={m['entrance_axis']:+.4f}, lat={m['entrance_lat']:.4f}, "
                    f"F_delta={m['F_delta']:.2f}N, tcp_err={m['tcp_err']:.4f}, "
                    f"entry={self._entry_latched}/{self._entry_confirmed}"
                ),
                C.GREEN if self._entry_latched else C.YELLOW,
            )
        except TransformException:
            pass

        # This policy follows the known-successful ground-truth trajectory. AIC
        # scoring will decide physical success. If we got here without safety
        # abort, return True just like Phase1GroundTruth.
        self.log_event("DONE", "Phase1ForceAwareGroundTruth complete", C.GREEN)
        send_feedback("Phase1ForceAwareGroundTruth complete")
        return True