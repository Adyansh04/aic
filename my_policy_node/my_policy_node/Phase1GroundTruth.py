from __future__ import annotations

import math
from typing import Callable, Optional

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


class Phase1GroundTruth(Policy):
    """Phase 1: ground-truth geometric insertion baseline.

    This policy intentionally uses ground-truth TF frames. It is meant for
    development/debugging only and must be run with:

        /entrypoint.sh ground_truth:=true start_aic_engine:=true

    Later phases should replace the port TF lookup with keypoint/PnP perception.
    """

    # Motion timing.
    APPROACH_STEPS = 100
    APPROACH_DT = 0.05

    DESCENT_DT = 0.05
    START_Z_OFFSET = 0.20
    END_Z_OFFSET = -0.018
    DESCENT_STEP = -0.0005

    STABILIZE_SECONDS = 3.0

    # Controller gains. These are deliberately close to the defaults used by
    # aic_model.Policy.set_pose_target, but separated here so we can tune later.
    FREE_SPACE_STIFFNESS = [90.0, 90.0, 90.0, 50.0, 50.0, 50.0]
    FREE_SPACE_DAMPING = [50.0, 50.0, 50.0, 20.0, 20.0, 20.0]

    INSERT_STIFFNESS = [70.0, 70.0, 70.0, 45.0, 45.0, 45.0]
    INSERT_DAMPING = [45.0, 45.0, 45.0, 18.0, 18.0, 18.0]

    # Small integral correction copied conceptually from CheatCode. This helps
    # reduce persistent XY error between the plug link and port link.
    XY_INTEGRATOR_LIMIT = 0.05
    XY_INTEGRATOR_GAIN = 0.15

    def __init__(self, parent_node):
        self._task: Optional[Task] = None
        self._xy_integrator = np.zeros(2, dtype=float)
        super().__init__(parent_node)
        self.get_logger().info("Phase1GroundTruth.__init__()")

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
                self.get_logger().info(
                    f"TF ready: {source_frame} -> {target_frame}"
                )
                return True
            except TransformException:
                if attempt % 20 == 0:
                    self.get_logger().info(
                        f"Waiting for TF '{source_frame}' -> '{target_frame}'. "
                        "Did you launch eval with ground_truth:=true?"
                    )
                attempt += 1
                self.sleep_for(0.1)

        self.get_logger().error(
            f"TF '{source_frame}' -> '{target_frame}' not available "
            f"after {timeout_sec:.1f}s"
        )
        return False

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

        We use the current ground-truth relation:

            base -> plug_link
            base -> gripper/tcp

        and compute the TCP target that would move the plug link to:

            x/y = target port x/y
            z   = target port z + z_offset
            R   = target port rotation

        This is still ground-truth based, but this formulation is modular:
        later we can replace port_transform with a PnP-estimated port pose.
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

        # XY integral correction. This makes the plug link converge to the port
        # center even if tiny persistent offsets remain.
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

        # Rigid transform delta that maps current plug pose to desired plug pose.
        q_delta = self._multiply_q(q_plug_desired, self._inverse_q(q_plug))

        # Preserve the current TCP offset from the plug, but rotate it by q_delta.
        p_tcp_offset_from_plug = p_tcp - p_plug
        p_tcp_desired = p_plug_desired + self._rotate_vec(
            q_delta, p_tcp_offset_from_plug
        )

        q_tcp_desired = self._multiply_q(q_delta, q_tcp)

        # Smooth interpolation from current TCP pose to target TCP pose.
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

        self.get_logger().info(
            "phase1 geometry "
            f"z_offset={z_offset:+.4f} "
            f"xy_err=({xy_error[0]:+.4f},{xy_error[1]:+.4f}) "
            f"xy_int=({self._xy_integrator[0]:+.4f},{self._xy_integrator[1]:+.4f})"
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
    # Observation helpers.
    # -------------------------------------------------------------------------

    def _force_norm(self, observation: Optional[Observation]) -> Optional[float]:
        if observation is None:
            return None

        f = observation.wrist_wrench.wrench.force
        return math.sqrt(f.x * f.x + f.y * f.y + f.z * f.z)

    def _log_observation_summary(
        self,
        get_observation: GetObservationCallback,
        prefix: str,
    ) -> None:
        observation = get_observation()
        force_norm = self._force_norm(observation)
        if force_norm is None:
            self.get_logger().info(f"{prefix}: no observation yet")
        else:
            self.get_logger().info(f"{prefix}: |F|={force_norm:.3f} N")

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

        self.get_logger().info(f"Phase1GroundTruth.insert_cable() task: {task}")
        send_feedback("Phase 1 ground-truth baseline starting")

        port_frame = f"task_board/{task.target_module_name}/{task.port_name}_link"
        plug_frame = f"{task.cable_name}/{task.plug_name}_link"

        self.get_logger().info(f"Target port frame: {port_frame}")
        self.get_logger().info(f"Current plug frame: {plug_frame}")

        # These frames require ground_truth:=true.
        for frame in [port_frame, plug_frame, "gripper/tcp"]:
            if not self._wait_for_tf("base_link", frame, timeout_sec=10.0):
                send_feedback(f"Missing TF frame: {frame}")
                return False

        try:
            port_tf = self._lookup_transform("base_link", port_frame).transform
        except TransformException as ex:
            self.get_logger().error(f"Could not look up port TF: {ex}")
            return False

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

            self.sleep_for(self.APPROACH_DT)

        self._log_observation_summary(get_observation, "after approach")

        # ---------------------------------------------------------------------
        # Phase B: slow geometric descent.
        #
        # This is intentionally close to CheatCode's known-working behavior:
        # reduce the z offset a little at a time while repeatedly recomputing
        # the TCP pose from current plug/TCP TF.
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

            # Log force occasionally. We are not using force logic yet; that is
            # Phase 4. This log helps us tune later.
            if int(abs(z_offset) * 10000) % 20 == 0:
                self._log_observation_summary(
                    get_observation,
                    f"descending z_offset={z_offset:+.4f}",
                )

            self.sleep_for(self.DESCENT_DT)

        # ---------------------------------------------------------------------
        # Phase C: hold briefly so Gazebo/contact settles.
        # ---------------------------------------------------------------------
        send_feedback("Holding final pose for stabilization")
        self.get_logger().info("Waiting for connector/contact to stabilize...")
        self.sleep_for(self.STABILIZE_SECONDS)

        self._log_observation_summary(get_observation, "final")
        self.get_logger().info("Phase1GroundTruth.insert_cable() complete")
        send_feedback("Phase 1 ground-truth baseline complete")

        return True