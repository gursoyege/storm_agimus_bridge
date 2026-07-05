from __future__ import annotations

import copy
import math
import os
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pinocchio as pin
import rclpy
from geometry_msgs.msg import Point, PoseArray, PoseStamped
from linear_feedback_controller_msgs.msg import Control, Sensor
from linear_feedback_controller_msgs_py.numpy_conversions import matrix_numpy_to_msg
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from std_msgs.msg import String
from tf2_ros import Buffer, TransformException
from tf2_ros.transform_listener import TransformListener
from visualization_msgs.msg import Marker, MarkerArray


@dataclass
class _StormCfg:
    storm_root: Path
    robot: str
    world_file: str
    task_file: str
    mppi_device: str


@dataclass
class _DynamicCollisionObservation:
    kind: str
    center: np.ndarray
    rotation: np.ndarray
    radius: float
    height: float


@dataclass
class _DynamicCollisionSlotSpec:
    kind: str
    radius: float
    height: float
    sphere_count: int


def _ensure_storm_on_path() -> Path:
    """Ensure `storm_kit` is importable and return the STORM repo root path."""
    candidates: list[Path] = []
    env_root = os.environ.get("STORM_ROOT")
    if env_root:
        candidates.append(Path(env_root))
    candidates.append(Path("/workspace/storm"))

    for root in candidates:
        if not root:
            continue
        if not (root / "storm_kit").exists():
            continue
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        try:
            import storm_kit  # noqa: F401

            return root
        except Exception:
            continue

    raise RuntimeError(
        "Unable to import STORM (`storm_kit`). Set env var `STORM_ROOT` to the STORM repo root "
        "(expected to contain a `storm_kit/` directory)."
    )


def _rpy_to_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = (float(rpy[0]), float(rpy[1]), float(rpy[2]))
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    # URDF uses fixed-axis roll-pitch-yaw (x-y-z), equivalent to Rz(yaw)*Ry(pitch)*Rx(roll).
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def _quat_to_matrix(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    n = float(qx * qx + qy * qy + qz * qz + qw * qw)
    if n < 1.0e-12:
        return np.eye(3, dtype=np.float64)
    s = 2.0 / n
    x, y, z, w = float(qx), float(qy), float(qz), float(qw)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1.0 - s * (yy + zz), s * (xy - wz), s * (xz + wy)],
            [s * (xy + wz), 1.0 - s * (xx + zz), s * (yz - wx)],
            [s * (xz - wy), s * (yz + wx), 1.0 - s * (xx + yy)],
        ],
        dtype=np.float64,
    )


def pose_msg_to_se3(pose) -> pin.SE3:
    R = _quat_to_matrix(pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w)
    t = np.array([pose.position.x, pose.position.y, pose.position.z], dtype=np.float64)
    return pin.SE3(R, t)


def transform_msg_to_se3(tf) -> pin.SE3:
    R = _quat_to_matrix(tf.rotation.x, tf.rotation.y, tf.rotation.z, tf.rotation.w)
    t = np.array([tf.translation.x, tf.translation.y, tf.translation.z], dtype=np.float64)
    return pin.SE3(R, t)


def se3_to_pose_msg(M: pin.SE3):
    from geometry_msgs.msg import Pose

    xyzquat = pin.SE3ToXYZQUAT(M)
    msg = Pose()
    msg.position.x = float(xyzquat[0])
    msg.position.y = float(xyzquat[1])
    msg.position.z = float(xyzquat[2])
    msg.orientation.x = float(xyzquat[3])
    msg.orientation.y = float(xyzquat[4])
    msg.orientation.z = float(xyzquat[5])
    msg.orientation.w = float(xyzquat[6])
    return msg


def _urdf_fixed_joint_parent_and_T_parent_child(urdf_xml: str, child_link: str) -> tuple[str, pin.SE3]:
    root = ET.fromstring(urdf_xml)
    for joint in root.findall("joint"):
        child = joint.find("child")
        if child is None or child.attrib.get("link") != child_link:
            continue

        parent = joint.find("parent")
        if parent is None or "link" not in parent.attrib:
            raise RuntimeError(f"URDF joint for child link '{child_link}' is missing <parent link=...>.")
        parent_link = str(parent.attrib["link"])

        origin = joint.find("origin")
        xyz = np.zeros(3, dtype=np.float64)
        rpy = np.zeros(3, dtype=np.float64)
        if origin is not None:
            if "xyz" in origin.attrib:
                xyz = np.fromstring(origin.attrib["xyz"], sep=" ", dtype=np.float64)
            if "rpy" in origin.attrib:
                rpy = np.fromstring(origin.attrib["rpy"], sep=" ", dtype=np.float64)

        if xyz.size != 3 or rpy.size != 3:
            raise RuntimeError(
                f"Invalid URDF <origin> for child link '{child_link}'. Parsed xyz={xyz} rpy={rpy}."
            )

        return parent_link, pin.SE3(_rpy_to_matrix(rpy), xyz)

    raise RuntimeError(f"Unable to find URDF joint with child link '{child_link}'.")


class StormMppiPlannerNode(Node):
    """Outer loop: STORM MPPI. Inner loop: torque LFC via linear_feedback_controller_msgs/Control."""

    def __init__(self) -> None:
        super().__init__("storm_mppi_planner")

        self._declare_parameters()

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self._armed: bool = bool(self.get_parameter("armed").value)

        self._goal_pose_msg: Optional[PoseStamped] = None
        self._objects_markers_msg: Optional[MarkerArray] = None
        self._collision_markers_msg: Optional[MarkerArray] = None
        self._collision_poses_msg: Optional[PoseArray] = None
        self._robot_description_msg: Optional[String] = None

        self._moving_joint_names: list[str] = []
        self._np_q: Optional[np.ndarray] = None
        self._np_dq: Optional[np.ndarray] = None

        # Causal low-pass state for the MPPI acceleration command used by inverse-dynamics
        # feedforward (see `_filter_ddq_for_feedforward`).
        self._ddq_ff_filt: Optional[np.ndarray] = None

        # Tier 3c: Ruckig jerk-limited reference smoothing state (see `_smooth_reference`).
        # REAL FIX (v2): the OTG is now stepped by a separate, faster timer
        # (`_ref_pub_timer` / `_ref_pub_tick`) decoupled from STORM's replanning rate, instead
        # of being re-targeted once per STORM tick before it could converge to the previous
        # target -- that v1 approach measured *worse* than no smoothing at all (smoothness
        # 153.75 vs 57.84 unfiltered) because retargeting every ~20ms never gave Ruckig room to
        # actually smooth anything; it just fought itself. `_tick()` now only updates the
        # *target* (`_ref_target_*`); the fast timer steps toward it and publishes.
        self._otg = None
        self._otg_input = None
        self._otg_output = None
        self._otg_initialized = False
        self._ref_pub_timer = None
        self._ref_target_q: Optional[np.ndarray] = None
        self._ref_target_dq: Optional[np.ndarray] = None
        self._ref_target_ddq: Optional[np.ndarray] = None
        self._ref_target_force_zero_gains = False
        self._ref_target_force_zero_feedforward = False

        self._storm_initialized = False
        self._storm_task = None
        self._storm_control_dt: Optional[float] = None
        self._storm_model_dt: Optional[float] = None
        self._storm_traj_base_dt: Optional[float] = None
        self._storm_traj_max_dt: Optional[float] = None
        self._storm_traj_base_ratio: Optional[float] = None
        self._storm_tensor_args = None
        self._storm_base_world: Optional[dict] = None
        self._world_inputs_dirty = False
        self._last_world_inputs_apply_time = 0.0
        self._last_world_inputs_signature: Optional[tuple] = None
        self._last_dynamic_objects_signature: Optional[tuple] = None
        self._dynamic_collision_slots_initialized = False
        self._dynamic_collision_slot_specs: list[_DynamicCollisionSlotSpec] = []
        self._dynamic_collision_slot_offsets: list[tuple[int, int]] = []
        self._dynamic_world_spheres_np: Optional[np.ndarray] = None
        self._dynamic_inactive_center = np.array([1.0e6, 1.0e6, 1.0e6], dtype=np.float32)
        self._last_collision_world_debug_signature: Optional[tuple] = None

        # Pinocchio reduced model for feedforward torque computation.
        self._pin_model: Optional[pin.Model] = None
        self._pin_data: Optional[pin.Data] = None
        self._pin_q_idx: list[int] = []
        self._pin_v_idx: list[int] = []

        # Fixed transforms for mapping a ROS TCP goal -> STORM ee_link goal.
        self._ros_T_hand_to_tcp: Optional[pin.SE3] = None
        self._storm_T_hand_to_ee: Optional[pin.SE3] = None
        self._last_goal_storm_ee_xyzquat: Optional[np.ndarray] = None

        self._t0_perf: float | None = None
        self._last_tick_perf: float | None = None
        self._tick_dt_ema: float | None = None
        self._cmd_wall_dt_ema: float | None = None
        self._mppi_opt_dt_ema: float | None = None

        qos_transient = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self._sub_robot_description = self.create_subscription(
            String, "robot_description", self._robot_description_callback, qos_profile=qos_transient
        )
        self._sub_sensor = self.create_subscription(
            Sensor,
            "sensor",
            self._sensor_callback,
            qos_profile=QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT),
        )
        goal_topic = str(self.get_parameter("goal_topic").value)
        self._sub_goal = self.create_subscription(
            PoseStamped,
            goal_topic,
            self._goal_callback,
            qos_profile=QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE),
        )
        include_objects_markers = bool(self.get_parameter("include_objects_markers_in_collision").value)
        objects_topic = str(self.get_parameter("objects_markers_topic").value).strip()
        self._sub_objects = None
        if include_objects_markers and objects_topic:
            self._sub_objects = self.create_subscription(
                MarkerArray,
                objects_topic,
                self._objects_callback,
                qos_profile=QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE),
            )

        collision_markers_topic = str(self.get_parameter("collision_markers_topic").value).strip()
        auto_discover_collision_topic = bool(self.get_parameter("auto_discover_collision_markers_topic").value)
        self._collision_markers_topic_resolved = ""
        self._sub_collision_markers = None
        self._collision_markers_autodiscovery_timer = None
        if collision_markers_topic:
            self._create_collision_markers_subscription(collision_markers_topic)
        elif auto_discover_collision_topic:
            self._collision_markers_autodiscovery_timer = self.create_timer(
                0.5, self._maybe_autodiscover_collision_markers_topic
            )
            self._maybe_autodiscover_collision_markers_topic()

        collision_poses_topic = str(self.get_parameter("collision_poses_topic").value).strip()
        self._sub_collision_poses = None
        if collision_poses_topic:
            self._sub_collision_poses = self.create_subscription(
                PoseArray,
                collision_poses_topic,
                self._collision_poses_callback,
                qos_profile=QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE),
            )

        self._pub_control = self.create_publisher(
            Control,
            "control",
            qos_profile=QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT),
        )
        self._pub_rollout_predictions = self.create_publisher(
            MarkerArray,
            str(self.get_parameter("rollout_predictions_topic").value),
            qos_profile=QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE),
        )
        self._pub_rollout_reference = self.create_publisher(
            MarkerArray,
            str(self.get_parameter("rollout_reference_topic").value),
            qos_profile=QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE),
        )
        self._rollout_viz_timer = None

        self._init_timer = self.create_timer(0.1, self._try_initialize)
        self._loop_timer = None

        objects_for_log = objects_topic if include_objects_markers and objects_topic else "<disabled>"
        if self._collision_markers_topic_resolved:
            collision_markers_for_log = self._collision_markers_topic_resolved
        elif collision_markers_topic:
            collision_markers_for_log = collision_markers_topic
        elif auto_discover_collision_topic:
            collision_markers_for_log = "<auto-discovery>"
        else:
            collision_markers_for_log = "<disabled>"

        self.get_logger().info(
            "storm_mppi_planner ready. "
            "Waiting for robot_description + sensor. "
            f"Collision inputs: objects_markers_topic={objects_for_log}, "
            f"collision_markers_topic={collision_markers_for_log}, "
            f"collision_poses_topic={collision_poses_topic or '<disabled>'}, "
            f"dynamic_collision_mode={bool(self.get_parameter('dynamic_collision_mode').value)}. "
            f"Goal topic: {goal_topic}."
        )

    def _declare_parameters(self) -> None:
        self.declare_parameter(
            "armed",
            False,
            ParameterDescriptor(
                description="Safety gate. When false, the node publishes a disarmed control policy."
            ),
        )
        self.declare_parameter(
            "goal_topic",
            "/goal_pose",
            ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description="PoseStamped topic providing end-effector goals (default matches RViz SetGoal tool).",
            ),
        )
        self.declare_parameter(
            "objects_markers_topic",
            "/storm_pick_place/objects_markers",
            ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description="Optional MarkerArray topic describing static scene objects for STORM. Empty string disables this input.",
            ),
        )
        self.declare_parameter(
            "include_objects_markers_in_collision",
            True,
            ParameterDescriptor(
                type=ParameterType.PARAMETER_BOOL,
                description="When true, objects_markers_topic is used as collision input.",
            ),
        )
        self.declare_parameter(
            "collision_markers_topic",
            "",
            ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description="Additional MarkerArray topic for collision primitives. "
                "CUBE and CYLINDER markers are converted to MPPI collision objects. "
                "Empty string enables autodiscovery.",
            ),
        )
        self.declare_parameter(
            "auto_discover_collision_markers_topic",
            True,
            ParameterDescriptor(
                type=ParameterType.PARAMETER_BOOL,
                description="Auto-discover a MarkerArray collision topic when collision_markers_topic is empty.",
            ),
        )
        self.declare_parameter(
            "collision_markers_topic_hint",
            "collision_markers",
            ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description="Hint used during collision MarkerArray topic autodiscovery.",
            ),
        )
        self.declare_parameter(
            "collision_poses_topic",
            "",
            ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description="Optional PoseArray topic converted to collision spheres.",
            ),
        )
        self.declare_parameter(
            "collision_pose_radius",
            0.05,
            ParameterDescriptor(
                type=ParameterType.PARAMETER_DOUBLE,
                description="Sphere radius (meters) used for each pose in collision_poses_topic.",
            ),
        )
        self.declare_parameter(
            "collision_pose_namespace",
            "rtcosmik_collision_pose",
            ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description="Namespace prefix for obstacles generated from collision_poses_topic.",
            ),
        )
        self.declare_parameter(
            "refresh_collision_world_on_updates",
            False,
            ParameterDescriptor(
                type=ParameterType.PARAMETER_BOOL,
                description="When true, live collision inputs trigger world primitive refresh in STORM.",
            ),
        )
        self.declare_parameter(
            "collision_update_rate_hz",
            30.0,
            ParameterDescriptor(
                type=ParameterType.PARAMETER_DOUBLE,
                description="Maximum refresh rate for live collision world updates.",
            ),
        )
        self.declare_parameter(
            "dynamic_collision_mode",
            False,
            ParameterDescriptor(
                type=ParameterType.PARAMETER_BOOL,
                description="Enable fixed-slot dynamic collision updates (no world SDF rebuild on each update).",
            ),
        )
        self.declare_parameter(
            "dynamic_collision_slots",
            0,
            ParameterDescriptor(
                type=ParameterType.PARAMETER_INTEGER,
                description="Number of fixed dynamic obstacle slots. 0 initializes slots from first collision message.",
            ),
        )
        self.declare_parameter(
            "dynamic_cylinder_spheres_per_object",
            5,
            ParameterDescriptor(
                type=ParameterType.PARAMETER_INTEGER,
                description="Number of spheres used to approximate each dynamic cylinder obstacle.",
            ),
        )
        self.declare_parameter(
            "use_base_world_collision",
            True,
            ParameterDescriptor(
                type=ParameterType.PARAMETER_BOOL,
                description="When true, keep collision objects from storm_world and merge live topics on top.",
            ),
        )
        self.declare_parameter(
            "ignore_collision_namespaces",
            ["pp_pick_object", "pp_place_target"],
            ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING_ARRAY,
                description="Marker namespaces to ignore when converting MarkerArray -> STORM collision objects. "
                "Pick/place visualization markers are ignored by default so the robot can reach the grasp pose.",
            ),
        )
        self.declare_parameter("base_frame_name", "world")
        self.declare_parameter("ee_frame_name", "fer_hand_tcp")
        self.declare_parameter("rate_hz", 50.0)
        self.declare_parameter(
            "debug_mppi_rate",
            False,
            ParameterDescriptor(
                type=ParameterType.PARAMETER_BOOL,
                description="When true, print throttled timing diagnostics including effective MPPI optimization rate.",
            ),
        )
        self.declare_parameter(
            "debug_collision_objects",
            False,
            ParameterDescriptor(
                type=ParameterType.PARAMETER_BOOL,
                description="When true, print collision primitives that STORM currently uses from scene/live inputs.",
            ),
        )
        self.declare_parameter(
            "debug_collision_objects_max_items",
            10,
            ParameterDescriptor(
                type=ParameterType.PARAMETER_INTEGER,
                description="Max number of cube/sphere names shown in collision debug output.",
            ),
        )

        # STORM config selection.
        self.declare_parameter(
            "storm_robot",
            "franka",
            ParameterDescriptor(description="Robot config name under storm/content/configs/gym/"),
        )
        self.declare_parameter(
            "storm_world",
            "collision_table.yml",
            ParameterDescriptor(description="World YAML name under storm/content/configs/gym/"),
        )
        self.declare_parameter(
            "storm_task_file",
            "franka_reacher.yml",
            ParameterDescriptor(description="Task YAML name under storm/content/configs/mpc/"),
        )
        self.declare_parameter(
            "mppi_device",
            "cuda:0",
            ParameterDescriptor(description='Torch device for STORM MPPI, e.g. "cpu" or "cuda:0".'),
        )

        # Inner-loop LFC gains (τ = τ_ff + [Kp Kd] * [q* - q; dq* - dq]).
        self.declare_parameter(
            "kp",
            [60.0],
            ParameterDescriptor(
                type=ParameterType.PARAMETER_DOUBLE_ARRAY,
                description="Proportional gains. Length 1 (broadcast) or nq.",
            ),
        )
        self.declare_parameter(
            "kd",
            [5.0],
            ParameterDescriptor(
                type=ParameterType.PARAMETER_DOUBLE_ARRAY,
                description="Derivative gains. Length 1 (broadcast) or nq.",
            ),
        )
        self.declare_parameter(
            "feedforward_mode",
            "inverse_dynamics",
            ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description=(
                    'Feedforward torque term: "none", "gravity" (g(q) only, zeroes v and a), '
                    '"coriolis_gravity" (C(q,v)v + g(q), real dq_des but zero acceleration -- '
                    'the limiting case of feedforward_accel_filter_coeff -> 0 reached by deletion '
                    'instead of attenuation: zero sensitivity to qdd_des noise, but zero inertial '
                    'feedforward), or "inverse_dynamics" (full RNEA, the default).'
                ),
            ),
        )
        self.declare_parameter(
            "feedforward_accel_filter_coeff",
            0.3,
            ParameterDescriptor(
                type=ParameterType.PARAMETER_DOUBLE,
                description=(
                    "Causal EMA low-pass coefficient (0,1] applied to the MPPI acceleration "
                    "command (qdd_des) before it is used for inverse-dynamics feedforward torque "
                    "(tau_ff = RNEA(q_des, dq_des, filtered_qdd_des)). MPPI's per-tick mean-action "
                    "estimate has sampling variance in acceleration space that the position/velocity "
                    "command trajectory hides (it is produced by integrating qdd_des, which smooths "
                    "it twice) but which otherwise passes straight through RNEA's mass-matrix "
                    "multiplication into raw commanded-torque jitter. 1.0 disables filtering "
                    "(reproduces pre-fix behavior). Lower = smoother feedforward but more lag "
                    "relative to true acceleration changes."
                ),
            ),
        )
        self.declare_parameter(
            "disarmed_behavior",
            "hold",
            ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description='When armed=false: "hold" publishes a stabilizing policy around current state; "zero_torque" publishes zero gains and zero feedforward.',
            ),
        )

        # Tier 2b: joint-space reference-error clamp (SERL-style reference-ball trick, joint
        # space instead of Cartesian since the LFC's diff_state_ is joint-space). Disabled by
        # default (<=0). Prevents a stale/jumped q_des from producing a large K*(q_des-q_meas)
        # torque kick; published q_des is pulled back toward the last known measured q so that
        # ||q_des - q_meas|| <= max_joint_ref_error_rad before publishing.
        self.declare_parameter(
            "max_joint_ref_error_rad",
            0.0,
            ParameterDescriptor(
                type=ParameterType.PARAMETER_DOUBLE,
                description="Clamp |q_des - q_meas| per joint to this radius before publishing. <=0 disables.",
            ),
        )

        # Tier 3b: Cartesian-impedance-equivalent feedback gain, computed entirely in the bridge.
        # The LFC's control law is tau = tau_ff + K @ [q_des(-)q_meas ; dq_des-dq_meas] with K a
        # generic (joint_nv x 2*nv) matrix -- it never assumed K had to be diagonal/joint-space.
        # For small tracking error, x_des - x_meas ~= J(q)(q_des - q_meas) (first-order FK), so
        # publishing K = [J^T Kx J | J^T Dx J] approximates true Cartesian impedance without any
        # change to linear_feedback_controller's C++. A small uniform joint-space term is added
        # on top since J^T Kx J is rank-deficient (6D task, 7 DoF arm) and would otherwise leave
        # the 1-DoF null space direction with zero feedback stiffness (drift risk).
        self.declare_parameter(
            "feedback_mode",
            "joint",
            ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description=(
                    '"joint" (default, diag kp/kd as before -- recommended) or "cartesian" '
                    "(J^T Kx J translation/rotation impedance + a stiffness/damping floor). "
                    "cartesian is NOT recommended for this task: after 3 fix iterations it "
                    "either gets stuck (weak floor) or completes with ~170x worse smoothness "
                    "than no filtering at all (strong floor), because K is rebuilt from a live "
                    "Jacobian every tick and is itself a new, q-dependent noise source that "
                    "feedback_mode=joint's constant diag(kp,kd) does not have. See "
                    "_compute_cartesian_feedback_gain's docstring for the full story."
                ),
            ),
        )
        self.declare_parameter(
            "cartesian_kp",
            [120.0, 120.0, 180.0, 10.0, 10.0, 8.0],
            ParameterDescriptor(
                type=ParameterType.PARAMETER_DOUBLE_ARRAY,
                description="Cartesian stiffness [x,y,z,rx,ry,rz] (N/m, N/m, N/m, Nm/rad, Nm/rad, Nm/rad). Only used when feedback_mode=cartesian.",
            ),
        )
        self.declare_parameter(
            "cartesian_kd",
            [20.0, 20.0, 25.0, 2.0, 2.0, 1.8],
            ParameterDescriptor(
                type=ParameterType.PARAMETER_DOUBLE_ARRAY,
                description="Cartesian damping [x,y,z,rx,ry,rz]. Only used when feedback_mode=cartesian.",
            ),
        )
        self.declare_parameter(
            "cartesian_null_kp",
            60.0,
            ParameterDescriptor(
                type=ParameterType.PARAMETER_DOUBLE,
                description=(
                    "v3: a flat (unprojected) joint-space stiffness floor added to J^T Kx J in "
                    "every direction, not just the null space. v1's flat floor (set to 8) caused "
                    "a phase-timeout + Gazebo crash; v2's fix (project it onto only the null "
                    "space) fixed the crash but the arm still got stuck, because the *task-space* "
                    "eigenvalues of J^T Kx J can themselves collapse to ~0.01 at some "
                    "configurations -- a problem strictly outside the null space that a "
                    "null-projected term can never fix. 60 fixes completion (2/2 cycles, no "
                    "crash, no timeout) but makes smoothness ~170x WORSE than no filtering at "
                    "all (~9769-10034 vs 57.84), because K is rebuilt from a live Jacobian every "
                    "tick and a larger floor transmits more of that per-tick jitter into torque, "
                    "amplified quadratically through J. Raising or lowering this value cannot "
                    "escape that tradeoff -- see _compute_cartesian_feedback_gain's docstring. "
                    "feedback_mode=cartesian is not recommended; use feedback_mode=joint."
                ),
            ),
        )
        self.declare_parameter(
            "cartesian_null_kd",
            15.0,
            ParameterDescriptor(
                type=ParameterType.PARAMETER_DOUBLE,
                description=(
                    "v3: flat damping floor added to J^T Dx J in every direction (see "
                    "cartesian_null_kp). 15 is close to critically damped (2*sqrt(60)~=15.5) for "
                    "the floor-dominated (worst-case) regime. Only used when feedback_mode=cartesian."
                ),
            ),
        )

        # Tier 3c: optional in-bridge jerk-limited reference smoothing (Ruckig OTG). REAL FIX
        # (v2): publishing is decoupled from STORM's replan rate (rate_hz). _tick() only updates
        # a stored *target*; a separate, faster timer (ref_smoothing_rate_hz) steps Ruckig toward
        # that target and publishes. The original v1 (re-target Ruckig once per STORM tick, same
        # rate as rate_hz) measured *worse* than no smoothing (153.75 vs 57.84 baseline) because
        # the target moved again before Ruckig had room to converge to it -- it never actually
        # got to smooth anything, it just chased a moving point every cycle. This is exactly the
        # "1kHz reference governor" idea, scoped down to one extra timer in this same node rather
        # than a separate process/package.
        self.declare_parameter(
            "enable_reference_smoothing",
            False,
            ParameterDescriptor(
                type=ParameterType.PARAMETER_BOOL,
                description="When true, route the published reference through a Ruckig jerk-limited generator stepped by ref_smoothing_rate_hz, decoupled from rate_hz.",
            ),
        )
        self.declare_parameter(
            "ref_smoothing_rate_hz",
            250.0,
            ParameterDescriptor(
                type=ParameterType.PARAMETER_DOUBLE,
                description="Rate at which the Ruckig OTG is stepped and republished, decoupled from rate_hz (STORM's replan rate). Must be >> rate_hz for the OTG to have room to converge between target updates.",
            ),
        )
        self.declare_parameter(
            "ref_smoothing_max_vel",
            [1.0],
            ParameterDescriptor(
                type=ParameterType.PARAMETER_DOUBLE_ARRAY,
                description="Ruckig max velocity (rad/s), length 1 (broadcast) or nq. Hardware ceiling is 2.175 (J1-4) / 2.61 (J5-7) rad/s.",
            ),
        )
        self.declare_parameter(
            "ref_smoothing_max_acc",
            [2.5],
            ParameterDescriptor(
                type=ParameterType.PARAMETER_DOUBLE_ARRAY,
                description="Ruckig max acceleration (rad/s^2), length 1 or nq. Hardware ceiling is 10.0 rad/s^2 uniform; STORM's own model.max_acc is 5.0.",
            ),
        )
        self.declare_parameter(
            "ref_smoothing_max_jerk",
            [40.0],
            ParameterDescriptor(
                type=ParameterType.PARAMETER_DOUBLE_ARRAY,
                description="Ruckig max jerk (rad/s^3), length 1 or nq. Hardware ceiling is 5000 rad/s^3 uniform -- this default is deliberately far below it.",
            ),
        )

        # Rollout visualization: STORM's MPPI already computes top_trajs (the lowest-cost
        # candidate end-effector position sequences each tick, exposed as a property all the
        # way from mppi.py through the multiprocess queue -- see top_trajs' own fix this
        # session: torch.topk defaulted to largest=True against a lower-is-better cost, so this
        # used to silently hold the *worst* 10 rollouts despite the name). Nothing in this
        # workspace was actually publishing this data; pick_place.rviz/vids.rviz already have
        # "MPC Reference"/"MPC Predictions" MarkerArray display slots wired to these exact topic
        # names (left over from agimus_controller_ros's mpc_debugger_node convention, which is
        # for the unrelated crocoddyl-based controller and isn't used by this launch graph), but
        # both were Enabled: false since nothing published to them.
        self.declare_parameter(
            "enable_rollout_visualization",
            True,
            ParameterDescriptor(
                type=ParameterType.PARAMETER_BOOL,
                description="Publish STORM's top-10 candidate rollouts and the selected best one as rviz MarkerArrays. Pure visualization, no control-path cost.",
            ),
        )
        self.declare_parameter(
            "rollout_viz_rate_hz",
            10.0,
            ParameterDescriptor(
                type=ParameterType.PARAMETER_DOUBLE,
                description="Publish rate for rollout markers, decoupled from rate_hz (the 50Hz control loop) since this is for human viewing only.",
            ),
        )
        self.declare_parameter(
            "rollout_predictions_topic",
            "/mpc_states_prediction_markers",
            ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description="Topic for the top-10 candidate rollout MarkerArray. Matches the pre-existing (previously unpublished) rviz display slot.",
            ),
        )
        self.declare_parameter(
            "rollout_reference_topic",
            "/mpc_states_reference_markers",
            ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description="Topic for the single best/selected rollout MarkerArray. Matches the pre-existing (previously unpublished) rviz display slot.",
            ),
        )

    def _robot_description_callback(self, msg: String) -> None:
        self._robot_description_msg = msg

    def _sensor_callback(self, msg: Sensor) -> None:
        js = msg.joint_state
        if not js.name:
            return
        if not self._moving_joint_names:
            self._moving_joint_names = list(js.name)
            self.get_logger().info(
                f"Derived moving_joint_names from sensor ({len(self._moving_joint_names)} joints)."
            )
        self._np_q = np.array(js.position, dtype=np.float64)
        self._np_dq = np.array(js.velocity, dtype=np.float64) if js.velocity else np.zeros_like(self._np_q)

    def _goal_callback(self, msg: PoseStamped) -> None:
        self._goal_pose_msg = msg

    def _create_collision_markers_subscription(self, topic_name: str) -> None:
        if not topic_name:
            return
        if self._sub_collision_markers is not None:
            return
        self._sub_collision_markers = self.create_subscription(
            MarkerArray,
            topic_name,
            self._collision_markers_callback,
            qos_profile=QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE),
        )
        self._collision_markers_topic_resolved = topic_name
        if self._collision_markers_autodiscovery_timer is not None:
            self.destroy_timer(self._collision_markers_autodiscovery_timer)
            self._collision_markers_autodiscovery_timer = None
        self.get_logger().info(f"Using collision MarkerArray topic: {topic_name}")

    def _maybe_autodiscover_collision_markers_topic(self) -> None:
        if self._sub_collision_markers is not None:
            return
        if not bool(self.get_parameter("auto_discover_collision_markers_topic").value):
            return

        marker_topics: list[str] = []
        for topic_name, topic_types in self.get_topic_names_and_types():
            if "visualization_msgs/msg/MarkerArray" in topic_types:
                marker_topics.append(str(topic_name))
        if not marker_topics:
            return

        hint = str(self.get_parameter("collision_markers_topic_hint").value).strip()
        selected_topic = ""
        if hint:
            if hint.startswith("/"):
                if hint in marker_topics:
                    selected_topic = hint
            else:
                hinted = sorted([topic_name for topic_name in marker_topics if hint in topic_name])
                if hinted:
                    selected_topic = hinted[0]

        if not selected_topic:
            collision_like = sorted(
                [topic_name for topic_name in marker_topics if "collision" in topic_name.lower()]
            )
            if collision_like:
                selected_topic = collision_like[0]

        if not selected_topic:
            return
        self._create_collision_markers_subscription(selected_topic)

    def _objects_callback(self, msg: MarkerArray) -> None:
        if not bool(self.get_parameter("include_objects_markers_in_collision").value):
            return
        self._objects_markers_msg = msg
        self._world_inputs_dirty = True

    def _collision_markers_callback(self, msg: MarkerArray) -> None:
        self._collision_markers_msg = msg
        self._world_inputs_dirty = True

    def _collision_poses_callback(self, msg: PoseArray) -> None:
        self._collision_poses_msg = msg
        self._world_inputs_dirty = True

    def _try_initialize(self) -> None:
        if self._storm_initialized:
            return

        self._armed = bool(self.get_parameter("armed").value)

        if self._robot_description_msg is None or self._np_q is None:
            return
        if (
            bool(self.get_parameter("include_objects_markers_in_collision").value)
            and self._sub_objects is not None
            and self._objects_markers_msg is None
        ):
            self.get_logger().info("Waiting for object markers...", throttle_duration_sec=2.0)
            return

        ee_frame_name = str(self.get_parameter("ee_frame_name").value)
        base_frame_name = str(self.get_parameter("base_frame_name").value)
        rate_hz = float(self.get_parameter("rate_hz").value)
        if rate_hz <= 0.0:
            raise ValueError("rate_hz must be > 0")

        storm_root = _ensure_storm_on_path()
        cfg = _StormCfg(
            storm_root=storm_root,
            robot=str(self.get_parameter("storm_robot").value),
            world_file=str(self.get_parameter("storm_world").value),
            task_file=str(self.get_parameter("storm_task_file").value),
            mppi_device=str(self.get_parameter("mppi_device").value),
        )

        self._initialize_pinocchio(self._robot_description_msg.data, self._moving_joint_names)
        self._initialize_storm(cfg)

        self._storm_initialized = True
        self.destroy_timer(self._init_timer)
        self._init_timer = None
        self._log_timing_consistency(rate_hz=rate_hz)
        self._loop_timer = self.create_timer(1.0 / rate_hz, self._tick)
        # Always created; _ref_pub_tick() itself no-ops cheaply when enable_reference_smoothing
        # is false (the common/default case), avoiding dynamic timer create/destroy complexity
        # if the param is ever flipped at runtime.
        ref_smoothing_rate_hz = float(self.get_parameter("ref_smoothing_rate_hz").value)
        if ref_smoothing_rate_hz <= 0.0:
            raise ValueError("ref_smoothing_rate_hz must be > 0")
        self._ref_pub_timer = self.create_timer(1.0 / ref_smoothing_rate_hz, self._ref_pub_tick)
        rollout_viz_rate_hz = float(self.get_parameter("rollout_viz_rate_hz").value)
        if rollout_viz_rate_hz <= 0.0:
            raise ValueError("rollout_viz_rate_hz must be > 0")
        self._rollout_viz_timer = self.create_timer(1.0 / rollout_viz_rate_hz, self._publish_rollout_markers)
        self.get_logger().info(
            f"storm_mppi_planner initialized. Publishing linear_feedback_controller_msgs/Control at ~{rate_hz:.1f} Hz "
            f"(armed={self._armed}), reference-smoothing fast timer ~{ref_smoothing_rate_hz:.1f} Hz "
            f"(enabled={bool(self.get_parameter('enable_reference_smoothing').value)}). base={base_frame_name} ee={ee_frame_name}."
        )

    def _initialize_pinocchio(self, urdf_xml: str, moving_joint_names: list[str]) -> None:
        try:
            full_model = pin.buildModelFromXML(urdf_xml)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Failed to build Pinocchio model from robot_description XML: {exc}") from exc

        q0 = pin.neutral(full_model)
        joints_to_lock: list[int] = []
        for j_id in range(1, int(full_model.njoints)):  # 0 is universe
            name = str(full_model.names[j_id])
            if name not in moving_joint_names:
                joints_to_lock.append(int(j_id))

        model = pin.buildReducedModel(full_model, joints_to_lock, q0)
        data = model.createData()

        pin_q_idx: list[int] = []
        pin_v_idx: list[int] = []
        for name in moving_joint_names:
            j_id = int(model.getJointId(str(name)))
            if j_id <= 0:
                raise RuntimeError(f"Joint '{name}' not found in reduced Pinocchio model.")
            joint = model.joints[j_id]
            if int(joint.nq) != 1 or int(joint.nv) != 1:
                raise RuntimeError(
                    f"Joint '{name}' has nq={int(joint.nq)} nv={int(joint.nv)}; expected 1-DoF joints."
                )
            pin_q_idx.append(int(joint.idx_q))
            pin_v_idx.append(int(joint.idx_v))

        self._pin_model = model
        self._pin_data = data
        self._pin_q_idx = pin_q_idx
        self._pin_v_idx = pin_v_idx

    def _initialize_storm(self, cfg: _StormCfg) -> None:
        import torch

        try:
            device = torch.device(cfg.mppi_device)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Invalid mppi_device '{cfg.mppi_device}': {exc}") from exc
        if device.type == "cuda" and not torch.cuda.is_available():
            self.get_logger().warn("CUDA requested for MPPI but torch.cuda.is_available() is false. Falling back to CPU.")
            device = torch.device("cpu")
        self._storm_tensor_args = {"device": device, "dtype": torch.float32}

        from storm_kit.mpc.task.reacher_task import ReacherTask
        from storm_kit.util_file import get_gym_configs_path, join_path, load_yaml

        base_world = load_yaml(join_path(get_gym_configs_path(), cfg.world_file))
        self._storm_base_world = copy.deepcopy(base_world)
        dynamic_collision_mode = bool(self.get_parameter("dynamic_collision_mode").value)
        world_params = self._build_world_params(
            self._storm_base_world,
            include_live_collision_inputs=not dynamic_collision_mode,
        )
        self._maybe_log_world_collision_objects(world_params, source="initialize")

        # Resolve and inspect task YAML directly so we can prove exactly what file/values are being fed to STORM.
        if Path(cfg.task_file).is_absolute():
            resolved_task_cfg_path = Path(cfg.task_file)
        else:
            from storm_kit.util_file import get_mpc_configs_path

            resolved_task_cfg_path = Path(join_path(get_mpc_configs_path(), cfg.task_file))
        raw_cat_enabled = None
        raw_cat_sb_enabled = None
        raw_primitive_weight = None
        raw_collision_weight = None
        if resolved_task_cfg_path.is_file():
            raw_task_cfg = load_yaml(str(resolved_task_cfg_path)) or {}
            raw_cost_cfg = raw_task_cfg.get("cost", {})
            raw_mppi_cfg = raw_task_cfg.get("mppi", {})
            raw_primitive_weight = float(raw_cost_cfg.get("primitive_collision", {}).get("weight", math.nan))
            raw_collision_weight = float(raw_cost_cfg.get("collision", {}).get("weight", math.nan))
            raw_cat_enabled = bool(raw_mppi_cfg.get("cat_primitive_collision", {}).get("enabled", False))
            raw_cat_sb_enabled = bool(raw_mppi_cfg.get("cat_state_bound", {}).get("enabled", False))
            self.get_logger().info(
                "Raw STORM task YAML (pre-load): "
                f"path='{resolved_task_cfg_path}', "
                f"cost.collision.weight={raw_collision_weight}, "
                f"cost.primitive_collision.weight={raw_primitive_weight}, "
                f"mppi.cat_primitive_collision.enabled={raw_cat_enabled}, "
                f"mppi.cat_state_bound.enabled={raw_cat_sb_enabled}."
            )
        else:
            self.get_logger().warn(
                f"Resolved STORM task YAML path does not exist: '{resolved_task_cfg_path}'. "
                "STORM may fail to initialize or load a different task file."
            )

        task = ReacherTask(
            task_file=cfg.task_file,
            robot_file=f"{cfg.robot}.yml",
            world_file=cfg.world_file,
            tensor_args=self._storm_tensor_args,
            world_params=world_params,
        )
        cost_cfg = task.exp_params.get("cost", {})
        mppi_cfg = task.exp_params.get("mppi", {})
        primitive_weight = float(cost_cfg.get("primitive_collision", {}).get("weight", math.nan))
        collision_weight = float(cost_cfg.get("collision", {}).get("weight", math.nan))
        cat_pc_enabled = bool(mppi_cfg.get("cat_primitive_collision", {}).get("enabled", False))
        cat_sb_enabled = bool(mppi_cfg.get("cat_state_bound", {}).get("enabled", False))
        self.get_logger().info(
            "Loaded STORM task config: "
            f"storm_task_file='{cfg.task_file}' (resolved='{resolved_task_cfg_path}'), "
            f"cost.collision.weight={collision_weight}, "
            f"cost.primitive_collision.weight={primitive_weight}, "
            f"mppi.cat_primitive_collision.enabled={cat_pc_enabled}, "
            f"mppi.cat_state_bound.enabled={cat_sb_enabled}."
        )
        if (
            raw_cat_enabled is not None
            and raw_cat_sb_enabled is not None
            and raw_primitive_weight is not None
            and raw_collision_weight is not None
            and (
                bool(raw_cat_enabled) != bool(cat_pc_enabled)
                or bool(raw_cat_sb_enabled) != bool(cat_sb_enabled)
                or not math.isclose(float(raw_primitive_weight), float(primitive_weight), rel_tol=0.0, abs_tol=1.0e-12)
                or not math.isclose(float(raw_collision_weight), float(collision_weight), rel_tol=0.0, abs_tol=1.0e-12)
            )
        ):
            self.get_logger().warn(
                "Mismatch between raw YAML values and STORM-loaded values. "
                "This indicates task config overrides or a different loader path is in effect."
            )
        self._storm_task = task
        self._storm_control_dt = float(task.exp_params["control_dt"])
        self._storm_model_dt = float(task.exp_params["model"]["dt"])
        dt_traj_params = dict(task.exp_params["model"].get("dt_traj_params", {}))
        if "base_dt" in dt_traj_params:
            self._storm_traj_base_dt = float(dt_traj_params["base_dt"])
        if "max_dt" in dt_traj_params:
            self._storm_traj_max_dt = float(dt_traj_params["max_dt"])
        if "base_ratio" in dt_traj_params:
            self._storm_traj_base_ratio = float(dt_traj_params["base_ratio"])

        # Precompute fixed transforms for converting a ROS TCP goal into STORM's internal ee_link goal.
        ee_link_name = str(task.exp_params["model"]["ee_link_name"])
        storm_urdf_rel = str(task.exp_params["model"]["urdf_path"])
        storm_urdf_abs = cfg.storm_root / "content" / "assets" / storm_urdf_rel
        if not storm_urdf_abs.exists():
            raise RuntimeError(f"STORM URDF not found: {storm_urdf_abs}")

        with open(storm_urdf_abs, "r") as f:
            storm_urdf_xml = f.read()
        ros_urdf_xml = str(self._robot_description_msg.data)

        ros_parent, ros_T_hand_to_tcp = _urdf_fixed_joint_parent_and_T_parent_child(
            ros_urdf_xml, str(self.get_parameter("ee_frame_name").value)
        )
        storm_parent, storm_T_hand_to_ee = _urdf_fixed_joint_parent_and_T_parent_child(storm_urdf_xml, ee_link_name)

        self.get_logger().info(
            f"Goal frame mapping: ROS tcp parent='{ros_parent}' -> STORM ee parent='{storm_parent}' using fixed-joint transforms."
        )
        self._ros_T_hand_to_tcp = ros_T_hand_to_tcp
        self._storm_T_hand_to_ee = storm_T_hand_to_ee
        if dynamic_collision_mode:
            base_frame_name = str(self.get_parameter("base_frame_name").value)
            observations = self._collect_dynamic_collision_observations(base_frame_name)
            self._initialize_dynamic_collision_slots(observations)
            dynamic_world_spheres = self._update_dynamic_world_spheres_from_observations(observations)
            if dynamic_world_spheres is not None:
                self._storm_task.update_params(dynamic_world_spheres=dynamic_world_spheres)
                self._maybe_log_dynamic_collision_spheres(dynamic_world_spheres, source="initialize")
            self._last_world_inputs_signature = None
            self._last_dynamic_objects_signature = self._marker_array_signature(self._objects_markers_msg)
        else:
            self._last_world_inputs_signature = self._compute_world_inputs_signature()
            self._last_dynamic_objects_signature = None
        self._world_inputs_dirty = False

    def _log_timing_consistency(self, *, rate_hz: float) -> None:
        if self._storm_control_dt is None:
            return
        timer_dt = 1.0 / float(rate_hz)
        msg = (
            f"Timing summary: loop rate_hz={rate_hz:.3f} (dt={timer_dt:.4f}s), "
            f"control_dt={self._storm_control_dt:.4f}s"
        )
        if self._storm_model_dt is not None:
            msg += f", model.dt={self._storm_model_dt:.4f}s"
        if self._storm_traj_base_dt is not None:
            msg += f", dt_traj.base_dt={self._storm_traj_base_dt:.4f}s"
        if self._storm_traj_max_dt is not None:
            msg += f", dt_traj.max_dt={self._storm_traj_max_dt:.4f}s"
        if self._storm_traj_base_ratio is not None:
            msg += f", dt_traj.base_ratio={self._storm_traj_base_ratio:.3f}"
        self.get_logger().info(msg)

        tol = 1.0e-3
        if abs(timer_dt - self._storm_control_dt) > tol:
            self.get_logger().warn(
                "Timing mismatch: planner loop period (1/rate_hz) differs from STORM control_dt by "
                f"{abs(timer_dt - self._storm_control_dt):.4f}s. Align `rate_hz` and `control_dt` for best tracking."
            )
        if self._storm_model_dt is not None and abs(self._storm_model_dt - self._storm_control_dt) > tol:
            self.get_logger().warn(
                "Timing mismatch: STORM model.dt differs from control_dt by "
                f"{abs(self._storm_model_dt - self._storm_control_dt):.4f}s."
            )
        if self._storm_traj_base_dt is not None and self._storm_model_dt is not None and abs(self._storm_traj_base_dt - self._storm_model_dt) > tol:
            self.get_logger().warn(
                "Timing mismatch: dt_traj.base_dt differs from model.dt by "
                f"{abs(self._storm_traj_base_dt - self._storm_model_dt):.4f}s."
            )

    def _pose_in_base_frame(self, pose, source_frame: str, base_frame_name: str) -> Optional[pin.SE3]:
        pose_M = pose_msg_to_se3(pose)
        if not source_frame or source_frame == base_frame_name:
            return pose_M
        try:
            tf = self._tf_buffer.lookup_transform(base_frame_name, source_frame, Time())
        except TransformException as exc:
            self.get_logger().warn(
                f"TF failed transforming collision object from '{source_frame}' to '{base_frame_name}': {exc}",
                throttle_duration_sec=2.0,
            )
            return None
        return transform_msg_to_se3(tf.transform) * pose_M

    def _transform_position_to_base_frame(
        self,
        *,
        position: list[float],
        source_frame: str,
        base_frame_name: str,
        object_name: str,
    ) -> list[float]:
        position_arr = np.asarray(position, dtype=np.float64).reshape(-1)
        if position_arr.size != 3:
            raise ValueError(
                f"World collision object '{object_name}' has invalid `position` length {position_arr.size}; expected 3."
            )
        if not source_frame or source_frame == base_frame_name:
            return [float(v) for v in position_arr]

        try:
            tf = self._tf_buffer.lookup_transform(base_frame_name, source_frame, Time())
        except TransformException as exc:
            raise ValueError(
                f"Failed to transform world collision object '{object_name}' from frame "
                f"'{source_frame}' to '{base_frame_name}': {exc}"
            ) from exc

        base_T_source = transform_msg_to_se3(tf.transform)
        base_position = base_T_source.translation + (base_T_source.rotation @ position_arr)
        return [float(v) for v in base_position]

    def _transform_pose_to_base_frame(
        self,
        *,
        pose: list[float],
        source_frame: str,
        base_frame_name: str,
        object_name: str,
    ) -> list[float]:
        pose_arr = np.asarray(pose, dtype=np.float64).reshape(-1)
        if pose_arr.size != 7:
            raise ValueError(
                f"World collision object '{object_name}' has invalid `pose` length {pose_arr.size}; expected 7."
            )
        pose_M = pin.XYZQUATToSE3(pose_arr)
        if not source_frame or source_frame == base_frame_name:
            xyzquat = pin.SE3ToXYZQUAT(pose_M)
            return [float(v) for v in xyzquat]

        try:
            tf = self._tf_buffer.lookup_transform(base_frame_name, source_frame, Time())
        except TransformException as exc:
            raise ValueError(
                f"Failed to transform world collision object '{object_name}' from frame "
                f"'{source_frame}' to '{base_frame_name}': {exc}"
            ) from exc

        base_T_source = transform_msg_to_se3(tf.transform)
        base_pose = base_T_source * pose_M
        xyzquat = pin.SE3ToXYZQUAT(base_pose)
        return [float(v) for v in xyzquat]

    def _resolve_world_collision_frames(self, world: dict, *, base_frame_name: str) -> dict:
        world_model = dict(world.get("world_model", {}))
        coll_objs = dict(world_model.get("coll_objs", {}))
        sphere_objs = dict(coll_objs.get("sphere", {}))
        cube_objs = dict(coll_objs.get("cube", {}))

        resolved_spheres: dict[str, dict] = {}
        for name, spec in sphere_objs.items():
            sphere_spec = dict(spec)
            source_frame = str(sphere_spec.pop("frame", "")).strip()
            if "position" in sphere_spec:
                sphere_spec["position"] = self._transform_position_to_base_frame(
                    position=sphere_spec["position"],
                    source_frame=source_frame,
                    base_frame_name=base_frame_name,
                    object_name=str(name),
                )
            resolved_spheres[str(name)] = sphere_spec

        resolved_cubes: dict[str, dict] = {}
        for name, spec in cube_objs.items():
            cube_spec = dict(spec)
            source_frame = str(cube_spec.pop("frame", "")).strip()
            if "pose" in cube_spec:
                cube_spec["pose"] = self._transform_pose_to_base_frame(
                    pose=cube_spec["pose"],
                    source_frame=source_frame,
                    base_frame_name=base_frame_name,
                    object_name=str(name),
                )
            resolved_cubes[str(name)] = cube_spec

        coll_objs["sphere"] = resolved_spheres
        coll_objs["cube"] = resolved_cubes
        world_model["coll_objs"] = coll_objs
        world["world_model"] = world_model
        return world

    def _merge_marker_array_into_world(
        self,
        *,
        markers_msg: Optional[MarkerArray],
        spheres: dict,
        cubes: dict,
        base_frame_name: str,
        name_prefix: str,
        ignored_ns: set[str],
        allowed_marker_types: Optional[set[int]] = None,
    ) -> None:
        if markers_msg is None:
            return

        for marker in markers_msg.markers:
            if marker.action in (Marker.DELETE, Marker.DELETEALL):
                continue
            if marker.ns and str(marker.ns) in ignored_ns:
                continue
            if allowed_marker_types is not None and int(marker.type) not in allowed_marker_types:
                continue

            marker_frame = str(marker.header.frame_id) if marker.header.frame_id else base_frame_name
            marker_M = self._pose_in_base_frame(marker.pose, marker_frame, base_frame_name)
            if marker_M is None:
                continue

            base_name = f"{marker.ns}_{int(marker.id)}" if marker.ns else f"obj_{int(marker.id)}"
            name = f"{name_prefix}_{base_name}" if name_prefix else base_name

            if marker.type == Marker.SPHERE:
                radius = 0.5 * float(max(marker.scale.x, marker.scale.y, marker.scale.z))
                if radius <= 0.0:
                    continue
                spheres[name] = {
                    "radius": float(radius),
                    "position": [
                        float(marker_M.translation[0]),
                        float(marker_M.translation[1]),
                        float(marker_M.translation[2]),
                    ],
                }
            elif marker.type in (Marker.CUBE, Marker.CYLINDER):
                dims = [float(marker.scale.x), float(marker.scale.y), float(marker.scale.z)]
                if min(dims) <= 0.0:
                    continue
                q = pin.SE3ToXYZQUAT(marker_M)
                cubes[name] = {
                    "dims": dims,
                    "pose": [
                        float(q[0]),
                        float(q[1]),
                        float(q[2]),
                        float(q[3]),
                        float(q[4]),
                        float(q[5]),
                        float(q[6]),
                    ],
                }

    def _merge_pose_array_into_world(
        self,
        *,
        poses_msg: Optional[PoseArray],
        spheres: dict,
        base_frame_name: str,
    ) -> None:
        if poses_msg is None:
            return

        source_frame = str(poses_msg.header.frame_id) if poses_msg.header.frame_id else base_frame_name
        radius = float(self.get_parameter("collision_pose_radius").value)
        if radius <= 0.0:
            raise ValueError("collision_pose_radius must be > 0")
        namespace = str(self.get_parameter("collision_pose_namespace").value).strip() or "collision_pose"

        for idx, pose in enumerate(poses_msg.poses):
            pose_M = self._pose_in_base_frame(pose, source_frame, base_frame_name)
            if pose_M is None:
                continue
            name = f"{namespace}_{idx}"
            spheres[name] = {
                "radius": radius,
                "position": [
                    float(pose_M.translation[0]),
                    float(pose_M.translation[1]),
                    float(pose_M.translation[2]),
                ],
            }

    def _build_world_params(self, base_world: dict, *, include_live_collision_inputs: bool = True) -> dict:
        base_frame_name = str(self.get_parameter("base_frame_name").value)
        use_base_world = bool(self.get_parameter("use_base_world_collision").value)
        if use_base_world:
            world = copy.deepcopy(base_world)
            world = self._resolve_world_collision_frames(world, base_frame_name=base_frame_name)
        else:
            world = {"world_model": {"coll_objs": {"sphere": {}, "cube": {}}}}
        world_model = dict(world.get("world_model", {}))
        coll_objs = dict(world_model.get("coll_objs", {}))
        spheres = dict(coll_objs.get("sphere", {}))
        cubes = dict(coll_objs.get("cube", {}))

        ignored_ns = {str(x) for x in self.get_parameter("ignore_collision_namespaces").value}

        if bool(self.get_parameter("include_objects_markers_in_collision").value):
            self._merge_marker_array_into_world(
                markers_msg=self._objects_markers_msg,
                spheres=spheres,
                cubes=cubes,
                base_frame_name=base_frame_name,
                name_prefix="objects",
                ignored_ns=ignored_ns,
            )
        if include_live_collision_inputs:
            self._merge_marker_array_into_world(
                markers_msg=self._collision_markers_msg,
                spheres=spheres,
                cubes=cubes,
                base_frame_name=base_frame_name,
                name_prefix="collision",
                ignored_ns=ignored_ns,
                allowed_marker_types={int(Marker.CUBE), int(Marker.CYLINDER)},
            )
            self._merge_pose_array_into_world(
                poses_msg=self._collision_poses_msg,
                spheres=spheres,
                base_frame_name=base_frame_name,
            )

        coll_objs["sphere"] = spheres
        coll_objs["cube"] = cubes
        world_model["coll_objs"] = coll_objs
        world["world_model"] = world_model
        return world

    def _maybe_log_world_collision_objects(self, world_params: dict, *, source: str) -> None:
        if not bool(self.get_parameter("debug_collision_objects").value):
            return

        world_model = dict(world_params.get("world_model", {}))
        coll_objs = dict(world_model.get("coll_objs", {}))
        cubes = dict(coll_objs.get("cube", {}))
        spheres = dict(coll_objs.get("sphere", {}))

        max_items = int(self.get_parameter("debug_collision_objects_max_items").value)
        max_items = max(1, max_items)

        cube_names = sorted(cubes.keys())
        sphere_names = sorted(spheres.keys())

        preview_cube_names = cube_names[:max_items]
        preview_sphere_names = sphere_names[:max_items]

        signature = (
            tuple(cube_names),
            tuple(sphere_names),
            tuple(
                (
                    name,
                    tuple(round(float(v), 6) for v in cubes[name].get("dims", [])),
                    tuple(round(float(v), 6) for v in cubes[name].get("pose", [0.0, 0.0, 0.0])[:3]),
                )
                for name in preview_cube_names
            ),
            tuple(
                (
                    name,
                    round(float(spheres[name].get("radius", 0.0)), 6),
                    tuple(round(float(v), 6) for v in spheres[name].get("position", [0.0, 0.0, 0.0])[:3]),
                )
                for name in preview_sphere_names
            ),
        )
        if signature == self._last_collision_world_debug_signature:
            return
        self._last_collision_world_debug_signature = signature

        extra_cube = max(0, len(cube_names) - len(preview_cube_names))
        extra_sphere = max(0, len(sphere_names) - len(preview_sphere_names))
        cube_suffix = f" (+{extra_cube} more)" if extra_cube > 0 else ""
        sphere_suffix = f" (+{extra_sphere} more)" if extra_sphere > 0 else ""

        self.get_logger().info(
            f"[collision-debug:{source}] STORM world primitives: cubes={len(cube_names)} "
            f"{preview_cube_names}{cube_suffix}; spheres={len(sphere_names)} {preview_sphere_names}{sphere_suffix}"
        )

        for name in preview_cube_names:
            dims = cubes[name].get("dims", [])
            pose = cubes[name].get("pose", [])
            pos = pose[:3] if isinstance(pose, list) else []
            self.get_logger().info(
                f"[collision-debug:{source}] cube {name}: dims={dims}, pos={pos}"
            )
        for name in preview_sphere_names:
            radius = spheres[name].get("radius", 0.0)
            pos = spheres[name].get("position", [])
            self.get_logger().info(
                f"[collision-debug:{source}] sphere {name}: radius={radius}, pos={pos}"
            )

    def _maybe_log_dynamic_collision_spheres(self, dynamic_world_spheres: np.ndarray, *, source: str) -> None:
        if not bool(self.get_parameter("debug_collision_objects").value):
            return
        if dynamic_world_spheres is None:
            return
        if dynamic_world_spheres.ndim != 2 or dynamic_world_spheres.shape[1] != 4:
            return

        max_items = int(self.get_parameter("debug_collision_objects_max_items").value)
        max_items = max(1, max_items)

        active_mask = dynamic_world_spheres[:, 3] > 0.0
        active = dynamic_world_spheres[active_mask]
        count_active = int(active.shape[0])
        total = int(dynamic_world_spheres.shape[0])
        preview_count = min(count_active, max_items)
        preview = []
        for row in active[:preview_count]:
            preview.append(
                [
                    round(float(row[0]), 4),
                    round(float(row[1]), 4),
                    round(float(row[2]), 4),
                    round(float(row[3]), 4),
                ]
            )

        self.get_logger().info(
            f"[collision-debug:{source}] dynamic spheres active={count_active}/{total}, preview_xyzr={preview}",
            throttle_duration_sec=1.0,
        )

    def _collect_dynamic_collision_observations(self, base_frame_name: str) -> list[_DynamicCollisionObservation]:
        observations: list[_DynamicCollisionObservation] = []
        ignored_ns = {str(x) for x in self.get_parameter("ignore_collision_namespaces").value}

        markers_msg = self._collision_markers_msg
        if markers_msg is not None:
            for marker in markers_msg.markers:
                if marker.action in (Marker.DELETE, Marker.DELETEALL):
                    continue
                if marker.ns and str(marker.ns) in ignored_ns:
                    continue
                if int(marker.type) not in (int(Marker.CYLINDER), int(Marker.CUBE)):
                    continue

                marker_frame = str(marker.header.frame_id) if marker.header.frame_id else base_frame_name
                marker_M = self._pose_in_base_frame(marker.pose, marker_frame, base_frame_name)
                if marker_M is None:
                    continue

                # Dynamic mode uses sphere chains.
                # For CUBE markers use a conservative circumscribed radius in XY
                # so we do not under-approximate cube corners.
                if int(marker.type) == int(Marker.CUBE):
                    radius = 0.5 * float(math.hypot(marker.scale.x, marker.scale.y))
                else:
                    radius = 0.5 * float(max(marker.scale.x, marker.scale.y))
                height = float(marker.scale.z)
                if radius <= 0.0 or height <= 0.0:
                    continue

                observations.append(
                    _DynamicCollisionObservation(
                        kind="cylinder",
                        center=marker_M.translation.astype(np.float64, copy=True),
                        rotation=marker_M.rotation.astype(np.float64, copy=True),
                        radius=radius,
                        height=height,
                    )
                )

        poses_msg = self._collision_poses_msg
        if poses_msg is not None:
            source_frame = str(poses_msg.header.frame_id) if poses_msg.header.frame_id else base_frame_name
            pose_radius = float(self.get_parameter("collision_pose_radius").value)
            if pose_radius <= 0.0:
                raise ValueError("collision_pose_radius must be > 0")
            for pose in poses_msg.poses:
                pose_M = self._pose_in_base_frame(pose, source_frame, base_frame_name)
                if pose_M is None:
                    continue
                observations.append(
                    _DynamicCollisionObservation(
                        kind="pose",
                        center=pose_M.translation.astype(np.float64, copy=True),
                        rotation=pose_M.rotation.astype(np.float64, copy=True),
                        radius=pose_radius,
                        height=0.0,
                    )
                )
        return observations

    def _initialize_dynamic_collision_slots(self, observations: list[_DynamicCollisionObservation]) -> bool:
        if self._dynamic_collision_slots_initialized:
            return True

        requested_slots = int(self.get_parameter("dynamic_collision_slots").value)
        if requested_slots < 0:
            raise ValueError("dynamic_collision_slots must be >= 0")
        inferred_slots = len(observations)
        slot_count = requested_slots if requested_slots > 0 else inferred_slots
        if slot_count <= 0:
            return False

        cylinder_sphere_count = int(self.get_parameter("dynamic_cylinder_spheres_per_object").value)
        if cylinder_sphere_count <= 0:
            raise ValueError("dynamic_cylinder_spheres_per_object must be >= 1")

        slot_specs: list[_DynamicCollisionSlotSpec] = []
        slot_offsets: list[tuple[int, int]] = []
        sphere_offset = 0
        for slot_idx in range(slot_count):
            if slot_idx < len(observations):
                obs = observations[slot_idx]
                if obs.kind == "cylinder":
                    slot_spec = _DynamicCollisionSlotSpec(
                        kind="cylinder",
                        radius=float(obs.radius),
                        height=float(obs.height),
                        sphere_count=cylinder_sphere_count,
                    )
                else:
                    slot_spec = _DynamicCollisionSlotSpec(
                        kind="pose",
                        radius=float(obs.radius),
                        height=0.0,
                        sphere_count=1,
                    )
            else:
                slot_spec = _DynamicCollisionSlotSpec(
                    kind="cylinder",
                    radius=0.0,
                    height=0.0,
                    sphere_count=cylinder_sphere_count,
                )

            slot_specs.append(slot_spec)
            slot_offsets.append((sphere_offset, sphere_offset + slot_spec.sphere_count))
            sphere_offset += slot_spec.sphere_count

        dynamic_world_spheres = np.zeros((sphere_offset, 4), dtype=np.float32)
        dynamic_world_spheres[:, :3] = self._dynamic_inactive_center
        dynamic_world_spheres[:, 3] = 0.0

        self._dynamic_collision_slot_specs = slot_specs
        self._dynamic_collision_slot_offsets = slot_offsets
        self._dynamic_world_spheres_np = dynamic_world_spheres
        self._dynamic_collision_slots_initialized = True
        self.get_logger().info(
            f"Initialized dynamic collision slots: slots={len(slot_specs)} total_spheres={sphere_offset}."
        )
        return True

    @staticmethod
    def _compute_cylinder_chain_centers(
        *,
        center: np.ndarray,
        rotation: np.ndarray,
        radius: float,
        height: float,
        sphere_count: int,
    ) -> np.ndarray:
        if sphere_count <= 1:
            return center.reshape(1, 3).astype(np.float32, copy=True)

        safe_height = max(float(height), 0.0)
        half_height = 0.5 * safe_height
        start = -half_height + float(radius)
        end = half_height - float(radius)
        if end < start:
            z_offsets = np.zeros((sphere_count,), dtype=np.float64)
        else:
            z_offsets = np.linspace(start, end, num=sphere_count, dtype=np.float64)

        local_points = np.zeros((sphere_count, 3), dtype=np.float64)
        local_points[:, 2] = z_offsets
        world_points = center.reshape(1, 3) + local_points @ rotation.T
        return world_points.astype(np.float32, copy=False)

    def _update_dynamic_world_spheres_from_observations(
        self,
        observations: list[_DynamicCollisionObservation],
    ) -> Optional[np.ndarray]:
        if not self._initialize_dynamic_collision_slots(observations):
            return None
        if self._dynamic_world_spheres_np is None:
            return None

        self._dynamic_world_spheres_np[:, :3] = self._dynamic_inactive_center
        self._dynamic_world_spheres_np[:, 3] = 0.0

        for slot_idx, slot_spec in enumerate(self._dynamic_collision_slot_specs):
            if slot_idx >= len(observations):
                continue
            obs = observations[slot_idx]
            start, end = self._dynamic_collision_slot_offsets[slot_idx]
            if slot_spec.kind == "cylinder":
                cylinder_radius = slot_spec.radius if slot_spec.radius > 0.0 else float(obs.radius)
                if cylinder_radius <= 0.0:
                    continue
                cylinder_height = slot_spec.height if slot_spec.height > 0.0 else float(obs.height)
                if slot_spec.radius <= 0.0:
                    slot_spec.radius = cylinder_radius
                if slot_spec.height <= 0.0 and cylinder_height > 0.0:
                    slot_spec.height = cylinder_height
                chain_centers = self._compute_cylinder_chain_centers(
                    center=obs.center,
                    rotation=obs.rotation,
                    radius=cylinder_radius,
                    height=cylinder_height,
                    sphere_count=slot_spec.sphere_count,
                )
                self._dynamic_world_spheres_np[start:end, :3] = chain_centers
                self._dynamic_world_spheres_np[start:end, 3] = float(cylinder_radius)
            else:
                pose_radius = slot_spec.radius if slot_spec.radius > 0.0 else obs.radius
                if pose_radius <= 0.0:
                    continue
                if slot_spec.radius <= 0.0:
                    slot_spec.radius = float(pose_radius)
                self._dynamic_world_spheres_np[start, :3] = obs.center.astype(np.float32, copy=False)
                self._dynamic_world_spheres_np[start, 3] = float(pose_radius)

        return self._dynamic_world_spheres_np

    @staticmethod
    def _q(value: float) -> float:
        return round(float(value), 6)

    def _marker_array_signature(self, markers_msg: Optional[MarkerArray]) -> tuple:
        if markers_msg is None:
            return ()
        sig = []
        for marker in markers_msg.markers:
            sig.append(
                (
                    int(marker.action),
                    int(marker.type),
                    str(marker.header.frame_id),
                    str(marker.ns),
                    int(marker.id),
                    self._q(marker.pose.position.x),
                    self._q(marker.pose.position.y),
                    self._q(marker.pose.position.z),
                    self._q(marker.pose.orientation.x),
                    self._q(marker.pose.orientation.y),
                    self._q(marker.pose.orientation.z),
                    self._q(marker.pose.orientation.w),
                    self._q(marker.scale.x),
                    self._q(marker.scale.y),
                    self._q(marker.scale.z),
                )
            )
        return tuple(sig)

    def _pose_array_signature(self, poses_msg: Optional[PoseArray]) -> tuple:
        if poses_msg is None:
            return ()
        sig = [str(poses_msg.header.frame_id)]
        for pose in poses_msg.poses:
            sig.append(
                (
                    self._q(pose.position.x),
                    self._q(pose.position.y),
                    self._q(pose.position.z),
                    self._q(pose.orientation.x),
                    self._q(pose.orientation.y),
                    self._q(pose.orientation.z),
                    self._q(pose.orientation.w),
                )
            )
        return tuple(sig)

    def _compute_world_inputs_signature(self) -> tuple:
        objects_signature = ()
        if bool(self.get_parameter("include_objects_markers_in_collision").value):
            objects_signature = self._marker_array_signature(self._objects_markers_msg)
        return (
            objects_signature,
            self._marker_array_signature(self._collision_markers_msg),
            self._pose_array_signature(self._collision_poses_msg),
            str(self.get_parameter("base_frame_name").value),
            self._q(self.get_parameter("collision_pose_radius").value),
            str(self.get_parameter("collision_pose_namespace").value),
        )

    def _maybe_update_dynamic_static_world_objects(self) -> bool:
        if self._storm_task is None or self._storm_base_world is None:
            return False
        if not bool(self.get_parameter("include_objects_markers_in_collision").value):
            return False

        objects_signature = self._marker_array_signature(self._objects_markers_msg)
        if objects_signature == self._last_dynamic_objects_signature:
            return False

        world_params = self._build_world_params(
            self._storm_base_world,
            include_live_collision_inputs=False,
        )
        self._storm_task.update_params(world_params=world_params)
        self._maybe_log_world_collision_objects(world_params, source="dynamic_static_update")
        self._last_dynamic_objects_signature = objects_signature
        return True

    def _maybe_update_dynamic_collision_inputs(self) -> None:
        if not self._storm_initialized or self._storm_task is None:
            return
        if not self._world_inputs_dirty:
            return
        if not bool(self.get_parameter("refresh_collision_world_on_updates").value):
            self._world_inputs_dirty = False
            return

        update_rate_hz = float(self.get_parameter("collision_update_rate_hz").value)
        if update_rate_hz < 0.0:
            raise ValueError("collision_update_rate_hz must be >= 0")

        now = time.perf_counter()
        if update_rate_hz > 0.0:
            min_period = 1.0 / update_rate_hz
            if (now - self._last_world_inputs_apply_time) < min_period:
                return

        updated_world = self._maybe_update_dynamic_static_world_objects()

        base_frame_name = str(self.get_parameter("base_frame_name").value)
        observations = self._collect_dynamic_collision_observations(base_frame_name)
        dynamic_world_spheres = self._update_dynamic_world_spheres_from_observations(observations)
        updated_dynamic = False
        if dynamic_world_spheres is not None:
            self._storm_task.update_params(dynamic_world_spheres=dynamic_world_spheres)
            updated_dynamic = True
            self._maybe_log_dynamic_collision_spheres(dynamic_world_spheres, source="dynamic_update")

        self._last_world_inputs_apply_time = now
        self._world_inputs_dirty = False
        if updated_world:
            self.get_logger().info(
                "Updated static world collision primitives from objects markers.",
                throttle_duration_sec=2.0,
            )
        if updated_dynamic:
            self.get_logger().info(
                "Updated dynamic collision spheres from live marker/pose inputs.",
                throttle_duration_sec=2.0,
            )

    def _maybe_update_world_collision_inputs(self) -> None:
        if bool(self.get_parameter("dynamic_collision_mode").value):
            self._maybe_update_dynamic_collision_inputs()
            return
        if not self._storm_initialized or self._storm_task is None or self._storm_base_world is None:
            return
        if not self._world_inputs_dirty:
            return
        if not bool(self.get_parameter("refresh_collision_world_on_updates").value):
            self._world_inputs_dirty = False
            return

        update_rate_hz = float(self.get_parameter("collision_update_rate_hz").value)
        if update_rate_hz < 0.0:
            raise ValueError("collision_update_rate_hz must be >= 0")

        now = time.perf_counter()
        if update_rate_hz > 0.0:
            min_period = 1.0 / update_rate_hz
            if (now - self._last_world_inputs_apply_time) < min_period:
                return

        signature = self._compute_world_inputs_signature()
        if signature == self._last_world_inputs_signature:
            self._world_inputs_dirty = False
            return

        world_params = self._build_world_params(self._storm_base_world, include_live_collision_inputs=True)
        self._storm_task.update_params(world_params=world_params)
        self._maybe_log_world_collision_objects(world_params, source="world_update")
        self._last_world_inputs_signature = signature
        self._last_world_inputs_apply_time = now
        self._world_inputs_dirty = False
        self.get_logger().info("Updated STORM collision world from live marker/pose inputs.", throttle_duration_sec=2.0)

    def _smooth_reference(
        self, *, q_des: np.ndarray, dq_des: np.ndarray, ddq_des: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Tier 3c (v2): step the Ruckig OTG once toward the current target.

        Called from `_ref_pub_tick()` at `ref_smoothing_rate_hz`, decoupled from STORM's
        replanning rate (`rate_hz`) -- see the parameter docstring for why this decoupling is
        the actual fix. The OTG's own control_cycle must match the rate it is *stepped* at
        (ref_smoothing_rate_hz), not the rate the target changes at (rate_hz).
        """
        from ruckig import InputParameter, OutputParameter, Result, Ruckig

        n = int(q_des.size)
        ref_rate_hz = float(self.get_parameter("ref_smoothing_rate_hz").value)
        if ref_rate_hz <= 0.0:
            raise ValueError("ref_smoothing_rate_hz must be > 0")

        if self._otg is None or self._otg.degrees_of_freedom != n:
            self._otg = Ruckig(n, 1.0 / ref_rate_hz)
            self._otg_input = InputParameter(n)
            self._otg_output = OutputParameter(n)
            self._otg_initialized = False

        max_vel = self._broadcast_param(
            np.array(self.get_parameter("ref_smoothing_max_vel").value, dtype=np.float64), n, "ref_smoothing_max_vel"
        )
        max_acc = self._broadcast_param(
            np.array(self.get_parameter("ref_smoothing_max_acc").value, dtype=np.float64), n, "ref_smoothing_max_acc"
        )
        max_jerk = self._broadcast_param(
            np.array(self.get_parameter("ref_smoothing_max_jerk").value, dtype=np.float64), n, "ref_smoothing_max_jerk"
        )
        if np.any(max_vel <= 0.0) or np.any(max_acc <= 0.0) or np.any(max_jerk <= 0.0):
            raise ValueError("ref_smoothing_max_vel/acc/jerk must all be > 0.")
        self._otg_input.max_velocity = max_vel.tolist()
        self._otg_input.max_acceleration = max_acc.tolist()
        self._otg_input.max_jerk = max_jerk.tolist()

        if not self._otg_initialized:
            seed_q = self._np_q if self._np_q is not None and self._np_q.shape == q_des.shape else q_des
            seed_v = self._np_dq if self._np_dq is not None and self._np_dq.shape == q_des.shape else np.zeros_like(q_des)
            self._otg_input.current_position = seed_q.tolist()
            self._otg_input.current_velocity = np.clip(seed_v, -max_vel, max_vel).tolist()
            self._otg_input.current_acceleration = [0.0] * n
            self._otg_initialized = True

        # Clamp the requested target into the same envelope Ruckig will enforce; an
        # out-of-envelope target velocity/acceleration is a Ruckig input error, not just
        # something it smooths away.
        self._otg_input.target_position = q_des.tolist()
        self._otg_input.target_velocity = np.clip(dq_des, -max_vel, max_vel).tolist()
        self._otg_input.target_acceleration = np.clip(ddq_des, -max_acc, max_acc).tolist()

        result = self._otg.update(self._otg_input, self._otg_output)
        if result not in (Result.Working, Result.Finished):
            self.get_logger().warn(
                f"Ruckig reference smoothing returned {result}; passing STORM output through unfiltered this tick.",
                throttle_duration_sec=2.0,
            )
            self._otg_initialized = False
            return q_des, dq_des, ddq_des

        q_smooth = np.array(self._otg_output.new_position, dtype=np.float64)
        dq_smooth = np.array(self._otg_output.new_velocity, dtype=np.float64)
        ddq_smooth = np.array(self._otg_output.new_acceleration, dtype=np.float64)
        self._otg_output.pass_to_input(self._otg_input)
        return q_smooth, dq_smooth, ddq_smooth

    def _dispatch_reference(
        self,
        *,
        q_des: np.ndarray,
        dq_des: np.ndarray,
        ddq_des: np.ndarray,
        force_zero_gains: bool = False,
        force_zero_feedforward: bool = False,
    ) -> None:
        """Single entry point for "STORM/disarmed-hold produced a new target" used by both
        `_tick()` and `_publish_disarmed_control()`.

        When reference smoothing is off (default): publish immediately, exactly as before this
        feature existed. When on: just record the target: `_ref_pub_tick()` (running at
        ref_smoothing_rate_hz, decoupled from this method's caller's rate) does the actual
        smoothing and publishing.
        """
        if bool(self.get_parameter("enable_reference_smoothing").value):
            self._ref_target_q = q_des
            self._ref_target_dq = dq_des
            self._ref_target_ddq = ddq_des
            self._ref_target_force_zero_gains = force_zero_gains
            self._ref_target_force_zero_feedforward = force_zero_feedforward
        else:
            self._publish_lfc_control(
                q_des=q_des,
                dq_des=dq_des,
                ddq_des=ddq_des,
                force_zero_gains=force_zero_gains,
                force_zero_feedforward=force_zero_feedforward,
            )

    def _ref_pub_tick(self) -> None:
        """Runs at ref_smoothing_rate_hz, decoupled from STORM's replan rate (rate_hz). Steps
        the Ruckig OTG once toward whatever target `_dispatch_reference` last recorded, and
        publishes the result. No-op (and cheap) whenever reference smoothing is disabled or no
        target has been recorded yet.
        """
        if not bool(self.get_parameter("enable_reference_smoothing").value):
            return
        if self._ref_target_q is None or self._ref_target_dq is None or self._ref_target_ddq is None:
            return
        q_smooth, dq_smooth, ddq_smooth = self._smooth_reference(
            q_des=self._ref_target_q, dq_des=self._ref_target_dq, ddq_des=self._ref_target_ddq
        )
        self._publish_lfc_control(
            q_des=q_smooth,
            dq_des=dq_smooth,
            ddq_des=ddq_smooth,
            force_zero_gains=self._ref_target_force_zero_gains,
            force_zero_feedforward=self._ref_target_force_zero_feedforward,
        )

    def _publish_rollout_markers(self) -> None:
        """Publishes STORM's current top-10 candidate rollouts and the single best/selected one
        as rviz MarkerArrays, on the exact topics pick_place.rviz/vids.rviz already have
        "MPC Predictions"/"MPC Reference" display slots for (see enable_rollout_visualization's
        declaration comment for why those were sitting unpublished). Pure visualization: reads
        data STORM already computes every optimize() call (top_trajs/top_values), no extra MPPI
        work, decoupled from the 50Hz control loop since humans don't need it that fast.

        top_trajs holds end-effector *positions* (ee_pos_seq, x/y/z only, no orientation) for
        STORM's internal ee_link, in STORM's base frame -- published here directly in
        base_frame_name with no further transform. This is approximate by a few cm (the fixed
        ee_link-to-tcp offset), which position-only data can't correct for exactly since that
        offset needs to be rotated by the (unavailable, here) end-effector orientation. Fine for
        a visual debugging aid; not used for control.

        Wrapped defensively: this crosses the same torch.multiprocessing/CUDA-IPC boundary as
        the actual control command (see ControlProcess in mpc_process_wrapper.py), which my
        CPU-only standalone tests for this feature could not exercise (those use the
        same-process WAIT=True debug path) -- a visualization bug here must never be allowed to
        destabilize the control-critical path around it.
        """
        if not bool(self.get_parameter("enable_rollout_visualization").value):
            return
        if self._storm_task is None:
            return
        try:
            top_trajs = getattr(self._storm_task, "top_trajs", None)
            top_values = getattr(self._storm_task, "top_values", None)
            if top_trajs is None or top_values is None:
                return

            trajs_np = np.asarray(top_trajs.detach().cpu().numpy() if hasattr(top_trajs, "detach") else top_trajs)
            values_np = np.asarray(top_values.detach().cpu().numpy() if hasattr(top_values, "detach") else top_values)
            if trajs_np.ndim != 3 or trajs_np.shape[-1] != 3 or trajs_np.shape[0] == 0:
                return

            base_frame_name = str(self.get_parameter("base_frame_name").value)
            now = self.get_clock().now().to_msg()
            rate_hz = float(self.get_parameter("rollout_viz_rate_hz").value)
            lifetime_sec = 3.0 / rate_hz if rate_hz > 0.0 else 1.0
            lifetime_whole = int(lifetime_sec)
            lifetime_nanosec = int((lifetime_sec - lifetime_whole) * 1.0e9)

            n_trajs = int(trajs_np.shape[0])
            v_min = float(values_np.min())
            v_span = max(float(values_np.max()) - v_min, 1.0e-9)

            prediction_markers = []
            for i in range(n_trajs):
                marker = Marker()
                marker.header.stamp = now
                marker.header.frame_id = base_frame_name
                marker.ns = "mpc_predictions"
                marker.id = i
                marker.type = Marker.LINE_STRIP
                marker.action = Marker.ADD
                marker.pose.orientation.w = 1.0
                marker.scale.x = 0.004
                marker.points = [Point(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in trajs_np[i]]
                # Rank 0 (lowest cost, best) -> green; worst of the top-10 -> red.
                frac = (float(values_np[i]) - v_min) / v_span
                marker.color.r = float(frac)
                marker.color.g = float(1.0 - frac)
                marker.color.b = 0.0
                marker.color.a = 0.5
                marker.lifetime.sec = lifetime_whole
                marker.lifetime.nanosec = lifetime_nanosec
                prediction_markers.append(marker)

            reference_marker = Marker()
            reference_marker.header.stamp = now
            reference_marker.header.frame_id = base_frame_name
            reference_marker.ns = "mpc_reference"
            reference_marker.id = 0
            reference_marker.type = Marker.LINE_STRIP
            reference_marker.action = Marker.ADD
            reference_marker.pose.orientation.w = 1.0
            reference_marker.scale.x = 0.01
            reference_marker.points = [Point(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in trajs_np[0]]
            reference_marker.color.r = 0.0
            reference_marker.color.g = 0.4
            reference_marker.color.b = 1.0
            reference_marker.color.a = 0.95
            reference_marker.lifetime.sec = lifetime_whole
            reference_marker.lifetime.nanosec = lifetime_nanosec

            self._pub_rollout_predictions.publish(MarkerArray(markers=prediction_markers))
            self._pub_rollout_reference.publish(MarkerArray(markers=[reference_marker]))
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(
                f"Rollout visualization failed this tick (non-fatal, control path unaffected): {exc}",
                throttle_duration_sec=5.0,
            )

    def _tick(self) -> None:
        self._armed = bool(self.get_parameter("armed").value)
        debug_mppi_rate = bool(self.get_parameter("debug_mppi_rate").value)
        self._maybe_autodiscover_collision_markers_topic()

        if not self._storm_initialized or self._storm_task is None or self._storm_control_dt is None:
            return
        self._maybe_update_world_collision_inputs()
        if self._np_q is None:
            return
        if self._goal_pose_msg is None:
            self._publish_disarmed_control()
            return

        base_frame_name = str(self.get_parameter("base_frame_name").value)

        goal_msg = self._goal_pose_msg
        goal_M = pose_msg_to_se3(goal_msg.pose)
        if goal_msg.header.frame_id and goal_msg.header.frame_id != base_frame_name:
            try:
                tf = self._tf_buffer.lookup_transform(
                    base_frame_name,
                    goal_msg.header.frame_id,
                    Time(),
                )
                base_T_src = transform_msg_to_se3(tf.transform)
                goal_M = base_T_src * goal_M
            except TransformException as exc:
                self.get_logger().warn(f"TF failed transforming goal to '{base_frame_name}': {exc}")
                return

        if self._ros_T_hand_to_tcp is None or self._storm_T_hand_to_ee is None:
            self.get_logger().error("Internal goal frame transforms are not initialized.")
            return

        goal_storm_ee_M = goal_M * self._ros_T_hand_to_tcp.inverse() * self._storm_T_hand_to_ee
        goal_ee_pos = goal_storm_ee_M.translation.copy()
        goal_ee_rot = goal_storm_ee_M.rotation.copy()

        goal_xyzquat = pin.SE3ToXYZQUAT(goal_storm_ee_M)
        if (
            self._last_goal_storm_ee_xyzquat is None
            or float(np.linalg.norm(goal_xyzquat - self._last_goal_storm_ee_xyzquat)) > 1.0e-6
        ):
            self._storm_task.update_params(goal_ee_pos=goal_ee_pos, goal_ee_rot=goal_ee_rot)
            self._last_goal_storm_ee_xyzquat = goal_xyzquat

        if not self._armed:
            self._publish_disarmed_control()
            return

        # Receding-horizon outer loop.
        if self._t0_perf is None:
            self._t0_perf = time.perf_counter()
        t_step = float(time.perf_counter() - self._t0_perf)
        tick_now = time.perf_counter()
        tick_dt = None
        if self._last_tick_perf is not None:
            tick_dt = float(tick_now - self._last_tick_perf)
        self._last_tick_perf = tick_now
        curr_state = {
            "position": self._np_q.astype(np.float32, copy=True),
            "velocity": self._np_dq.astype(np.float32, copy=True),
            "acceleration": np.zeros_like(self._np_q, dtype=np.float32),
        }
        cmd_t0 = time.perf_counter()
        cmd_des = self._storm_task.get_command(
            t_step, curr_state, control_dt=float(self._storm_control_dt), WAIT=False
        )
        cmd_wall_dt = float(time.perf_counter() - cmd_t0)
        if debug_mppi_rate:
            alpha = 0.90
            if tick_dt is not None and tick_dt > 0.0:
                self._tick_dt_ema = tick_dt if self._tick_dt_ema is None else alpha * self._tick_dt_ema + (1.0 - alpha) * tick_dt
            self._cmd_wall_dt_ema = cmd_wall_dt if self._cmd_wall_dt_ema is None else alpha * self._cmd_wall_dt_ema + (1.0 - alpha) * cmd_wall_dt
            opt_dt = float(getattr(self._storm_task, "opt_dt", 0.0))
            if opt_dt > 0.0:
                self._mppi_opt_dt_ema = opt_dt if self._mppi_opt_dt_ema is None else alpha * self._mppi_opt_dt_ema + (1.0 - alpha) * opt_dt
            mpc_dt = float(getattr(self._storm_task, "mpc_dt", 0.0))
            # Tier 2c: N_eff out of num_particles. Low N_eff (e.g. << 50/1000) means the
            # executed "mean" action is effectively an average of very few particles each
            # tick -- the direct, measurable signature of the sampling-noise mechanism behind
            # the feedforward jitter this whole investigation is about.
            n_eff = float(getattr(self._storm_task, "effective_sample_size", float("nan")))

            loop_hz = (1.0 / self._tick_dt_ema) if self._tick_dt_ema and self._tick_dt_ema > 0.0 else float("nan")
            cmd_hz = (1.0 / self._cmd_wall_dt_ema) if self._cmd_wall_dt_ema and self._cmd_wall_dt_ema > 0.0 else float("nan")
            mppi_hz = (1.0 / self._mppi_opt_dt_ema) if self._mppi_opt_dt_ema and self._mppi_opt_dt_ema > 0.0 else float("nan")
            self.get_logger().info(
                "MPPI timing: "
                f"loop~{loop_hz:.1f}Hz, get_command~{cmd_hz:.1f}Hz, "
                f"opt~{mppi_hz:.1f}Hz (opt_dt~{1.0e3 * (self._mppi_opt_dt_ema or 0.0):.1f}ms), "
                f"control_dt={self._storm_control_dt:.4f}s, mpc_dt={mpc_dt:.4f}s, "
                f"N_eff~{n_eff:.1f}",
                throttle_duration_sec=1.0,
            )

        q_des = np.asarray(cmd_des.get("position", self._np_q), dtype=np.float64)
        dq_des = np.asarray(cmd_des.get("velocity", np.zeros_like(q_des)), dtype=np.float64)
        ddq_des = np.asarray(cmd_des.get("acceleration", np.zeros_like(q_des)), dtype=np.float64)
        self._dispatch_reference(q_des=q_des, dq_des=dq_des, ddq_des=ddq_des)

    @staticmethod
    def _broadcast_param(arr: np.ndarray, size: int, name: str) -> np.ndarray:
        if arr.size == 1:
            return np.full((size,), float(arr[0]), dtype=np.float64)
        if arr.size != size:
            raise ValueError(f"{name} must have length 1 or {size}, got {arr.size}")
        return arr.astype(np.float64, copy=False)

    def _publish_disarmed_control(self) -> None:
        behavior = str(self.get_parameter("disarmed_behavior").value)
        if behavior not in ("hold", "zero_torque"):
            raise ValueError(f"disarmed_behavior must be 'hold' or 'zero_torque', got '{behavior}'")
        if self._np_q is None:
            return

        q_des = self._np_q.astype(np.float64, copy=True)
        dq_des = np.zeros_like(q_des, dtype=np.float64)
        ddq_des = np.zeros_like(q_des, dtype=np.float64)

        if behavior == "zero_torque":
            self._dispatch_reference(
                q_des=q_des,
                dq_des=dq_des,
                ddq_des=ddq_des,
                force_zero_gains=True,
                force_zero_feedforward=True,
            )
        else:
            self._dispatch_reference(q_des=q_des, dq_des=dq_des, ddq_des=ddq_des)

    def _publish_lfc_control(
        self,
        *,
        q_des: np.ndarray,
        dq_des: np.ndarray,
        ddq_des: np.ndarray,
        force_zero_gains: bool = False,
        force_zero_feedforward: bool = False,
    ) -> None:
        if self._moving_joint_names and q_des.size != len(self._moving_joint_names):
            raise ValueError(f"q_des size {q_des.size} != {len(self._moving_joint_names)} moving joints.")
        if dq_des.size != q_des.size or ddq_des.size != q_des.size:
            raise ValueError("dq_des and ddq_des must match q_des size.")

        q_des = self._clamp_q_des_to_measured(q_des)

        n = int(q_des.size)
        K = np.zeros((n, 2 * n), dtype=np.float64)
        if not force_zero_gains:
            feedback_mode = str(self.get_parameter("feedback_mode").value)
            if feedback_mode not in ("joint", "cartesian"):
                raise ValueError(f"feedback_mode must be 'joint' or 'cartesian', got '{feedback_mode}'")
            if feedback_mode == "cartesian":
                K = self._compute_cartesian_feedback_gain(q_des)
            else:
                kp = self._broadcast_param(np.array(self.get_parameter("kp").value, dtype=np.float64), n, "kp")
                kd = self._broadcast_param(np.array(self.get_parameter("kd").value, dtype=np.float64), n, "kd")
                K[:, :n] = np.diag(kp)
                K[:, n:] = np.diag(kd)

        tau_ff = np.zeros((n,), dtype=np.float64)
        if not force_zero_feedforward:
            tau_ff = self._compute_feedforward(q_des=q_des, dq_des=dq_des, ddq_des=ddq_des)

        now = self.get_clock().now().to_msg()
        init = Sensor()
        init.header.stamp = now
        init.joint_state.name = list(self._moving_joint_names)
        init.joint_state.position = [float(v) for v in q_des.tolist()]
        init.joint_state.velocity = [float(v) for v in dq_des.tolist()]
        init.joint_state.effort = [float(v) for v in tau_ff.tolist()]

        msg = Control()
        msg.header.stamp = now
        msg.feedback_gain = matrix_numpy_to_msg(K)
        msg.feedforward = matrix_numpy_to_msg(tau_ff)
        msg.initial_state = init

        self._pub_control.publish(msg)

    def _filter_ddq_for_feedforward(self, ddq_des: np.ndarray) -> np.ndarray:
        """Causal EMA low-pass on the MPPI acceleration command, used only for the
        inverse-dynamics feedforward torque (see `feedforward_accel_filter_coeff`).

        STORM's MPPI re-solves every tick from a finite particle batch and adopts the new
        weighted-mean action with little cross-tick memory (`step_size_mean` close to 1 in
        franka_reacher.yml), so the raw `qdd_des` it returns carries sampling variance. The
        published q_des/dq_des hide this because they are produced by integrating qdd_des
        (a smoothing operation, applied twice for q_des); RNEA's mass-matrix multiplication
        does the opposite and turns raw acceleration noise directly into torque noise. STORM
        ships a `command_filter` intended for exactly this purpose
        (storm_kit/mpc/task/task_base.py), but it is never actually invoked in
        `BaseTask.get_command()` -- so we filter at the point where acceleration is consumed
        for torque instead of relying on it.
        """
        alpha = float(self.get_parameter("feedforward_accel_filter_coeff").value)
        if not (0.0 < alpha <= 1.0):
            raise ValueError(f"feedforward_accel_filter_coeff must be in (0, 1], got {alpha}")
        if alpha >= 1.0 or self._ddq_ff_filt is None or self._ddq_ff_filt.shape != ddq_des.shape:
            self._ddq_ff_filt = ddq_des.astype(np.float64, copy=True)
        else:
            self._ddq_ff_filt = alpha * ddq_des + (1.0 - alpha) * self._ddq_ff_filt
        return self._ddq_ff_filt

    def _build_full_q(self, q_des: np.ndarray) -> np.ndarray:
        q = pin.neutral(self._pin_model)
        for i in range(len(self._moving_joint_names)):
            q[int(self._pin_q_idx[i])] = float(q_des[i])
        return q

    def _build_full_v(self, values: np.ndarray) -> np.ndarray:
        v = np.zeros((int(self._pin_model.nv),), dtype=np.float64)
        for i in range(len(self._moving_joint_names)):
            v[int(self._pin_v_idx[i])] = float(values[i])
        return v

    def _compute_feedforward(self, *, q_des: np.ndarray, dq_des: np.ndarray, ddq_des: np.ndarray) -> np.ndarray:
        mode = str(self.get_parameter("feedforward_mode").value)
        if mode not in ("none", "gravity", "coriolis_gravity", "inverse_dynamics"):
            raise ValueError(
                f"feedforward_mode must be one of: none, gravity, coriolis_gravity, inverse_dynamics (got '{mode}')"
            )
        if mode == "none":
            return np.zeros_like(q_des, dtype=np.float64)

        if self._pin_model is None or self._pin_data is None:
            raise RuntimeError("Pinocchio model not initialized; cannot compute feedforward.")

        if mode == "inverse_dynamics":
            ddq_des = self._filter_ddq_for_feedforward(ddq_des)

        q = self._build_full_q(q_des)
        v = self._build_full_v(dq_des)
        a = self._build_full_v(ddq_des)

        if mode == "gravity":
            v[:] = 0.0
            a[:] = 0.0
        elif mode == "coriolis_gravity":
            # Tier 2a: C(q,v)v + g(q), real (smooth, integrated) dq_des but zero acceleration.
            # This is the limiting case of feedforward_accel_filter_coeff -> 0 reached by
            # deleting the inertial term entirely instead of attenuating it: zero sensitivity
            # to qdd_des sampling noise, at the cost of all inertial feedforward (M(q)*a).
            a[:] = 0.0

        tau = pin.rnea(self._pin_model, self._pin_data, q, v, a)
        tau_out = np.zeros((len(self._moving_joint_names),), dtype=np.float64)
        for i in range(len(self._moving_joint_names)):
            tau_out[i] = float(tau[int(self._pin_v_idx[i])])
        return tau_out

    def _clamp_q_des_to_measured(self, q_des: np.ndarray) -> np.ndarray:
        """Tier 2b: joint-space reference-error clamp (SERL-style reference-ball trick).

        Pulls a published q_des back towards the last known measured q if it would exceed
        max_joint_ref_error_rad, so a stale/jumped reference cannot produce an oversized
        K*(q_des - q_meas) torque kick. Disabled (no-op) when the param is <= 0.
        """
        max_err = float(self.get_parameter("max_joint_ref_error_rad").value)
        if max_err <= 0.0 or self._np_q is None or self._np_q.shape != q_des.shape:
            return q_des
        err = q_des - self._np_q
        clamped_err = np.clip(err, -max_err, max_err)
        return self._np_q + clamped_err

    def _safe_joint_feedback_gain(self, n: int) -> np.ndarray:
        """Fallback K used by Cartesian mode's safety net: the same fixed diag(kp,kd) used by
        feedback_mode=joint, so a rejected Cartesian gain degrades to the known-good behavior
        instead of either crashing or silently sending something dangerous."""
        kp = self._broadcast_param(np.array(self.get_parameter("kp").value, dtype=np.float64), n, "kp")
        kd = self._broadcast_param(np.array(self.get_parameter("kd").value, dtype=np.float64), n, "kd")
        K = np.zeros((n, 2 * n), dtype=np.float64)
        K[:, :n] = np.diag(kp)
        K[:, n:] = np.diag(kd)
        return K

    def _compute_cartesian_feedback_gain(self, q_des: np.ndarray) -> np.ndarray:
        """Tier 3b (v3): Cartesian-impedance-equivalent feedback gain, built in the bridge.

        The LFC's control law is tau = tau_ff + K @ [q_des(-)q_meas ; dq_des-dq_meas], with K a
        generic (joint_nv x 2*nv) matrix multiplied against a joint-space error -- it never
        assumed K had to be block-diagonal. For small tracking error, the true Cartesian error
        obeys x_des - x_meas ~= J(q)(q_des - q_meas) (first-order forward-kinematics expansion),
        so publishing K_pos = J^T Kx J and K_vel = J^T Dx J approximates real Cartesian impedance
        without any change to linear_feedback_controller's C++ control law (verified to ~0.6%
        linearization error for representative joint errors in our tracking-error regime).

        v1 added a flat null_kp*I on top of J^T Kx J to cover the 1-DoF null space (6D task, 7
        DoF arm) -- this caused a phase-timeout and an Ignition Gazebo crash. v2 replaced it
        with a properly null-space-*projected* term (N=I-J^+J via a damped pseudo-inverse) so it
        wouldn't perturb the task-space directions -- this fixed the crash (verified: the safety
        check below never tripped, no more crashes) but the arm *still* could not reliably reach
        the pregrasp target. The real cause, found by sampling J^T Kx J's eigenvalues across many
        configurations rather than just the one or two used to validate v1/v2: the *task-space*
        eigenvalues themselves collapse far below the null space at some configurations -- as
        low as ~0.01 against a working joint-space baseline of kp=100 uniform, a ~13,000x spread,
        because J's own singular values vary by more than an order of magnitude across this
        arm's workspace and J^T Kx J inherits that spread squared. v2's null-space-only term
        could never have fixed this: it was scoped to the wrong 1 dimension out of 7. v3 fixes
        the actual problem with a much simpler mechanism: a flat (unprojected) floor large
        enough to dominate the worst-case collapse everywhere, not just in the null space --
        calibrated empirically (see the session's diagnosis) so that even the worst sampled
        configuration's weakest task eigenvalue lands close to the working joint-space baseline
        (default floor 60 -> worst-case K_pos eigenvalue exactly 60, best-case ~199, a sane 3.3x
        spread instead of 13,000x; floor_kd=15 is very close to critically damped for that
        regime, 2*sqrt(60)~=15.5). This is no longer "pure" Cartesian impedance with isolated
        null-space compliance -- every direction now has at least a joint-space-like floor mixed
        in. The pseudo-inverse projector and its damping parameter are gone entirely: they are
        not needed once the floor dominates, and removing them deletes a matrix inversion (one
        less place to be numerically fragile near singularities) for free.

        RESULT, do not re-raise the floor expecting this to be "the fix": v3 does complete the
        task reliably (2/2 cycles, ~42-51s, no crash, no phase-timeout -- confirmed on 2
        independent runs) but its smoothness is *catastrophically* worse than doing nothing at
        all: ~9769-10034 vs the original unfiltered bug's 57.84 (a ~170x regression), consistent
        across both runs. Mechanism: unlike feedback_mode=joint where K=diag(kp,kd) never
        depends on q (zero jitter contribution regardless of anything else), K here is rebuilt
        from a live Jacobian every bridge tick and held constant by the LFC until the next
        message -- it is itself a q-dependent, quadratically-Jacobian-sensitive signal. v1/v2's
        weak floor accidentally multiplied real tracking-error jitter by a near-zero gain
        (calm-looking, but unable to generate enough torque to converge); v3's larger floor
        transmits that same jitter, now amplified by a much larger and itself-jittery K,
        directly into torque. The floor magnitude and the noise amplification are two sides of
        one tradeoff in this architecture -- raising or lowering this floor alone cannot escape
        it. A real fix would need to filter K itself across ticks, not attempted: not worth it
        given storm_mppi_planner_params.yaml's shipped default (internalfilter+tier1) already
        satisfies the actual goal this bridge exists for. feedback_mode defaults to "joint" and
        should stay there; this mode is kept available, documented, and working-but-not-useful
        rather than deleted, in case directional Cartesian compliance is wanted for an unrelated
        reason (e.g. human-safety compliance) where this tradeoff might be acceptable.
        """
        n = len(self._moving_joint_names)
        ee_frame_name = str(self.get_parameter("ee_frame_name").value)
        frame_id = int(self._pin_model.getFrameId(ee_frame_name))
        if frame_id <= 0 or frame_id >= len(self._pin_model.frames):
            raise RuntimeError(f"ee_frame_name '{ee_frame_name}' not found in bridge's Pinocchio model.")

        q_full = self._build_full_q(q_des)
        pin.computeJointJacobians(self._pin_model, self._pin_data, q_full)
        pin.framesForwardKinematics(self._pin_model, self._pin_data, q_full)
        J_full = pin.getFrameJacobian(self._pin_model, self._pin_data, frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        J = np.zeros((6, n), dtype=np.float64)
        for i in range(n):
            J[:, i] = J_full[:, int(self._pin_v_idx[i])]

        kx = self._broadcast_param(np.array(self.get_parameter("cartesian_kp").value, dtype=np.float64), 6, "cartesian_kp")
        dx = self._broadcast_param(np.array(self.get_parameter("cartesian_kd").value, dtype=np.float64), 6, "cartesian_kd")
        floor_kp = float(self.get_parameter("cartesian_null_kp").value)
        floor_kd = float(self.get_parameter("cartesian_null_kd").value)

        K_pos = J.T @ np.diag(kx) @ J + floor_kp * np.eye(n)
        K_vel = J.T @ np.diag(dx) @ J + floor_kd * np.eye(n)

        K = np.zeros((n, 2 * n), dtype=np.float64)
        K[:, :n] = K_pos
        K[:, n:] = K_vel

        # Safety net: a finite-and-bounded check, not a crash. max_safe_eig is a generous bound
        # (well above the largest eigenvalue (~240) seen sampling many configurations with the
        # default cartesian_kp/kd + floor) so this only trips on a genuine problem (e.g. a
        # future bad parameter choice well outside what was calibrated here).
        max_safe_eig = 1000.0
        if not np.all(np.isfinite(K)) or float(np.max(np.abs(np.linalg.eigvalsh(K_pos)))) > max_safe_eig:
            self.get_logger().error(
                "Cartesian feedback gain failed its finite/magnitude safety check "
                "(non-finite or |eigenvalue| > {:.0f}); falling back to joint-space diag(kp,kd) for this tick.".format(
                    max_safe_eig
                ),
                throttle_duration_sec=1.0,
            )
            return self._safe_joint_feedback_gain(n)
        return K

    def destroy_node(self) -> bool:
        try:
            if self._storm_task is not None:
                self._storm_task.close()
        except Exception:
            pass
        return super().destroy_node()


def main(args=None) -> None:
    import torch

    torch.multiprocessing.set_start_method("spawn", force=True)

    rclpy.init(args=args)
    node = StormMppiPlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except rclpy.executors.ExternalShutdownException:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
