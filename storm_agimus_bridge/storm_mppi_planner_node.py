from __future__ import annotations

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
from geometry_msgs.msg import PoseStamped
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
        self._robot_description_msg: Optional[String] = None

        self._moving_joint_names: list[str] = []
        self._np_q: Optional[np.ndarray] = None
        self._np_dq: Optional[np.ndarray] = None

        self._storm_initialized = False
        self._storm_task = None
        self._storm_control_dt: Optional[float] = None
        self._storm_tensor_args = None

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
        objects_topic = str(self.get_parameter("objects_markers_topic").value)
        self._sub_objects = self.create_subscription(
            MarkerArray,
            objects_topic,
            self._objects_callback,
            qos_profile=QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE),
        )

        self._pub_control = self.create_publisher(
            Control,
            "control",
            qos_profile=QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT),
        )

        self._init_timer = self.create_timer(0.1, self._try_initialize)
        self._loop_timer = None

        self.get_logger().info(
            f"storm_mppi_planner ready. Waiting for robot_description, sensor, objects on {objects_topic}, and goals on {goal_topic}."
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
                description="MarkerArray topic describing obstacles/objects as primitives for STORM.",
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
                description='Feedforward torque term: "none", "gravity", or "inverse_dynamics".',
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

    def _objects_callback(self, msg: MarkerArray) -> None:
        self._objects_markers_msg = msg

    def _try_initialize(self) -> None:
        if self._storm_initialized:
            return

        self._armed = bool(self.get_parameter("armed").value)

        if self._robot_description_msg is None or self._np_q is None:
            return
        if self._objects_markers_msg is None:
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
        self._loop_timer = self.create_timer(1.0 / rate_hz, self._tick)
        self.get_logger().info(
            f"storm_mppi_planner initialized. Publishing linear_feedback_controller_msgs/Control at ~{rate_hz:.1f} Hz (armed={self._armed}). base={base_frame_name} ee={ee_frame_name}."
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
        world_params = self._merge_markers_into_world(base_world, self._objects_markers_msg)

        task = ReacherTask(
            task_file=cfg.task_file,
            robot_file=f"{cfg.robot}.yml",
            world_file=cfg.world_file,
            tensor_args=self._storm_tensor_args,
            world_params=world_params,
        )
        self._storm_task = task
        self._storm_control_dt = float(task.exp_params["control_dt"])

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

    def _merge_markers_into_world(self, base_world: dict, markers: MarkerArray) -> dict:
        world = dict(base_world)
        wm = dict(world.get("world_model", {}))
        coll_objs = dict(wm.get("coll_objs", {}))
        spheres = dict(coll_objs.get("sphere", {}))
        cubes = dict(coll_objs.get("cube", {}))

        ignored_ns = {str(x) for x in self.get_parameter("ignore_collision_namespaces").value}
        for m in markers.markers:
            if m.action == Marker.DELETE:
                continue
            if m.ns and str(m.ns) in ignored_ns:
                continue
            name = f"{m.ns}_{int(m.id)}" if m.ns else f"obj_{int(m.id)}"

            if m.type == Marker.SPHERE:
                radius = 0.5 * float(m.scale.x)
                spheres[name] = {
                    "radius": float(radius),
                    "position": [float(m.pose.position.x), float(m.pose.position.y), float(m.pose.position.z)],
                }
            elif m.type == Marker.CUBE:
                cubes[name] = {
                    "dims": [float(m.scale.x), float(m.scale.y), float(m.scale.z)],
                    "pose": [
                        float(m.pose.position.x),
                        float(m.pose.position.y),
                        float(m.pose.position.z),
                        float(m.pose.orientation.x),
                        float(m.pose.orientation.y),
                        float(m.pose.orientation.z),
                        float(m.pose.orientation.w),
                    ],
                }

        coll_objs["sphere"] = spheres
        coll_objs["cube"] = cubes
        wm["coll_objs"] = coll_objs
        world["world_model"] = wm
        return world

    def _tick(self) -> None:
        self._armed = bool(self.get_parameter("armed").value)

        if not self._storm_initialized or self._storm_task is None or self._storm_control_dt is None:
            return
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
        curr_state = {
            "position": self._np_q.astype(np.float32, copy=True),
            "velocity": self._np_dq.astype(np.float32, copy=True),
            "acceleration": np.zeros_like(self._np_q, dtype=np.float32),
        }
        cmd_des = self._storm_task.get_command(
            t_step, curr_state, control_dt=float(self._storm_control_dt), WAIT=False
        )

        q_des = np.asarray(cmd_des.get("position", self._np_q), dtype=np.float64)
        dq_des = np.asarray(cmd_des.get("velocity", np.zeros_like(q_des)), dtype=np.float64)
        ddq_des = np.asarray(cmd_des.get("acceleration", np.zeros_like(q_des)), dtype=np.float64)
        self._publish_lfc_control(q_des=q_des, dq_des=dq_des, ddq_des=ddq_des)

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
            self._publish_lfc_control(
                q_des=q_des,
                dq_des=dq_des,
                ddq_des=ddq_des,
                force_zero_gains=True,
                force_zero_feedforward=True,
            )
        else:
            self._publish_lfc_control(q_des=q_des, dq_des=dq_des, ddq_des=ddq_des)

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

        n = int(q_des.size)
        kp = self._broadcast_param(np.array(self.get_parameter("kp").value, dtype=np.float64), n, "kp")
        kd = self._broadcast_param(np.array(self.get_parameter("kd").value, dtype=np.float64), n, "kd")

        K = np.zeros((n, 2 * n), dtype=np.float64)
        if not force_zero_gains:
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

    def _compute_feedforward(self, *, q_des: np.ndarray, dq_des: np.ndarray, ddq_des: np.ndarray) -> np.ndarray:
        mode = str(self.get_parameter("feedforward_mode").value)
        if mode not in ("none", "gravity", "inverse_dynamics"):
            raise ValueError(f"feedforward_mode must be one of: none, gravity, inverse_dynamics (got '{mode}')")
        if mode == "none":
            return np.zeros_like(q_des, dtype=np.float64)

        if self._pin_model is None or self._pin_data is None:
            raise RuntimeError("Pinocchio model not initialized; cannot compute feedforward.")

        q = pin.neutral(self._pin_model)
        v = np.zeros((int(self._pin_model.nv),), dtype=np.float64)
        a = np.zeros((int(self._pin_model.nv),), dtype=np.float64)
        for i in range(len(self._moving_joint_names)):
            q[int(self._pin_q_idx[i])] = float(q_des[i])
            v[int(self._pin_v_idx[i])] = float(dq_des[i])
            a[int(self._pin_v_idx[i])] = float(ddq_des[i])

        if mode == "gravity":
            v[:] = 0.0
            a[:] = 0.0

        tau = pin.rnea(self._pin_model, self._pin_data, q, v, a)
        tau_out = np.zeros((len(self._moving_joint_names),), dtype=np.float64)
        for i in range(len(self._moving_joint_names)):
            tau_out[i] = float(tau[int(self._pin_v_idx[i])])
        return tau_out

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
