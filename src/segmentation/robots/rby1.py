"""RB-Y1 dataset joint layout constants.

Dataset observation.state / action layout (44 values):
  [0:6]   torso_0 ~ torso_5
  [6:13]  right_arm_0 ~ right_arm_6
  [13:20] left_arm_0 ~ left_arm_6
  [20:32] right_gripper (12 values; first 2 are prismatic joints)
  [32:44] left_gripper  (12 values; first 2 are prismatic joints)

Indices 0..19 are the 20 joints used for FK.
Gripper prismatic joints (gripper_finger_r1/r2, l1/l2) are mapped
from the first values of each gripper block.
"""

DATASET_JOINT_NAMES: tuple[str, ...] = (
    "torso_0", "torso_1", "torso_2", "torso_3", "torso_4", "torso_5",
    "right_arm_0", "right_arm_1", "right_arm_2", "right_arm_3",
    "right_arm_4", "right_arm_5", "right_arm_6",
    "left_arm_0", "left_arm_1", "left_arm_2", "left_arm_3",
    "left_arm_4", "left_arm_5", "left_arm_6",
)

GRIPPER_MAPPING: dict[str, int] = {
    "gripper_finger_r1": 20,
    "gripper_finger_r2": 21,
    "gripper_finger_l1": 32,
    "gripper_finger_l2": 33,
}

EE_FRAMES: tuple[str, str] = ("ee_right", "ee_left")
