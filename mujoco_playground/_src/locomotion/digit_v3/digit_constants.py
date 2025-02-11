# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Digit constants."""

from mujoco_playground._src import mjx_env

ROOT_PATH = mjx_env.ROOT_PATH / "locomotion" / "digit_v3" / "xmls"
FEET_ONLY_XML = ROOT_PATH / "scene_mjx_feetonly.xml"

FEET_SITES = [
    "left_foot",
    "right_foot",
]

FEET_BODIES = [
    "left-foot",
    "right-foot",
]

ARM_GEOMS = [
    "left-shoulder-pitch-c",
    "left-shoulder-yaw-c1",
    "left-elbow-c1",
    "right-shoulder-pitch-c1",
    "right-shoulder-yaw-c1",
    "right-elbow-c1",
]

LEFT_FEET_GEOMS = [
    "left-foot",
]
RIGHT_FEET_GEOMS = [
    "left-foot",
]

ROOT_BODY = "torso_link"

GRAVITY_SENSOR = "upvector"
GLOBAL_LINVEL_SENSOR = "global_linvel"
GLOBAL_ANGVEL_SENSOR = "global_angvel"
LOCAL_LINVEL_SENSOR = "local_linvel"
ACCELEROMETER_SENSOR = "accelerometer"
GYRO_SENSOR = "gyro"
