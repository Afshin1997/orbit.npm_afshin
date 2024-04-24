# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations.

The functions can be passed to the :class:`omni.isaac.orbit.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.orbit.assets import Articulation, RigidObject
from omni.isaac.orbit.managers import SceneEntityCfg
from omni.isaac.orbit.sensors import ContactSensor
from omni.isaac.orbit.utils.math import matrix_from_quat
import math
import numpy 



if TYPE_CHECKING:
    from omni.isaac.orbit.envs import RLTaskEnv

"""
MDP terminations.
"""


def time_out(env: RLTaskEnv) -> torch.Tensor:
    """Terminate the episode when the episode length exceeds the maximum episode length."""
    return env.episode_length_buf >= env.max_episode_length


def command_resample(env: RLTaskEnv, num_resamples: int = 1) -> torch.Tensor:
    """Terminate the episode based on the total number of times commands have been re-sampled.

    This makes the maximum episode length fluid in nature as it depends on how the commands are
    sampled. It is useful in situations where delayed rewards are used :cite:`rudin2022advanced`.
    """
    return torch.logical_and(
        (env.command_manager.time_left <= env.step_dt), (env.command_manager.command_counter == num_resamples)
    )


"""
Root terminations.
"""


def bad_orientation(
    env: RLTaskEnv, limit_angle: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when the asset's orientation is too far from the desired orientation limits.

    This is computed by checking the angle between the projected gravity vector and the z-axis.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.acos(-asset.data.projected_gravity_b[:, 2]).abs() > limit_angle


def base_height(
    env: RLTaskEnv, minimum_height: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when the asset's height is below the minimum height.

    Note:
        This is currently only supported for flat terrains, i.e. the minimum height is in the world frame.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_pos_w[:, 2] < minimum_height


"""
Joint terminations.
"""


def joint_pos_limit(env: RLTaskEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Terminate when the asset's joint positions are outside of the soft joint limits."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute any violations
    out_of_upper_limits = torch.any(asset.data.joint_pos > asset.data.soft_joint_pos_limits[..., 1], dim=1)
    out_of_lower_limits = torch.any(asset.data.joint_pos < asset.data.soft_joint_pos_limits[..., 0], dim=1)
    return torch.logical_or(out_of_upper_limits, out_of_lower_limits)


def joint_pos_manual_limit(
    env: RLTaskEnv, bounds: tuple[float, float], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when the asset's joint positions are outside of the configured bounds.

    Note:
        This function is similar to :func:`joint_pos_limit` but allows the user to specify the bounds manually.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    if asset_cfg.joint_ids is None:
        asset_cfg.joint_ids = slice(None)
    # compute any violations
    out_of_upper_limits = torch.any(asset.data.joint_pos[:, asset_cfg.joint_ids] > bounds[1], dim=1)
    out_of_lower_limits = torch.any(asset.data.joint_pos[:, asset_cfg.joint_ids] < bounds[0], dim=1)
    return torch.logical_or(out_of_upper_limits, out_of_lower_limits)


def joint_vel_limit(env: RLTaskEnv, max_velocity, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Terminate when the asset's joint velocities are outside of the soft joint limits."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # TODO read max velocities per joint from robot
    return torch.any(torch.abs(asset.data.joint_vel) > max_velocity, dim=1)


def joint_torque_limit(env: RLTaskEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Terminate when torque applied on the asset's joints are are outside of the soft joint limits."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.any(
        torch.isclose(asset.data.computed_torques, asset.data.applied_torque),
        dim=1,
    )


"""
Contact sensor.
"""
 

def illegal_contact(env: RLTaskEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Terminate when the contact force on the sensor exceeds the force threshold."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    net_contact_forces = contact_sensor.data.net_forces_w_history
    # check if any contact force exceeds the threshold

    return  torch.any(
        torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold, dim=1
    )


"""norm between the 3 axis, the max between the histories, look if the firs element is greater than the threshold"""
"""the output of torch.max is a column vector. Torch.any() of a column vector would return a boolean, (false, if all the elements are false, true if ust once is true)"""
"""But doing Torch.any(tensor, dim=1) means to check only along the rows if among all the elements there 
   is at least a True. If I do: torch.any(tensor) where tensor = [true,
                                                                 false,
                                                                 true]
    it would return only one True (because in the tensor there is at least one True). But if I perform:
    torch.any(tensor, dim = 1), it returns tensor = [true,
                                                    false,
                                                    true]
    because it is looking if for each row (iterating among the columns, because dim = 1) there is at least one true"""

"""Since the tensor is a col vector, doing dim=1 means that it return exectly that element for each row, 
   because it means to check if there is at least one true iterating along the columns, but in each row
   there is only one element and therefore it return the element itself"""

def yawCondition(env: RLTaskEnv, sensor_cfg: SceneEntityCfg):

    z_axis = torch.tensor([0, 0, 1]).to('cuda:0')
    z_axis_ex = z_axis.expand(env.num_envs, -1) # is (num_envs, 3)

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    bodyOrientation: RigidObject = env.scene["robot"]
    quat = bodyOrientation.data.root_state_w[:,3:7]
    #orientation of the root (N, 4)
    #quat = contact_sensor.data.quat_w
    """it has dimension (N,B,4)"""

    yaw = matrix_from_quat(quat)
    """it has dimension (N,B,3,3) if taken from env.scene.sensors, (N,3,3) otherwise"""
    yaw = yaw[:,:,2]
    "Take the z-orientation of the base_link"

    a = torch.sum(yaw*z_axis_ex, dim = 1) #* is a dot product between matrices
    #The product return a matrix (N, 3) where for each environment I have done the dot product z_curr_i*z_axis_i, z_curr_j*z_axis_j, z_curr_k*z_axis_k 
    #Then summing along the dimension 1 (that is the dimension of the cols, therefore summing along the rows because it is iterating along the cols)
    
    yaw_angle = torch.acos(torch.sum( yaw * z_axis_ex, dim = 1)) #the product with * is a dot product
    #make the product and then sum along the rows (dim=1 means columsn, sum among the element of each col)
    #column vector of angles in radiants

    yaw_angle = yaw_angle*180/math.pi
    #yaw angle is shape [num_envs, ] and contains the value of the angles:
    """a = torch.tensor([[1, 2], [1, 5]]) #(2,2)
    print(torch.unsqueeze(a, dim = 2)) #(2,2,1) """
    
    yaw_unsqueezed = torch.unsqueeze(yaw_angle, dim = 1) #turn from [num_envs] into [num_envs, 1]
    #Now I can perform torch.any along the cols, and obtain a vector of true or false if the single element of each row violate or not that condition
    #print(yaw_unsqueezed[0])
    return torch.any(yaw_unsqueezed > 25, dim = 1)
        