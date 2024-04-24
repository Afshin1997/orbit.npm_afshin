# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from omni.isaac.orbit.utils import configclass

from .velocity_env_cfg import LocomotionVelocityRoughEnvCfg

##
# Pre-defined configs
##
from .asset.lbr_iiwa import NonPrehensileManipulator_lbr_iiwa  # isort: skip
import carb
from dataclasses import MISSING
import omni.isaac.orbit_tasks.locomotion.velocity.mdp as mdp
import torch
import math 
from omni.isaac.orbit.managers import SceneEntityCfg


@configclass
class Lbr_iiwa(LocomotionVelocityRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # switch robot to anymal-d
        self.scene.robot = PRISMA_WALKER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None
        # remove conditions on contact sensors removal
        self.rewards.undesired_contacts = None
        # Modify the randomization of weights
        """self.events.add_base_mass.params.update({'asset_cfg': SceneEntityCfg("robot", body_names="base_link")})
        self.events.base_external_force_torque.params.update({'asset_cfg': SceneEntityCfg("robot", body_names="base_link")})
        """
        #self.terminations.base_contact.params["threshold"] = 25
        
        #self.terminations.base_contact.func = zitto
        #self.terminations.base_contact.params = {}
       


@configclass
class PrismaWalkerEnvCfg_PLAY(PrismaWalkerFlatEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.randomization.base_external_force_torque = None
        self.randomization.push_robot = None
