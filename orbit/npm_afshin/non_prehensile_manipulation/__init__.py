# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agent, env_cfg
from . import rlTaskImit #once the file is imported, he know which classe are within it

##
# Register Gym environments.
##

gym.register(
    id="prisma-walker-train",
    entry_point="prisma_walker.rlTaskImit:rlTaskIm",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": env_cfg.PrismaWalkerFlatEnvCfg,
        "rsl_rl_cfg_entry_point": agent.rsl_rl_cfg.PrismaWalkerFlatPPORunnerCfg,
    },
)

gym.register(
    id="Prisma-Velocity-Prisma-Walker-Flat-v0-PLAY",
    entry_point="prisma_walker:RLTaskEnv",  #when the main will call the gym.make() it will call the init() of RLTaskenvs
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": env_cfg.PrismaWalkerEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": agent.rsl_rl_cfg.PrismaWalkerFlatPPORunnerCfg,
    },
)

