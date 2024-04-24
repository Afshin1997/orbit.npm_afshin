import gym
import torch

from omni.isaac.orbit.envs import RLTaskEnv
from omni.isaac.orbit.envs import RLTaskEnvCfg
import numpy as np
import os


class rlTaskIm(RLTaskEnv):
      def __init__(self, cfg: RLTaskEnvCfg, render_mode: str | None = None, **kwargs):
            """super take 2 parameter, the name of the classe and the name of the method which you are extending in the son class
            It can be implemented as super(rlTaskIm, self) to say that you are extending just this method"""
            #os.getcwd() take the path of the folder in which you alunch the command
            
            m1 = np.loadtxt(os.getcwd() + "/prisma_walker/pos_m1_18s.txt", dtype=float)
            m2 = np.loadtxt(os.getcwd() + "/prisma_walker/pos_m2_18s.txt", dtype=float)
            self.m1JointPos = torch.from_numpy(m1).float().to('cuda:0')
            self.m2JointPos = torch.from_numpy(m2).float().to('cuda:0')
            #self.device still doesn't exist
            """The super must go afterwards because otherwise the baseEnv calls the actionManagar
            which call the joint_actions.py when this attribute still doesn't exist"""
            self.index_imit = 0
 
            super().__init__(cfg, render_mode, **kwargs)
           
            self.m1_joint_pos_extended = self.m1JointPos.expand(self.num_envs, -1)
            self.m2_joint_pos_extended = self.m2JointPos.expand(self.num_envs, -1)     


      def step(self, action):

            self.index_imit += 1
            if(self.index_imit > 1818):
                  self.index_imit = 1
            super().step(action)
            
