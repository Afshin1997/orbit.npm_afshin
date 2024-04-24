

import omni.isaac.orbit.sim as sim_utils
from omni.isaac.orbit.actuators import ImplicitActuatorCfg, IdealPDActuatorCfg
from omni.isaac.orbit.assets.articulation import ArticulationCfg
 
##
# Configuration - Actuators.
##


NonPrehensileManipulator_lbr_iiwa = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"/workspace/orbit/source/IsaacSimPrismaLab/prisma_walker/asset/prisma_walker.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1.0,
            max_angular_velocity=1.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
        # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.02, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        joint_pos={
            "lbr_iiwa_joint_.*": 0.0,   
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "All_Actuators": ImplicitActuatorCfg(
            joint_names_expr=["lbr_iiwa_joint_.*"],
            effort_limit=87.0,
            velocity_limit=100.0,
            stiffness=800,
            damping=40,
        ),
    },
)
