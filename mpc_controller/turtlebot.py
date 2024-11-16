import pybullet as p
import numpy as np
import time
import pybullet_data

from mpc_controller.quadruped_ctrl import Turtle

from bots.AC_unicycle import Unicycle
from controllers.QP_controller_unicycle import QP_Controller_Unicycle

# Creating Bots

bot1_config_file_path = 'bots//bot_config//bot1.json'
bot2_config_file_path = 'bots//bot_config//bot2.json'

bot1 = Unicycle.from_JSON(bot1_config_file_path)
bot2 = Unicycle.from_JSON(bot2_config_file_path)

bot = bot1
bot_2 = bot2

CTRL0 = Turtle()
p.connect(p.GUI)
offset = [0,0,0]
p.setAdditionalSearchPath(pybullet_data.getDataPath())
turtle = p.loadURDF("urdf/turtlebot.urdf",offset)
plane = p.loadURDF("plane.urdf")

p.resetDebugVisualizerCamera(cameraDistance=1.5,
                             cameraYaw=-90,
                             cameraPitch=-30,
                             cameraTargetPosition=[0, 0.0, 0.5]
                             )

DURATION = 25
SIM_FREQ = 240

for j in range (p.getNumJoints(turtle)):
	print(p.getJointInfo(turtle,j))

forward=0
turn=0
p.setGravity(0,0,-9.8)

p.setJointMotorControl2(turtle,0,p.VELOCITY_CONTROL,force=0)
p.setJointMotorControl2(turtle,1,p.VELOCITY_CONTROL,force=0)

p.setJointMotorControl2(turtle,0,p.TORQUE_CONTROL,force=0)
p.setJointMotorControl2(turtle,1,p.TORQUE_CONTROL,force=0)

# Initialize the target trajectory   
TARGET_POSITION = np.array([[0.001*i, 0] for i in range(DURATION*SIM_FREQ)]) 
TARGET_VELOCITY = np.zeros([DURATION * SIM_FREQ, 2])
TARGET_ACCELERATION = np.zeros([DURATION * SIM_FREQ, 2])

# Derive the target trajectory to obtain target velocities and accelerations
TARGET_VELOCITY[1:, :] = (TARGET_POSITION[1:, :] - TARGET_POSITION[0:-1, :]) / SIM_FREQ
TARGET_ACCELERATION[1:, :] = (TARGET_VELOCITY[1:, :] - TARGET_VELOCITY[0:-1, :]) / SIM_FREQ


i = 0

pos_obs = [5.5, 0.1]
vel_obs = [0, 0]

# p.loadURDF("sphere2.urdf", 
#             pos_obs, 
#             p.getQuaternionFromEuler([0,0,0]), 
#             physicsClientId=PYB_CLIENT)

# pos_obs_1 = [1., 0.1, 0.5]

p.loadURDF("urdf/cylinder.urdf", 
            [pos_obs[0], pos_obs[1], 1.5],
            p.getQuaternionFromEuler([0,0,0]))

# pos_obs_2 = [2., -0.2, 0.5]

# p.loadURDF("urdfs/cylinder_small.urdf", 
#             pos_obs_2, 
#             p.getQuaternionFromEuler([0,0,0]), 
#             physicsClientId=PYB_CLIENT)

# pos_obs_3 = [3., 0.35, 0.5]

# p.loadURDF("urdfs/cylinder_small.urdf", 
#             pos_obs_3, 
#             p.getQuaternionFromEuler([0,0,0]), 
#             physicsClientId=PYB_CLIENT)


# p.loadURDF("urdfs/portal.urdf", 
#             pos_obs, 
#             p.getQuaternionFromEuler([0,0,0]), 
#             physicsClientId=PYB_CLIENT)

print('start')
while (1):#i<DURATION*SIM_FREQ-1):
        i += 1

        time.sleep(1./SIM_FREQ)

        pos, quat = p.getBasePositionAndOrientation(turtle)
        rpy = p.getEulerFromQuaternion(quat)
        vel, ang_v = p.getBaseVelocity(turtle)

        pos_x_y = np.array([pos[0], pos[1]])
        vel_x_y = np.array([vel[0], vel[1]])
        th = rpy[2]
        w = ang_v[2]

        thrusts = CTRL0.compute_control(current_position=pos_x_y,
                                        current_velocity=vel_x_y,
                                        current_theta=th,
                                        current_theta_dot = w,
                                        target_position=TARGET_POSITION[i, :],
                                        target_velocity=TARGET_VELOCITY[i, :],
                                        target_acceleration=TARGET_ACCELERATION[i, :]
                                        )

        #p.setJointMotorControl2(turtle,0,p.VELOCITY_CONTROL,force=0.1)
        #p.setJointMotorControl2(turtle,1,p.VELOCITY_CONTROL,force=-0.1)

        # _, _, _, T1 = p.getJointState(turtle, 0)
        # _, _, _, T2 = p.getJointState(turtle, 1)

        # print(T1, T2)

        # Controller 
        # QP Parameters
        gamma = 1
        qp = QP_Controller_Unicycle(gamma, obs_radius= 0.5)
        u_ref = thrusts
        qp.set_reference_control(u_ref)
        qp.setup_QP(bot, pos_obs, vel_obs) #  
        
        # Simulation
        # Solve QP
        state_of_QP, value_of_h = qp.solve_QP(bot)
        
        # Bot Kinematics
        u_star = qp.get_optimal_control()

        thrusts = u_star

        p.setJointMotorControl2(turtle,0,p.TORQUE_CONTROL,force=thrusts[0])
        p.setJointMotorControl2(turtle,1,p.TORQUE_CONTROL,force=thrusts[1])

        vel = np.linalg.norm(vel_x_y)
        # print(vel)
        bot.update_state(pos_x_y, th, vel, w, 1/SIM_FREQ)

        p.stepSimulation()

        

        #p.applyExternalTorque(turtle, 0, torqueObj = [0,100,0], flags=p.LINK_FRAME)
#p.applyExternalTorque(turtle, 1, torqueObj = [0,100,0], flags=p.LINK_FRAME)


