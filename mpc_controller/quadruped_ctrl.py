import numpy as np
# from gym_cbf.envs.BaseAviary import BaseAviary
from scipy import integrate

class Quadruped():
    """"""

    def __init__(self, timestep=0.001):
        """ 
        Parameters
        ----------
        env : BaseAviary
            The PyBullet-based simulation environment.

        """
        self.g = 9.8
        """float: Gravity acceleration, in meters per second squared."""
        self.mass = 2.4#0.0316
        """float: The mass of quad from environment."""
        self.inertia_zz =0.1#env.J[0][0]
        """float: The inertia of quad around x axis."""
        self.L = 0.07
        self.R = 0.033
        """float: The inertia of quad around x axis."""
        self.timestep = timestep#env.TIMESTEP
        """float: Simulation and control timestep."""
        self.last_rpy = np.zeros(3)
        """ndarray: Store the last roll, pitch, and yaw."""
        self.p_coeff_position = {}
        """dict[str, float]: Proportional coefficient(s) for position control."""
        self.d_coeff_position = {}
        """dict[str, float]: Derivative coefficient(s) for position control."""

        self.matrix_u2rpm = np.array([ [1,   1,   1,   1],
                                       [0,   1,   0,  -1],
                                       [1,   0,  -1,   0],
                                       [1,  -1,   1,  -1] 
                                      ])

        self.matrix_u2rpm_inv = np.linalg.inv(self.matrix_u2rpm)

        self.p_coeff_position["x"] = 0.7 * 0.7*15
        self.d_coeff_position["x"] = 2 * 0.5 * 0.7
        
        self.p_coeff_position["y"] = 0.7 * 0.7*15
        self.d_coeff_position["y"] = 2 * 0.5 * 0.7
        
        self.p_coeff_position["t"] = 0.7 * 0.7 * 3
        self.d_coeff_position["t"] = 2 * 2.5 * 0.7*0.00


        self.reset()

    def reset(self):
        """ Resets the controller counter."""
        self.control_counter = 0

    def compute_control(self,
                        current_position,
                        current_velocity,
                        current_theta,
                        current_theta_dot,
                        target_position=np.zeros(2),
                        target_velocity=np.zeros(2),
                        target_acceleration=np.zeros(2),
                        ):
        """Computes the propellers' RPMs for the target state, given the current state.

        Parameters
        ----------
        current_position : ndarray
            (2,)-shaped array of floats containing global x, y, in meters.
        current_velocity : ndarray
            (2,)-shaped array of floats containing global vx, vy, in m/s.
        current_theta : ndarray
            (1,)-shaped array of floats containing yaw, in rad.
        target_position : ndarray
            (2,)-shaped array of float containing global x, y, in meters.
        target_velocity : ndarray, optional
            (2,)-shaped array of floats containing global, in m/s.
        target_acceleration : ndarray, optional
            (2,)-shaped array of floats containing global, in m/s^2.

        Returns
        -------
        ndarray
            (4,)-shaped array of ints containing the desired RPMs of each propeller.
        """
        self.control_counter += 1

        # Compute theta, pitch, and yaw rates
        # current_theta_dot = (current_theta - self.last_theta) / self.timestep

        ## Calculate PD control in x, y
        # x_ddot = self.pd_control(target_position[0],
        #                          current_position[0],
        #                          target_velocity[0],
        #                          current_velocity[0],
        #                          target_acceleration[0],
        #                          "x"
        #                          )
        
        # # print('error',current_position[0]-target_position[0])

        # y_ddot = self.pd_control(target_position[1],
        #                          current_position[1],
        #                          target_velocity[1],
        #                          current_velocity[1],
        #                          target_acceleration[1],
        #                          "y"
        #                          )

        # Calculate desired theta and rates given by PD
        # desired_theta = np.arctan((target_velocity[1]) / (target_velocity[0])) #-y_ddot/(self.g + z_ddot)#
        # desired_theta_dot = (desired_theta - current_theta) / self.timestep #- last_desired_theta_dot
        # theta_ddot = (desired_theta_dot - current_theta_dot[0]) / self.timestep# - last_theta_ddot
        vel_ = np.linalg.norm(current_velocity)

        # print("vel_", vel_)

        desired_acc = self.pd_control(0.162, vel_, 0, 0, 0, "x")

        # print(current_position)

        # desired_theta = -np.arctan(( vel_*self.timestep) / (current_position[1] + 0.00000000001))#np.arctan((target_velocity[1]) / (target_velocity[0]+ 0.00001)) 
        # print("desired_theta", desired_theta)
        desired_theta = 0 -np.arctan((current_position[1]) / (vel_*10 + 0.0001))
        desired_theta_dot = (desired_theta - current_theta) / self.timestep
        self.old_theta = desired_theta
        self.old_theta_dot = desired_theta_dot
        theta_ddot = self.pd_control(desired_theta, 
                                    current_theta,
                                    desired_theta_dot, 
                                    current_theta_dot,
                                    0,
                                    "t"
                                    )
        # print("desired_theta", desired_theta,"desired_theta_dot", desired_theta_dot, "theta_ddot", theta_ddot)
        # print("current_theta", current_theta,"current_theta_dot", current_theta_dot)

        
        # Store the last step's theta, pitch, and yaw
        self.last_theta = current_theta

        # acc = np.linalg.norm(np.array([x_ddot, y_ddot]))

        # if acc > 0.5: acc = 0.5

        acc = desired_acc

        # print(acc)

        # torques = np.array([((self.R)/2)*(self.mass * acc + self.inertia_zz*theta_ddot/self.L),((self.R)/2)*(self.mass* acc - self.inertia_zz*theta_ddot/self.L)])
        # torques = np.array([100,-100])

        # print("torques", torques)

        return acc, theta_ddot

    def pd_control(self,
                   desired_position,
                   current_position,
                   desired_velocity,
                   current_velocity,
                   desired_acceleration,
                   opt
                   ):
        """Computes PD control for the acceleration minimizing position error.

        Parameters
        ----------
        desired_position :
            float: Desired global position.
        current_position :
            float: Current global position.
        desired_velocity :
            float: Desired global velocity.
        current_velocity :
            float: Current global velocity.
        desired_acceleration :
            float: Desired global acceleration.

        Returns
        -------
        float
            The commanded acceleration.
        """
        u = desired_acceleration + \
            self.d_coeff_position[opt] * (desired_velocity - current_velocity) + \
            self.p_coeff_position[opt] * (desired_position - current_position) 

        return u
