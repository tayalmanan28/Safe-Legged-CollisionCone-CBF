import numpy as np

# custom imports
import controllers.reference_trajectory_control.cubic_spline_planner as cubic_spline_planner
from controllers.reference_trajectory_control.stanley_controller import State, pid_control, calc_target_index, stanley_control
from bots.AC_unicycle import AC_Unicycle
from bots.drone import Drone

class AC_UnicycleReferenceControl:
    def __init__(self, unicycle_bot):
        """
        Reference Control class sets the reference trajectory and returns back 
        reference controls based on trajectory tracking algorithms
        lateral control : stanley 
        longitudinal control : PID

        Returns
        -------
        None.

        """
        self.bot :AC_Unicycle = unicycle_bot
        self.a_ref:float = 0.0
        self.alpha_ref:float = 0.0
        
    
    def set_bot_at_the_start(self):
        """
        Given the reference trajectory, the bot is set at the starting point of
        trajectory

        Returns
        -------
        None.

        """
        self.bot.x = self.key_points_x[0]
        self.bot.y = self.key_points_y[0]
        self.bot.theta = self.cyaw[0]
        self.bot.v = self.target_velocity
        self.bot.omega = 0
        
        
    def build_spline_path(self):
        self.cx, self.cy, self.cyaw, self.ck, self.s = cubic_spline_planner.calc_spline_course(self.key_points_x, self.key_points_y, ds=0.1)
        self.last_idx = len(self.cx) - 1
        
    def set_keypoints_manual(self, x_list, y_list):
        self.key_points_x = x_list
        self.key_points_y = y_list
    
    def set_keypoints_from_sin_curve(self, x_start, x_end, step_size):
        step_count = int(abs(x_end - x_start)/step_size)
        x_list = np.linspace(x_start, x_end, step_count)
        y_list = np.sin(x_list)
        self.key_points_x = list(x_list)
        self.key_points_y = list(y_list)
       
    def set_target_velocity(self, target_velocity):
        self.target_velocity = target_velocity
    
    def get_reference_control_stanley(self, bot: AC_Unicycle):
        """
        

        Parameters
        ----------
        bot : AC_Unicycle
            DESCRIPTION.

        Returns
        -------
        float - tuple first position
            acceleration reference control.
        float - tuple second position
            angular acceleration reference controls.

        """
        bot_state = State(x=bot.x, y=bot.y, yaw=bot.theta, v=bot.v)
        target_idx, _ = calc_target_index(bot_state, self.cx, self.cy)
        
        K_alpha = 0.5
        if self.last_idx >= target_idx:
            # longitudinal reference controls
            self.a_ref = pid_control(self.target_velocity, bot_state.v)
            # lateral reference controls
            steer_ref, target_idx = stanley_control(bot_state, self.cx, self.cy, self.cyaw, target_idx)
            self.alpha_ref = K_alpha*(steer_ref - bot.theta)
            
        assert self.last_idx >= target_idx, "Cannot reach goal"
        return (self.a_ref, self.alpha_ref) 
    
    def reference_control_null(self):
        """
        invariant of state of bot a null reference control is set and returned.

        Returns
        -------
         : tuple
            (acceleration reference control,  angular acceleration reference controls).

        """
        self.a_ref = 0.0
        self.alpha_ref = 0.0
        return (self.a_ref, self.alpha_ref)

class DroneReferenceControl:
    def __init__(self, drone):
        """
        Reference Control class sets the reference trajectory and returns back 
        reference controls based on trajectory tracking algorithms
        lateral control : stanley 
        longitudinal control : PID

        Returns
        -------
        None.

        """
        self.bot :Drone = drone
        self.f1_ref:float = 0.0
        self.f2_ref:float = 0.0
        self.f3_ref:float = 0.0
        self.f4_ref:float = 0.0
        
    
    def set_bot_at_the_start(self):
        """
        Given the reference trajectory, the bot is set at the starting point of
        trajectory

        Returns
        -------
        None.

        """
        self.bot.x = self.key_points_x[0]
        self.bot.y = self.key_points_y[0]
        self.bot.z = self.key_points_z[0]
        self.bot.x_d = self.key_points_x_d[0]
        self.bot.y_d = self.key_points_y_d[0]
        self.bot.z_d = self.key_points_z_d[0]
        self.bot.phi = self.cyaw[0]
        self.bot.theta = self.cyaw[0]
        self.bot.psi = self.cyaw[0]
        self.bot.w_1 = self.w[0]
        self.bot.w_2 = self.w[1]
        self.bot.w_3 = self.w[2]
        
    def build_spline_path(self):
        self.cx, self.cy, self.cyaw, self.ck, self.s = cubic_spline_planner.calc_spline_course(self.key_points_x, self.key_points_y, ds=0.1)
        self.last_idx = len(self.cx) - 1
        
    def set_keypoints_manual(self, x_list, y_list, z_list):
        self.key_points_x = x_list
        self.key_points_y = y_list
        self.key_points_z = z_list
    
    def set_keypoints_from_sin_curve(self, x_start, x_end, step_size):
        step_count = int(abs(x_end - x_start)/step_size)
        x_list = np.linspace(x_start, x_end, step_count)
        y_list = np.sin(x_list)
        self.key_points_x = list(x_list)
        self.key_points_y = list(y_list)
       
    def set_target_velocity(self, target_velocity):
        self.target_velocity = target_velocity
    
    def get_reference_control(self, bot: Drone):
        """
        

        Parameters
        ----------
        bot : Drone
            DESCRIPTION.

        Returns
        -------
        float - tuple first position
            acceleration reference control.
        float - tuple second position
            angular acceleration reference controls.

        """
        bot_state = State(x=bot.x, y=bot.y, yaw=bot.theta, v=bot.v)
        target_idx, _ = calc_target_index(bot_state, self.cx, self.cy)
        
        K_alpha = 0.5
        if self.last_idx >= target_idx:
            # longitudinal reference controls
            self.a_ref = pid_control(self.target_velocity, bot_state.v)
            # lateral reference controls
            steer_ref, target_idx = stanley_control(bot_state, self.cx, self.cy, self.cyaw, target_idx)
            self.alpha_ref = K_alpha*(steer_ref - bot.theta)
            
        assert self.last_idx >= target_idx, "Cannot reach goal"
        return (self.a_ref, self.alpha_ref) 
    
    def reference_control_null(self):
        """
        invariant of state of bot a null reference control is set and returned.

        Returns
        -------
         : tuple
            (acceleration reference control,  angular acceleration reference controls).

        """
        self.f1_ref = 0.0
        self.f2_ref = 0.0
        self.f3_ref = 0.0
        self.f4_ref = 0.0
        return (self.f1_ref, self.f2_ref, self.f3_ref, self.f4_ref)