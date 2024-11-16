# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 19:56:58 2022

@author: Madhusudhan
"""
# imports
import numpy as np

# custom imports
from controllers.reference_trajectory_control.reference_control import AC_UnicycleReferenceControl

class MultiAgentAC_UnicycleReferenceControl:
    def __init__(self, bot_list):
        self.n = len(bot_list)
        self.rc_list = [AC_UnicycleReferenceControl(bot) for bot in bot_list]
        self.ref_control = np.zeros(self.n*2)
        
    def get_reference_control(self):
        for rc_index in range(self.n):
            self.ref_control[rc_index] = self.rc_list[rc_index].a_ref
            self.ref_control[rc_index + 1] = self.rc_list[rc_index].alpha_ref
        return self.ref_control
    
    def set_reference_control_null(self):
        for rc_index in range(self.n):
            self.rc_list[rc_index].reference_control_null()
    
    def bulid_spline_path(self):
        for rc_index in range(self.n):
            self.bulid_spline_path_for_bot(rc_index)
    
    def bulid_spline_path_for_bot(self, bot_index:int):
        self.rc_list[bot_index].build_spline_path()
        
    def set_target_velocity(self,  target_velocity:float = 1 ):
        for rc_index in range(self.n):
            self.set_target_velocity_for_bot(rc_index, target_velocity)
            
    def set_target_velocity_for_bot(self, bot_index:int, target_velocity: float):
        self.rc_list[bot_index].set_target_velocity_for_bot(target_velocity)
    
    def plot_reference_trajectory(self, ax):
        for rc_index in range(self.n):
            self.plot_reference_trajectory_for_bot(rc_index, ax)
          
    def plot_reference_trajectory_for_bot(self, bot_index:int, ax):
        self.rc_list[bot_index].plot_reference_trajectory(ax)
        
        
class MultiAgentDroneReferenceControl:
    def __init__(self, bot_list):
        self.n = len(bot_list)
        self.rc_list = [AC_UnicycleReferenceControl(bot) for bot in bot_list]
        self.ref_control = np.zeros(self.n*2)
        
    def get_reference_control(self):
        for rc_index in range(self.n):
            self.ref_control[rc_index] = self.rc_list[rc_index].a_ref
            self.ref_control[rc_index + 1] = self.rc_list[rc_index].alpha_ref
        return self.ref_control
    
    def set_reference_control_null(self):
        for rc_index in range(self.n):
            self.rc_list[rc_index].reference_control_null()
    
    def bulid_spline_path(self):
        for rc_index in range(self.n):
            self.bulid_spline_path_for_bot(rc_index)
    
    def bulid_spline_path_for_bot(self, bot_index:int):
        self.rc_list[bot_index].build_spline_path()
        
    def set_target_velocity(self,  target_velocity:float = 1 ):
        for rc_index in range(self.n):
            self.set_target_velocity_for_bot(rc_index, target_velocity)
            
    def set_target_velocity_for_bot(self, bot_index:int, target_velocity: float):
        self.rc_list[bot_index].set_target_velocity_for_bot(target_velocity)
    
    def plot_reference_trajectory(self, ax):
        for rc_index in range(self.n):
            self.plot_reference_trajectory_for_bot(rc_index, ax)
          
    def plot_reference_trajectory_for_bot(self, bot_index:int, ax):
        self.rc_list[bot_index].plot_reference_trajectory(ax)
        
        
