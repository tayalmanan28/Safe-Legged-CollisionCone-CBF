import math
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# custom imports
from utility_functions import norm


class Drone:
    def __init__(self, name:str, x:float, y:float, z:float,  x_d:float, y_d:float, z_d:float, phi:float, theta:float, psi:float, w_1:float, w_2:float, w_3:float,
                 Ixx:float, Iyy:float, Izz:float, m:float, length_offset_COM:float, dimensions:tuple):
        """
        Construction of Drone class

        Parameters
        ----------
        name : str
            name of the bot.
        x : float
            x position of bot from global frame of reference.
        y : float
            y position of bot from global frame of reference.
        z : float
            z position of bot from global frame of reference.
        x_d : float
            x velocity of bot from global frame of reference.
        y_d : float
            y velocity of bot from global frame of reference.
        z_d : float
            z velocity of bot from global frame of reference.
        phi : float
            orientation of bot from global frame of reference.
        theta : float
            orientation of bot from global frame of reference.
        psi : float
            orientation of bot from global frame of reference.
        w_1 : float
            angular velocity of bot.
        w_2 : float
            angular velocity of bot.
        w_3 : float
            angular velocity of bot.
        length_offset_COM : float
            distance from axis center to COM of bot.
        dimensions : tuple
            first entry of tuple contains lenght of bot.
            second entry of tuple contains width of bot.
            third entry of tuple contains width of bot.
            
        Returns
        -------
        None.

        """
        self.g = 9.81
        self.m = m
        self.Ixx = Ixx
        self.Iyy = Iyy
        self.Izz = Izz

        # State Variables
        self.name = name
        self.x = x
        self.y = y
        self.z = z
        self.phi = phi
        self.theta = theta
        self.psi = psi
        self.w_1 = w_1
        self.w_2 = w_2
        self.w_3 = w_3
        self.l = length_offset_COM
        length, width, height = dimensions
        
        self.length = length
        self.width = width
        self.height = height
        self.encompassing_radius = norm(self.length, self.width, self.height)
        self.L = 1.414*length
        
        # time derivatives
        self.x_dot = x_d
        self.y_dot = y_d
        self.z_dot = z_d
        self.x_ddot = 0
        self.y_ddot = 0
        self.z_ddot = 0
        self.phi_dot = 0
        self.theta_dot = 0
        self.psi_dot = 0
        self.w_1_dot = 0
        self.w_2_dot = 0
        self.w_3_dot = 0
        
        # control variables
        self.f1 = 0
        self.f2 = 0
        self.f3 = 0
        self.f4 = 0
    
    @classmethod
    def from_JSON(cls, file_path: str):
        """
        This function from the given config files bulid a bot with initial
        state variables values and dimentions and returns the bot objects

        Parameters
        ----------
        file_path : str
            path of JSON configuration file

        Returns
        -------
        bot : Drone
            the bulit bot object

        """
        with open(file_path, 'r') as bot_json:
            bot_dict = json.load(bot_json)
            bot = cls(bot_dict['name'], 
                      bot_dict['states']['x'], 
                      bot_dict['states']['y'],
                      bot_dict['states']['z'], 
                      bot_dict['states']['x_d'], 
                      bot_dict['states']['y_d'],
                      bot_dict['states']['z_d'], 
                      bot_dict['states']['phi'],
                      bot_dict['states']['theta'],
                      bot_dict['states']['psi'],
                      bot_dict['states']['w_1'],
                      bot_dict['states']['w_2'],
                      bot_dict['states']['w_3'],
                      bot_dict['mass-inertia']['Ixx'],
                      bot_dict['mass-inertia']['Iyy'],
                      bot_dict['mass-inertia']['Izz'],
                      bot_dict['mass-inertia']['m'],
                      bot_dict['dimensions']['length_offset_COM'],
                      (bot_dict['dimensions']['length'], bot_dict['dimensions']['width'], bot_dict['dimensions']['height']))
        return bot
        
    def update_state(self, p, v, rpy, delta_t: float):
        """
        given x_t x_(t+1) is calculated using x_(t+1) = x_t + x_dot*delta_t,
        updating states for the next time step based on calculated derivatives

        Parameters
        ----------
        delta_t : float
            simulation time step.

        Returns
        -------
        None.

        """
        self.x = p[0]
        self.y = p[1]
        self.z = p[2]
        self.x_dot = v[0]
        self.y_dot = v[1]
        self.z_dot = v[2]
        self.phi = rpy[0]
        self.theta = rpy[1]
        self.psi = rpy[2]
        self.w_1 = self.phi -math.sin(self.theta)*self.psi
        self.w_2 = math.cos(self.phi)*self.theta + math.cos(self.theta)*math.sin(self.phi)*self.psi
        self.w_3 = -math.sin(self.phi)*self.theta + math.cos(self.theta)*math.cos(self.phi)*self.psi
    
    def set_control(self, f1:float, f2:float, f3:float, f4:float):
        """
        takes control values values and sets the control variables

        Parameters
        ----------
        f1, f2, f3, f4 : float
            Thrust force on drone rotors
        
        Returns
        -------
        None.

        """
        self.f1 = f1
        self.f2 = f2
        self.f3 = f3
        self.f4 = f4
    
    def apply_control(self):
        """
        the function applies control and takes a time step in control system. 
        This function should be executed only once during a simulation timestep

        Returns
        -------
        None.

        """
        av_f = (self.f1 + self.f2 + self.f3 + self.f4)/self.m
        self.x_ddot = (math.cos(self.psi)*math.sin(self.theta)*math.cos(self.phi) + math.sin(self.psi)*math.sin(self.phi))*av_f
        # print('x_ddot', self.x_ddot)
        self.y_ddot = (math.sin(self.psi)*math.sin(self.theta)*math.cos(self.phi) - math.cos(self.psi)*math.sin(self.phi))*av_f
        self.z_ddot = - self.g  + (math.cos(self.theta)*math.cos(self.phi)) * av_f
        self.phi_dot = self.w_1 + self.w_2*math.sin(self.phi)*math.tan(self.theta) + self.w_3*math.cos(self.phi)*math.tan(self.theta)
        self.theta_dot = self.w_2*math.cos(self.phi) - self.w_3*math.sin(self.phi)
        self.psi_dot = (self.w_2*math.sin(self.phi) + self.w_3*math.cos(self.phi))/math.cos(self.theta)
        self.w_1_dot = (self.Iyy - self.Izz)*self.w_2*self.w_3/self.Ixx + self.L*(self.f1-self.f3)/self.Ixx
        self.w_2_dot = (self.Izz - self.Ixx)*self.w_1*self.w_3/self.Iyy + self.L*(self.f2-self.f4)/self.Iyy
        self.w_3_dot = (self.Ixx - self.Iyy)*self.w_1*self.w_2/self.Izz
                