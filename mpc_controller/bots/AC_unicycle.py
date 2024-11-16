# imports
import math
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# custom imports
from utility_functions import norm2


class Unicycle:
    def __init__(self, name:str, x:float, y:float, theta:float, v:float, w:float,
                 Ixx:float, Iyy:float, Izz:float, m:float,
                 length_offset_COM:float, radius:float, dimensions:tuple):
        """
        Construction of AC_Unicycle class

        Parameters
        ----------
        name : str
            name of the bot.
        x : float
            x position of bot from global frame of reference.
        y : float
            y position of bot from global frame of reference..
        theta : float
            orientation of bot from globla frame of reference.
        v : float
            linear velocity of bot.
        w : float
            angular velocity of bot.
        length_offset_COM : float
            distance from axis center to COM of bot.
        dimensions : tuple
            first entry of tuple contains lenght of bot.
            second entry og tuple contains width of bot.

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
        self.theta = theta
        self.v = v
        self.w = w
        self.L = length_offset_COM
        self.R = radius
        length, width = dimensions
        self.length = length
        self.width = width
        self.encompassing_radius = norm2(self.length, self.width)/2
        
        # time derivatives
        self.x_dot = 0
        self.y_dot = 0
        self.theta_dot = 0
        self.v_dot = 0
        self.w_dot = 0
        
        # control variables
        self.u_1 = 0
        self.u_2 = 0
    
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
        bot : AC_Unicycle
            the bulit bot object

        """
        with open(file_path, 'r') as bot_json:
            bot_dict = json.load(bot_json)
            bot = cls(bot_dict['name'], 
                      bot_dict['states']['x'], 
                      bot_dict['states']['y'], 
                      bot_dict['states']['theta'],
                      bot_dict['states']['v'],
                      bot_dict['states']['w'],                      
                      bot_dict['mass-inertia']['Ixx'],
                      bot_dict['mass-inertia']['Iyy'],
                      bot_dict['mass-inertia']['Izz'],
                      bot_dict['mass-inertia']['m'],
                      bot_dict['dimensions']['length_offset_COM'],
                      bot_dict['dimensions']['radius'],
                     (bot_dict['dimensions']['length'], bot_dict['dimensions']['width']))
        return bot
        
    def update_state(self, p, theta, v, w, delta_t: float):
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
        self.theta = theta
        self.v = v
        self.w = w
    
    def set_control(self, u_1:float, u_2:float):
        """
        takes control values values and sets the control variables

        Parameters
        ----------
        u_1 : float
            Torque on left wheel.
        u_2 : float
            Torque on right wheel.
        
        Returns
        -------
        None.

        """
        self.u_1 = u_1
        self.u_2 = u_2
    
    def apply_contorl(self):
        """
        the function applies control and takes a time step in control system. 
        This function should be executed only once during a simulation timestep

        Returns
        -------
        None.

        """
        self.x_dot = self.v*math.cos(self.theta) 
        self.y_dot = self.v*math.sin(self.theta) 
        self.theta_dot = self.w
        self.v_dot = (self.u_1 + self.u_2)/(self.m*self.R)
        self.w_dot = (self.u_1 - self.u_2)*self.L/(self.Izz*self.R)