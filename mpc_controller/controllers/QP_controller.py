# imports
import math
import numpy as np
import matplotlib.pyplot as plt

#from utility_functions import norm
from math import cos, sin, pi, sqrt
from sympy import symbols, diff, Matrix, simplify
from sympy import *
from abc import ABC, abstractmethod


class QP_Controller(ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    def set_reference_control(self):
        pass
    
    @abstractmethod
    def get_reference_control(self):
        pass
    
    @abstractmethod
    def setup_QP(self):
        pass
    
    @abstractmethod
    def solve_QP(self):
        pass
    


        