import sympy
import numpy as np
from sympy import symbols, diff, Matrix, simplify
from math import cos, sin, tan, pi, sqrt
from sympy import *
from cvxopt import matrix, solvers
import time

# custom imports
from controllers.QP_controller import QP_Controller
from bots.AC_unicycle import Unicycle

solvers.options['show_progress'] = False

def norm(x, y):
    return sqrt(x**2 + y**2)

class QP_Controller_Unicycle(QP_Controller):
    def __init__(self, gamma:float, obs_radius=0.1):
        """
        Constructor creates QP_Controller object for Multi Agent system where 
        each agent is a Unicycle model bot.        

        Parameters
        ----------
        gamma : float
            class k function is taken as simple function f(x) = \gamma x. The 
            valueo f gamma is assumed to 1 for all the agents for time being.
        no_of_agents : int
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.gamma = gamma
        self.u_ref = None
        self.u_star = None
        self.G = 9.81
        self.obs_r = obs_radius
        self.k = 0
        
    def set_reference_control(self, u_ref:np.ndarray):
        """
        sets the reference controller

        Parameters
        ----------
        u_ref : numpy.ndarray
            numpy array of size (n, 2) containing the reference controls.

        Returns
        -------
        None.

        """
        self.u_ref = u_ref
        self.u_star = np.copy(u_ref)
    
    def get_optimal_control(self):
        return self.u_star
    
    def get_reference_control(self):
        return self.u_ref
    
    
    def setup_QP(self, bot, c, c_d):# k, k_d, m, m_d):
        """
        the function takes bot list and creates symbolic varaibles associated 
        with each bot required for computation in QP. The functions also 
        precomputes the required symbolic expressions for QP such as L_f, L_g,
        etc...
        Parameters
        ----------
        bot_list : list
            list of Unicycle bot objects 
            
        Returns
        -------
        None.

        """
        c_x, c_y = c
        c_x_d, c_y_d = c_d

        # k_x, k_y = k
        # k_x_d, k_y_d = k_d

        # m_x, m_y = m
        # m_x_d, m_y_d = m_d

        # c_x = (k_x + m_x)/2
        # c_y = (k_y + m_y)/2

        # c_x_d = (k_x_d + m_x_d)/2
        # c_y_d = (k_y_d + m_y_d)/2

        # Create placholders for symbolic expressions
        self.f = [0] # f Matrix in control system
        self.g = [0] # g Matrix in control system
        self.h = [[0]] # C3BF Matrix
        
        # Create placholder for terms in QP
        self.Psi = []
        self.B = [[]]
        self.C = [[]]
        
        # create state and parameter symbolic varaibles for each bot
        
        symbols_string = 'x y v theta w L R Ixx Iyy Izz m r'
        bot.sym_x, bot.sym_y, bot.sym_v, bot.sym_theta, bot.sym_w, bot.sym_L, bot.sym_R, bot.sym_Ixx, bot.sym_Iyy, bot.sym_Izz, bot.sym_m, bot.sym_r =  symbols(symbols_string)
        self.f = Matrix([bot.sym_v*cos(bot.sym_theta), 
                         bot.sym_v*sin(bot.sym_theta),
                         bot.sym_w,
                         0, 
                         0])
        
        self.g = Matrix([[0, 0],
                         [0, 0],
                         [0, 0],
                         [1, 0],
                         [0, 1]])
        
        # Relative position terms
        p_rel_x = c_x - (bot.sym_x + bot.sym_L*cos(bot.sym_theta))
        p_rel_y = c_y - (bot.sym_y + bot.sym_L*sin(bot.sym_theta))

        # k_rel_x = k_x - (bot.sym_x + bot.sym_L*cos(bot.sym_theta))
        # k_rel_y = k_y - (bot.sym_y + bot.sym_L*sin(bot.sym_theta))

        # m_rel_x = m_x - (bot.sym_x + bot.sym_L*cos(bot.sym_theta))
        # m_rel_y = m_y - (bot.sym_y + bot.sym_L*sin(bot.sym_theta))
        
        # Relative velocity terms
        v_rel_x = c_x_d - (bot.sym_v*cos(bot.sym_theta) - bot.sym_L*sin(bot.sym_theta)*bot.sym_w) + 0.0001
        v_rel_y = c_y_d - (bot.sym_v*sin(bot.sym_theta) + bot.sym_L*cos(bot.sym_theta)*bot.sym_w) + 0.0001
        
        # # Cosine terms
        # self.cos1 = (p_rel_x*k_rel_x + p_rel_y*k_rel_y)/norm(k_rel_x, k_rel_y)
        # self.cos2 = (p_rel_x*m_rel_x + p_rel_y*m_rel_y)/norm(m_rel_x, m_rel_y)
        
        # 3-D C3BF Candidate
        self.h = p_rel_x*v_rel_x + p_rel_y*v_rel_y \
            + norm(v_rel_x, v_rel_y)*sqrt(norm(p_rel_x, p_rel_y)**2 - bot.sym_r**2)
        
        # # PC3BF Candidate
        # print("k = ", self.k)
        # if self.k == 1:
        #     self.h = p_rel_x*v_rel_x + p_rel_y*v_rel_y \
        #         + norm(v_rel_x, v_rel_y)*(p_rel_x*k_rel_x + p_rel_y*k_rel_y)/norm(k_rel_x, k_rel_y)
        # elif self.k == 2:
        #     self.h = p_rel_x*v_rel_x + p_rel_y*v_rel_y \
        #         + norm(v_rel_x, v_rel_y)*(p_rel_x*m_rel_x + p_rel_y*m_rel_y)/norm(m_rel_x, m_rel_y)
        # else:
        #     self.h = bot.sym_m
        

        # # HO-CBF Candidate
        # gamma = 1
        # p = 0.5
        # self.h = p_rel_x*v_rel_x + p_rel_y*v_rel_y + p_rel_z*v_rel_z \
        #     + gamma*(norm(p_rel_x, p_rel_y, p_rel_z)**2 - bot.sym_r**2)**p

        # self.h =1

        # # Classical CBF
        # self.h = norm(c_x - bot.sym_x, c_y - bot.sym_y, c_z - bot.sym_z)**2/bot.sym_r**2 -1
            
        rho_h_by_rho_x = diff(self.h, bot.sym_x)
        rho_h_by_rho_y = diff(self.h, bot.sym_y)
        rho_h_by_rho_v = diff(self.h, bot.sym_v)
        rho_h_by_rho_theta = diff(self.h, bot.sym_theta)
        rho_h_by_rho_w = diff(self.h, bot.sym_w)
        
        Delta_h_wrt_bot = Matrix([[rho_h_by_rho_x, 
                                   rho_h_by_rho_y, 
                                   rho_h_by_rho_v, 
                                   rho_h_by_rho_theta,
                                   rho_h_by_rho_w]])
        
        self.C = Delta_h_wrt_bot*self.g
        self.u_ref = self.u_ref.reshape((2,1))
        n = self.C * self.u_ref
        n_f = Delta_h_wrt_bot*self.f
        self.Psi = self.gamma*self.h
        self.Psi += n_f[0] + n[0]
        self.B = (self.C).transpose()
        # print(np.matmul(self.B, 1/(np.matmul(self.C,self.B))).dot(self.Psi))
                 
    def solve_QP(self, bot):
        """
        Solving Quadratic Program to set the optimal controls. This functions
        substitutes the values in symbolic expression and evalutes closed form
        solution to QP, modifies the reference control and sets the optimal 
        control

        Parameters
        ----------
        bot_list : list
            list of Unicycle bot objects .

        Returns
        -------
        TYPE: tuple of 2 numpy arrays
            first numpy array in the tuple returns of state of CBF if they are
            active or inactive 1 denotes active CBF and 0 denotes inactive CBF.
            second numpy array in the tupes returns the value of CBF since in 
            C3BF the value of the function is directly proportional to how 
            unsafe system is.
        """
        # build value substitution list
        uk_vs = [bot.sym_x, bot.sym_y,
                bot.sym_v, bot.sym_theta, bot.sym_w,
                bot.sym_L, bot.sym_R,
                bot.sym_Ixx, bot.sym_Iyy, bot.sym_Izz, 
                bot.sym_m, bot.sym_r]

        uk_gs = [ bot.x,  bot.y,
                 bot.v,  bot.theta, bot.w,
                 bot.L, bot.R, 
                 bot.Ixx,  bot.Iyy,  bot.Izz, 
                 bot.m,  bot.encompassing_radius + self.obs_r ]

        d = {uk: uk_gs[i] for i, uk in enumerate(uk_vs)}

        # build value substitution list        
        self.h = np.array(re(self.h.xreplace(d)))
        self.Psi = np.array(re(self.Psi.xreplace(d)))
        self.C = np.array(re(self.C.xreplace(d)))
        self.B = (self.C).transpose()
        # self.cos1 = np.array(re(self.cos1.xreplace(d)))
        # self.cos2 = np.array(re(self.cos2.xreplace(d)))

        # if self.cos1<self.cos2:
        #     self.k = 1
        # elif self.cos1>self.cos2:
        #     self.k = 2
        # else:
        #     self.k = 0

        # print('cos1',self.cos1, 'cos2',self.cos2, 'k',self.k) 

        print('h',self.h)

        if self.Psi<0:
            self.u_safe = - np.matmul(self.B, np.linalg.inv(np.matmul(self.C,self.B).astype('float64'))).dot(self.Psi)
            self.u_safe[1] = self.u_safe[1]
            print('u_safe',self.u_safe)
        else:
            self.u_safe = 0
            
        self.u_star = self.u_ref + self.u_safe

        # print(self.u_star)
        state_of_h1, state_of_h2 = 0, 0 
        term_h1 , term_h2 = 0, 0
        return ((state_of_h1, state_of_h2), (term_h1, term_h2))