#!/bin/python3
"""

The Obstacle classes containing the neccessary gradients and hessian functions for
seamless integration with optimal solvers, includes several utility objects like 
the obstacle list for use in real time simulation.

"""
# Removal of the following method for Type Hinting Enclosing
# classes is possible. Be cautious about the changes.

from __future__ import annotations

import sys
import os
import warnings
import enum

import numpy as np

from euclid import *
from cvxopt import matrix
from collections.abc import MutableMapping

from cbf.utils import vec_norm


sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                "../")

try:
    from cbf.geometry import Rotation, Transform
    from cbf.utils import Timer, ZERO_TOL
except:
    raise

# Identity Objects
class IdentityObjects(enum.Enum):
    """
    Enumerations for Required Empty Identity Objects.
    """
    DICT_EMPTY_UPDATE = ()

# Object Selectors for utility
class Obstacle3DTypes(enum.Enum):
    """
    Enumerations for the available 3D obstacle classes.
    """
    ELLIPSE3D = 0
    COLLISION_CONE3D = 1

class BoundingBox():
    def __init__(self, extent=Vector3(), location=Vector3(), rotation=Rotation(), velocity=Vector3()):
        self.extent = extent
        self.location = location
        self.rotation = rotation
        self.velocity = velocity

    def __eq__(self, other):
        return self.location == other.location and self.extent == other.extent
    
    def __ne__(self, other):
        return not self.__eq__(other)

    def get_local_vertices(self):
        up = self.rotation.get_up_vector().normalized()
        right = self.rotation.get_right_vector().normalized()
        forward = self.rotation.get_forward_vector().normalized()
        v1 = -self.extent.z*up + self.extent.x*forward + self.extent.y*right
        v2 = -self.extent.z*up + self.extent.x*forward - self.extent.y*right
        v3 = -self.extent.z*up - self.extent.x*forward - self.extent.y*right
        v4 = -self.extent.z*up - self.extent.x*forward + self.extent.y*right
        v5 = self.extent.z*up + self.extent.x*forward + self.extent.y*right
        v6 = self.extent.z*up + self.extent.x*forward - self.extent.y*right
        v7 = self.extent.z*up - self.extent.x*forward - self.extent.y*right
        v8 = self.extent.z*up - self.extent.x*forward + self.extent.y*right
        return [v1, v2, v3, v4, v5, v6, v7, v8]

    def get_world_vertices(self, transform=Transform):
        v_list = self.get_local_vertices()
        return [transform.transform(v) for v in v_list]

class Obstacle3DBase():
    """
    The base class each 3D obstacle class will inherit from. Created to enforce specific
    validation checks in the obstacle list objects and creating the neccessary interface
    for all 3D obstacle CBF classes.
    """
    def __init__(self):
        pass

    def evaluate(self, p):
        if not isinstance(p, Vector3):
            raise TypeError("Expected an object of type euclid.Vector3 for arg p, but got " + type(p).__name__ + ".")

    def gradient(self, p):
        if not isinstance(p, Vector3):
            raise TypeError("Expected an object of type euclid.Vector3 for arg p, but got " + type(p).__name__ + ".")
        return matrix(0.0, (3,1))

    def f(self, p):
        if not isinstance(p, Vector3):
            raise TypeError("Expected an object of type euclid.Vector3 for arg p, but got " + type(p).__name__ + ".")
        return 0
    
    def dx(self, p):
        if not isinstance(p, Vector3):
            raise TypeError("Expected an object of type euclid.Vector3 for arg p, but got " + type(p).__name__ + ".")
        return 0
    
    def dy(self, p):
        if not isinstance(p, Vector3):
            raise TypeError("Expected an object of type euclid.Vector3 for arg p, but got " + type(p).__name__ + ".")
        return 0

    def dz(self, p):
        if not isinstance(p, Vector3):
            raise TypeError("Expected an object of type euclid.Vector3 for arg p, but got " + type(p).__name__ + ".")
        return 0

    def dphi(self, p):
        if not isinstance(p, Vector3):
            raise TypeError("Expected an object of type euclid.Vector3 for arg p, but got " + type(p).__name__ + ".")
        return 0

    def dtheta(self, p):
        if not isinstance(p, Vector3):
            raise TypeError("Expected an object of type euclid.Vector3 for arg p, but got " + type(p).__name__ + ".")
        return 0

    def dpsi(self, p):
        if not isinstance(p, Vector3):
            raise TypeError("Expected an object of type euclid.Vector3 for arg p, but got " + type(p).__name__ + ".")
        return 0

    def dv(self, p):
        if not isinstance(p, Vector3):
            raise TypeError("Expected an object of type euclid.Vector3 for arg p, but got " + type(p).__name__ + ".")
        return 0
    
    def dt(self, p: Point3):
        if not isinstance(p, Vector3):
            raise TypeError("Expected an object of type euclid.Vector3 for arg p, but got " + type(p).__name__ + ".")
        return 0
    def update(self):
        pass

    def update_coords(self, xy):
        if not isinstance(xy, Vector3):
            raise TypeError("Expected an object of type euclid.Vector3 for arg p, but got " + type(xy).__name__ + ".")
        pass

    def update_orientation(self):
        pass

class Ellipse3D(Obstacle3DBase):
    """
    Generates the 3D Ellipse obstacle representation for use in control barrier functions.
    Exposes the required functionality for direct usage in CBF as a barrier constraint.

    """
    def __init__(self, a: float, b: float, c: float, center: Vector3 = Vector3(0, 0), phi: float=0, theta: float=0, psi: float=0, buffer: float=0):
        """
        Initializes the Ellipse3D Object. 
        """
        if not isinstance(center, Vector3):
            raise TypeError("Expected an object of type euclid.Vector3 for arg center, but got " + type(center).__name__ + ".")
        self.center = center
        self.phi = phi
        self.theta = theta
        self.psi = psi
        self.vel = Vector3()
        self.a = a + buffer
        self.b = b + buffer
        self.c = c + buffer
        self.buffer = buffer
        self.BUFFER_FLAG = True

    def __repr__(self):
        return f"{type(self).__name__}(a = {self.a}, b = {self.b}, c = {self.c}, center = {self.center}, phi = {self.phi}, theta = {self.theta}, psi = {self.psi}, buffer = {self.buffer}, buffer_applied: {self.BUFFER_FLAG} )\n"
    
    def apply_buffer(self):
        if not self.BUFFER_FLAG:
            self.a = self.a + self.buffer
            self.b = self.b + self.buffer
            self.c = self.c + self.buffer
            self.BUFFER_FLAG = True
        else:
            warnings.warn("Warning: Buffer already applied. Call Ignored.")
        
    def remove_buffer(self):
        if self.BUFFER_FLAG:
            self.a = self.a - self.buffer
            self.b = self.b - self.buffer
            self.c = self.c - self.buffer
            self.BUFFER_FLAG = False
        else:
            warnings.warn("Warning: Buffer already removed. Call Ignored.")
    
    def evaluate(self, p: Point3):
        """
        Evaluate the value of the ellipse at a given point.
        """
        super().evaluate(p)
        dx = p.x - self.center.x
        dy = p.y - self.center.y
        dz = p.z - self.center.z
        cph = np.cos(self.phi)
        sph = np.sin(self.phi)
        ct = np.cos(self.theta)
        st = np.sin(self.theta)
        cps = np.cos(self.psi)
        sps = np.sin(self.psi)

        eval = ( ( dx * ct + dy * st )/self.a )**2 + ( ( -dx * st + dy * ct )/self.b )**2 - 1
        return eval

    def gradient(self, p: Point3):
        super().gradient(p)
        return matrix([self.dx(p), self.dy(p), self.dz(p), self.dphi(p), self.dtheta(p), self.dpsi(p)])

    # f = evaluate
        
    def f(self, p: Point3):
        """
        Alias of the evaluate function, semantically significant for cvxopt.
        """
        return self.evaluate(p)
    
    def dx(self, p: Point3):
        super().dx(p)
        xd = p.x - self.center.x
        yd = p.y - self.center.y
        zd = p.z - self.center.z
        cph = np.cos(self.phi)
        sph = np.sin(self.phi)
        ct = np.cos(self.theta)
        st = np.sin(self.theta)
        cps = np.cos(self.psi)
        sps = np.sin(self.psi)

        dx_ = (2 * ct/(self.a**2)) * ( xd * ct + yd * st ) + (-2 * st/(self.b**2)) * ( -xd * st + yd * ct )
        return dx_
    
    def dy(self, p: Point3):
        super().dy(p)
        xd = p.x - self.center.x
        yd = p.y - self.center.y
        zd = p.z - self.center.z
        cph = np.cos(self.phi)
        sph = np.sin(self.phi)
        ct = np.cos(self.theta)
        st = np.sin(self.theta)
        cps = np.cos(self.psi)
        sps = np.sin(self.psi)

        dy_ = (2 * st/(self.a**2)) * ( xd * ct + yd * st ) + (2 * ct/(self.b**2)) * ( -xd * st + yd * ct )
        return dy_

    def dz(self, p: Point3):
        super().dz(p)
        xd = p.x - self.center.x
        yd = p.y - self.center.y
        zd = p.z - self.center.z
        cph = np.cos(self.phi)
        sph = np.sin(self.phi)
        ct = np.cos(self.theta)
        st = np.sin(self.theta)
        cps = np.cos(self.psi)
        sps = np.sin(self.psi)

        dz_ = (2 * ct/(self.a**2)) * ( xd * ct + yd * st ) + (-2 * st/(self.b**2)) * ( -xd * st + yd * ct )
        return dz_

    def dv(self, p: Point3):
        """
        Despite being zero. This function is still created for the sake of completeness w.r.t API.
        """
        return super().dy(p)
    
    def update(self, a: float=None, b: float=None, c: float=None, center: float=None, phi: float=None, theta: float=None, psi: float=None, buffer: float=None):
        if a is not None:
            self.a = a
        if b is not None:
            self.b = b
        if c is not None:
            self.c = c
        if center is not None:
            if not isinstance(center, Vector3):
                raise TypeError("Expected an object of type euclid.Vector3 for arg center.")
            self.center = center
        if phi is not None:
            self.phi = phi
        if theta is not None:
            self.theta = theta
        if psi is not None:
            self.psi = psi
        if buffer is not None:
            if self.BUFFER_FLAG:
                self.a = self.a - self.buffer + buffer
                self.b = self.b - self.buffer + buffer
                self.c = self.c - self.buffer + buffer
                self.buffer = buffer
            else:
                self.buffer = buffer
    
    def update_coords(self, xyz: Point3):
        super().update_coords(xyz)
        self.center = xyz
    
    def update_state(self, xyz: Point3, phi: float, theta: float, psi: float, v: Vector3):
        self.update_coords(xyz)
        self.vel = v
        self.phi = phi
        self.theta = theta
        self.psi = psi
    
    def update_velocity_by_magnitude(self, v: float):
        """
        Assumes that theta is the heading the calculates the vector.
        """
        self.vel = Vector3(x=v*np.cos(self.theta), y=v*np.sin(self.theta))
        pass

    def update_velocity(self, v: Vector3):
        """
        Sets the velocity using the Vector3 object. Note that this will
        create a copy of the argument's object to avoid external mutation
        of the attributes.
        """
        self.vel = v.copy()
        pass

    def update_orientation(self, roll: float, pitch: float, yaw: float):
        self.phi = roll
        self.theta = pitch
        self.psi = yaw
        _v_mag = self.vel.magnitude()
        self.update_velocity_by_magnitude(_v_mag)
        pass

    def update_by_bounding_box(self, bbox: BoundingBox):
        if not isinstance(bbox, BoundingBox):
            raise TypeError("Expected an object of type cbf.obstacles.BoundingBox as an input to fromBoundingBox() method, but got ", type(bbox).__name__)
            
        a = bbox.extent.x
        b = bbox.extent.y
        c = bbox.extent.z
        center = Vector3(bbox.location.x, bbox.location.y, bbox.location.z)
        phi = bbox.rotation.roll
        theta = bbox.rotation.pitch
        psi = bbox.rotation.yaw
        self.update(a=a, b=b, c=c, center=center, phi=phi, theta=theta, psi=psi)

    def dphi(self, p: Point3):
        """
        Despite being zero. This function is still created for the sake of completeness w.r.t API.
        """
        return super().dphi(p)

    def dtheta(self, p: Point3):
        """
        Despite being zero. This function is still created for the sake of completeness w.r.t API.
        """
        return super().dtheta(p)

    def dpsi(self, p: Point3):
        """
        Despite being zero. This function is still created for the sake of completeness w.r.t API.
        """
        return super().dpsi(p)
    
    def dt(self, p: Point3):
        super().dt(p)
        xd = p.x - self.center.x
        yd = p.y - self.center.y
        zd = p.z - self.center.z

        dt_ = -2 * ( (xd/self.a**2) * self.vel.x + (yd/self.b**2) * self.vel.y + (zd/self.c**2) * self.vel.z )
        return dt_

    
    @classmethod
    def from_bounding_box(cls, bbox = BoundingBox(), buffer = 0.5) -> Ellipse3D:
        if not isinstance(bbox, BoundingBox):
            raise TypeError("Expected an object of type cbf.obstacles.BoundingBox as an input to fromBoundingBox() method, but got ", type(bbox).__name__)
        
        a = bbox.extent.x
        b = bbox.extent.y
        c = bbox.extent.z
        center = Vector3(bbox.location.x, bbox.location.y, bbox.location.z)
        phi = bbox.rotation.roll
        theta = bbox.rotation.pitch
        psi = bbox.rotation.yaw
        return cls(a, b, c, center, phi, theta, psi, buffer)
    
class CollisionCone3D(Obstacle3DBase):
    """
    Generates a 3D Collision Cone based CBF for dynamic obstacle avoidance.
    """
    def __init__(self, 
                 a: float = 0.0, 
                 s: matrix = matrix(0, (12,1)), 
                 s_obs: matrix = matrix(0, (12,1)),
                 buffer: float=1.50):
        """
        Initializes the CollisionCone3D Object. F
        """
        self.s = s
        self.s_obs = s_obs
        self.a = a + buffer
        self.buffer = buffer
        self.BUFFER_FLAG = True
        
        self.s = matrix(s)
        self.s_obs = matrix(s_obs)
        self.cx = self.s_obs[0]
        self.cy = self.s_obs[1]
        self.cz = self.s_obs[2]
        self.s_vx = s[3]
        self.s_vy = s[4]
        self.s_vz = s[5]
        self.s_obs_vx = s_obs[3]
        self.s_obs_vy = s_obs[4]
        self.s_obs_vz = s_obs[5]
        self.p_rel = self.s[:3] - self.s_obs[:3]
        self.v_rel = matrix([ self.s_vx - self.s_obs_vx, self.s_vy - self.s_obs_vy, self.s_vz - self.s_obs_vz])
        self.dist = vec_norm(self.p_rel)
        self.v_rel_norm = vec_norm(self.v_rel)
        self.cone_boundary = 0
        
        if (self.dist - self.a) >= 0:
            self.cone_boundary = np.sqrt(self.dist**2 - self.a**2) + ZERO_TOL
        
        if self.dist > ZERO_TOL:
            self.cos_phi = self.cone_boundary/self.dist
        else:
            self.cos_phi = np.pi/2
        
    def __repr__(self):
        return f"{type(self).__name__}(a = {self.a}, center = {self.center}, theta = {self.theta}, buffer = {self.buffer}, buffer_applied: {self.BUFFER_FLAG} )\n"
    
    def apply_buffer(self):
        if not self.BUFFER_FLAG:
            self.a = self.a + self.buffer
            self.BUFFER_FLAG = True
        else:
            warnings.warn("Warning: Buffer already applied. Call Ignored.")
        
    def remove_buffer(self):
        if self.BUFFER_FLAG:
            self.a = self.a - self.buffer
            self.BUFFER_FLAG = False
        else:
            warnings.warn("Warning: Buffer already removed. Call Ignored.")
    
    def evaluate(self, p: Point3):
        """
        Since the cone depends on relative parameter this function uses the 
        current state to calculate the evaluation of the cone at the current point
        therefore doesn't take any other arguments. It is mandatory to update the
        state of the vehicle for this obstacle type to function properly.
        """
        eval = (self.p_rel.T * self.v_rel) + (self.dist * self.v_rel_norm * self.cos_phi)
        return eval

    def gradient(self, p: Point3):
        return matrix([self.dx(), self.dy(), self.dz(), self.dvx(), self.dvy(), self.dvz(), self.dphi(), self.dtheta(), self.dpsi(), self.dw1(), self.dw2(), self.dw3()])

    # f = evaluate
        
    def f(self, p: Point3):
        """
        Alias of the evaluate function, semantically significant for cvxopt.
        """
        return self.evaluate(p)
    
    def dx(self, p: Point3):

        q_dx = self.s_vx - self.s_obs_vx
        phi_term_dx = self.v_rel_norm * (self.s[0] - self.cx)/(self.cone_boundary + ZERO_TOL)
        dx_ = q_dx + phi_term_dx
        return dx_
    
    def dy(self, p: Point3):
        
        q_dy = self.s_vy - self.s_obs_vy
        phi_term_dy = self.v_rel_norm * (self.s[1] - self.cy)/(self.cone_boundary + ZERO_TOL)
        dy_ = q_dy + phi_term_dy
        return dy_

    def dz(self, p: Point3):
        
        q_dz = self.s_vz - self.s_obs_vz
        phi_term_dz = self.v_rel_norm * (self.s[2] - self.cz)/(self.cone_boundary + ZERO_TOL)
        dz_ = q_dz + phi_term_dz
        return dz_

    def dvx(self, p: Point3):
        
        q_dv = (self.s[0] - self.cx) * np.cos(self.s[2]) + (self.s[1] - self.cy) * np.sin(self.s[2])
        phi_term_dv = ( (self.s_vx - self.s_obs_vx)*np.cos(self.s[2]) + (self.s_vy - self.s_obs_vy)*np.sin(self.s[2]) ) * self.cone_boundary/self.v_rel_norm
        dv_ = q_dv + phi_term_dv
        return dv_

    def dvy(self, p: Point3):
        
        q_dv = (self.s[0] - self.cx) * np.cos(self.s[2]) + (self.s[1] - self.cy) * np.sin(self.s[2])
        phi_term_dv = ( (self.s_vx - self.s_obs_vx)*np.cos(self.s[2]) + (self.s_vy - self.s_obs_vy)*np.sin(self.s[2]) ) * self.cone_boundary/self.v_rel_norm
        dv_ = q_dv + phi_term_dv
        return dv_

    def dvz(self, p: Point3):
        
        q_dv = (self.s[0] - self.cx) * np.cos(self.s[2]) + (self.s[1] - self.cy) * np.sin(self.s[2])
        phi_term_dv = ( (self.s_vx - self.s_obs_vx)*np.cos(self.s[2]) + (self.s_vy - self.s_obs_vy)*np.sin(self.s[2]) ) * self.cone_boundary/self.v_rel_norm
        dv_ = q_dv + phi_term_dv
        return dv_
    
    def dphi(self, p: Point3):
        
        q_dtheta = - (self.s[0] - self.cx) * self.s_vy + (self.s[1] - self.cy) * self.s_vx
        phi_term_dtheta = ( -(self.s_vx - self.s_obs_vx)*self.s_vy + (self.s_vy - self.s_obs_vy)*self.s_vx ) * self.cone_boundary/self.v_rel_norm
        dtheta_ = q_dtheta + phi_term_dtheta
        return dtheta_

    def dtheta(self, p: Point3):
        
        q_dtheta = - (self.s[0] - self.cx) * self.s_vy + (self.s[1] - self.cy) * self.s_vx
        phi_term_dtheta = ( -(self.s_vx - self.s_obs_vx)*self.s_vy + (self.s_vy - self.s_obs_vy)*self.s_vx ) * self.cone_boundary/self.v_rel_norm
        dtheta_ = q_dtheta + phi_term_dtheta
        return dtheta_

    def dpsi(self, p: Point3):
        
        q_dtheta = - (self.s[0] - self.cx) * self.s_vy + (self.s[1] - self.cy) * self.s_vx
        phi_term_dtheta = ( -(self.s_vx - self.s_obs_vx)*self.s_vy + (self.s_vy - self.s_obs_vy)*self.s_vx ) * self.cone_boundary/self.v_rel_norm
        dtheta_ = q_dtheta + phi_term_dtheta
        return dtheta_

    def dw1(self, p: Point3):

        q_dx = self.s_vx - self.s_obs_vx
        phi_term_dx = self.v_rel_norm * (self.s[0] - self.cx)/(self.cone_boundary + ZERO_TOL)
        dx_ = q_dx + phi_term_dx
        return dx_

    def dw2(self, p: Point3):

        q_dx = self.s_vx - self.s_obs_vx
        phi_term_dx = self.v_rel_norm * (self.s[0] - self.cx)/(self.cone_boundary + ZERO_TOL)
        dx_ = q_dx + phi_term_dx
        return dx_

    def dw3(self, p: Point3):

        q_dx = self.s_vx - self.s_obs_vx
        phi_term_dx = self.v_rel_norm * (self.s[0] - self.cx)/(self.cone_boundary + ZERO_TOL)
        dx_ = q_dx + phi_term_dx
        return dx_
    
    def dt(self, p: Point3):
        
        q_dt = - (self.s_vx - self.s_obs_vx) * self.s_obs_vx - (self.s_vy - self.s_obs_vy) * self.s_obs_vy
        phi_term_dt = -self.v_rel_norm * ( (self.s[0] - self.cx)*self.s_obs_vx + (self.s[1] - self.cy)*self.s_obs_vy )/(self.cone_boundary + ZERO_TOL)
        dt_ = q_dt + phi_term_dt
        return dt_

    
    def update(self, a: float=None, s: matrix=None, s_obs: matrix=None, buffer: float=None):
        if a is not None:
            self.a = a
        if s is not None:
            self.s = matrix(s)
        if s_obs is not None:
            self.s_obs = matrix(s_obs)
        if buffer is not None:
            if self.BUFFER_FLAG:
                self.a = self.a - self.buffer + buffer
                self.buffer = buffer
            else:
                self.buffer = buffer
        
        self.cx = self.s_obs[0]
        self.cy = self.s_obs[1]
        self.cz = self.s_obs[2]
        self.s_vx = s[3]
        self.s_vy = s[4]
        self.s_vz = s[5]
        self.s_obs_vx = s_obs[3]
        self.s_obs_vy = s_obs[4]
        self.s_obs_vz = s_obs[5]
        self.p_rel = self.s[:3] - self.s_obs[:3]
        self.v_rel = self.s[3:6] - self.s_obs[3:6]
        self.dist = vec_norm(self.p_rel)
        self.v_rel_norm = vec_norm(self.v_rel)
        self.cone_boundary = np.sqrt(self.dist**2 - self.a**2) + ZERO_TOL
        if self.dist > ZERO_TOL:
            self.cos_phi = self.cone_boundary/self.dist
        else:
            self.cos_phi = np.pi/2
    
    def update_state(self, s: matrix, s_obs: matrix):
        self.update(s=s, s_obs=s_obs)
    
    def get_half_angle(self):
        """Returns the apex half angle of the collision cone.
        """
        return np.arccos(self.cos_phi)

    def update_by_bounding_box(self, bbox: BoundingBox):
        """Updates the obstacle state for the collision cone using the
        obstacle's BoundingBox object. Calls the update function after
        making the obstacle state vector. The `a` parameter is taken as
        the diagonal of the box's base/projection on ground plane.
        
        Parameters:
        ----------
            bbox (BoundingBox): The bounding box object associated with the obstacle.

        Raises:
        ------
            TypeError: The argument has to be strictly of type `cbf.obstacles.BoundingBox`
        """
        if not isinstance(bbox, BoundingBox):
            raise TypeError("Expected an object of type cbf.obstacles.BoundingBox as an input to fromBoundingBox() method, but got ", type(bbox).__name__)
            
        self.a = np.hypot(bbox.extent.x, bbox.extent.y, bbox.extent.z)
        s_obs = matrix([bbox.location.x, bbox.location.y, bbox.location.z, bbox.velocity.x, bbox.velocity.y, bbox.velocity.z, bbox.rotation.roll, bbox.rotation.pitch, bbox.rotation.yaw, 0, 0, 0])
        self.update(s_obs=s_obs)
    
    @classmethod
    def from_bounding_box(cls, s: matrix = matrix(0.0, (12,1)), bbox = BoundingBox(), buffer = 0.5) -> CollisionCone3D:
        if not isinstance(bbox, BoundingBox):
            raise TypeError("Expected an object of type cbf.obstacles.BoundingBox as an input to fromBoundingBox() method, but got ", type(bbox).__name__)
        
        a = np.hypot(bbox.extent.x, bbox.extent.y)
        s_obs = matrix([bbox.location.x, bbox.location.y, bbox.location.z, bbox.velocity.x, bbox.velocity.y, bbox.velocity.z, bbox.rotation.roll, bbox.rotation.pitch, bbox.rotation.yaw, 0, 0, 0])
        return cls(a=a, s=s, s_obs=s_obs, buffer=buffer)
        
# class ObstacleList2D(MutableMapping):

#     def __init__(self, data=()):
#         self.mapping = {}
#         self.update(data)
#         self.timestamp = 0.0
    
#     def __getitem__(self, key):
#         return self.mapping[key]
    
#     def __delitem__(self, key):
#         del self.mapping[key]
    
#     def __setitem__(self, key, value):
#         """
#         Contaions the enforced base class check to ensure it contains
#         an object derived from the 3D obstacle base class.
#         """
#         # Enforcing base class check using mro.
#         if not Obstacle3DBase in value.__class__.__mro__:
#             raise TypeError("Expected an object derived from Obstacle3DBase as value. Received " + type(value).__name__)
#         self.mapping[key] = value

#     def __iter__(self):
#         return iter(self.mapping)

#     def __len__(self):
#         return len(self.mapping)
    
#     def __repr__(self):
#         return f"{type(self).__name__}({self.mapping})"
    
#     def set_timestamp(self, timestamp: float):
#         self.timestamp = timestamp

#     def update_by_bounding_box(self, bbox_dict=None, obs_type=Obstacle3DTypes.ELLIPSE3D, buffer=0.5):
#         """
#         Will update the obstacle based on the dynamic obstacle
#         list criteria. Remove the IDs which are not present in
#         the scene and add those which entered the scene. Update
#         the IDs which have changed locations for reformulation 
#         of the contained obstacle objects.
#         """
#         if bbox_dict is not None:
#             for key, bbox in bbox_dict.items():
#                 if key in self.mapping.keys():
#                     self.mapping[key].update_by_bounding_box(bbox)
#                 else:
#                     if obs_type == Obstacle3DTypes.ELLIPSE3D:
#                         self.__setitem__(key, Ellipse3D.from_bounding_box(bbox, buffer))
#                     if obs_type == Obstacle3DTypes.COLLISION_CONE3D:
#                         self.__setitem__(key, CollisionCone3D.from_bounding_box(bbox, buffer))
            
#             rm_keys = []
#             for key in self.mapping.keys():
#                 if key not in bbox_dict.keys():
#                     rm_keys.append(key)
            
#             for key in rm_keys:
#                 self.pop(key)

#     def f(self, p: Point3) -> float:
#         f = matrix(0.0, (len(self.mapping), 1))
#         idx = 0
#         for obs in self.mapping.values():
#             f[idx] = obs.f(p)
#             idx = idx + 1
#         return f

#     def dx(self, p: Point3) -> float:
#         dx = matrix(0.0, (len(self.mapping), 1))
#         idx = 0
#         for obs in self.mapping.values():
#             dx[idx] = obs.dx(p)
#             idx = idx + 1
#         return dx

#     def dy(self, p: Point3) -> float:
#         dy = matrix(0.0, (len(self.mapping), 1))
#         idx = 0
#         for obs in self.mapping.values():
#             dy[idx] = obs.dy(p)
#             idx = idx + 1
#         return dy
    
#     def dtheta(self, p: Point3) -> float:
#         dtheta = matrix(0.0, (len(self.mapping), 1))
#         idx = 0
#         for obs in self.mapping.values():
#             dtheta[idx] = obs.dtheta(p)
#             idx = idx + 1
#         return dtheta
    
#     def dv(self, p: Point3) -> float:
#         dv = matrix(0.0, (len(self.mapping), 1))
#         idx = 0
#         for obs in self.mapping.values():
#             dv[idx] = obs.dv(p)
#             idx = idx + 1
#         return dv
    
#     def dt(self, p: Point3) -> float:
#         dt = matrix(0.0, (len(self.mapping), 1))
#         idx = 0
#         for obs in self.mapping.values():
#             dt[idx] = obs.dt(p)
#             idx = idx + 1
#         return dt

#     def gradient(self, p: Point3) -> float:
#         df = matrix(0.0, (len(self.mapping), 3))
#         idx = 0
#         for obs in self.mapping.values():
#             df[idx,:] = obs.gradient(p).T
#             idx = idx + 1
#         return df

class ObstacleList3D(MutableMapping):

    def __init__(self, data=()):
        self.mapping = {}
        self.update(data)
        self.timestamp = 0.0
    
    def __getitem__(self, key):
        return self.mapping[key]
    
    def __delitem__(self, key):
        del self.mapping[key]
    
    def __setitem__(self, key, value):
        """
        Contaions the enforced base class check to ensure it contains
        an object derived from the 3D obstacle base class.
        """
        # Enforcing base class check using mro.
        if not Obstacle3DBase in value.__class__.__mro__:
            raise TypeError("Expected an object derived from Obstacle3DBase as value. Received " + type(value).__name__)
        self.mapping[key] = value

    def __iter__(self):
        return iter(self.mapping)

    def __len__(self):
        return len(self.mapping)
    
    def __repr__(self):
        return f"{type(self).__name__}({self.mapping})"
    
    def set_timestamp(self, timestamp: float):
        self.timestamp = timestamp

    def update_by_bounding_box(self, bbox_dict=None, obs_type=Obstacle3DTypes.ELLIPSE3D, buffer=0.5):
        """
        Will update the obstacle based on the dynamic obstacle
        list criteria. Remove the IDs which are not present in
        the scene and add those which entered the scene. Update
        the IDs which have changed locations for reformulation 
        of the contained obstacle objects.
        """
        if bbox_dict is not None:
            for key, bbox in bbox_dict.items():
                if key in self.mapping.keys():
                    self.mapping[key].update_by_bounding_box(bbox)
                else:
                    if obs_type == Obstacle3DTypes.ELLIPSE3D:
                        self.__setitem__(key, Ellipse3D.from_bounding_box(bbox, buffer))
                    if obs_type == Obstacle3DTypes.COLLISION_CONE3D:
                        self.__setitem__(key, CollisionCone3D.from_bounding_box(bbox, buffer))
            
            rm_keys = []
            for key in self.mapping.keys():
                if key not in bbox_dict.keys():
                    rm_keys.append(key)
            
            for key in rm_keys:
                self.pop(key)

    def f(self, p: Point3) -> float:
        f = matrix(0.0, (len(self.mapping), 1))
        idx = 0
        for obs in self.mapping.values():
            f[idx] = obs.f(p)
            idx = idx + 1
        return f

    def dx(self, p: Point3) -> float:
        dx = matrix(0.0, (len(self.mapping), 1))
        idx = 0
        for obs in self.mapping.values():
            dx[idx] = obs.dx(p)
            idx = idx + 1
        return dx

    def dy(self, p: Point3) -> float:
        dy = matrix(0.0, (len(self.mapping), 1))
        idx = 0
        for obs in self.mapping.values():
            dy[idx] = obs.dy(p)
            idx = idx + 1
        return dy

    def dz(self, p: Point3) -> float:
        dz = matrix(0.0, (len(self.mapping), 1))
        idx = 0
        for obs in self.mapping.values():
            dz[idx] = obs.dz(p)
            idx = idx + 1
        return dz
    
    def dphi(self, p: Point3) -> float:
        dphi = matrix(0.0, (len(self.mapping), 1))
        idx = 0
        for obs in self.mapping.values():
            dphi[idx] = obs.dphi(p)
            idx = idx + 1
        return dphi

    def dtheta(self, p: Point3) -> float:
        dtheta = matrix(0.0, (len(self.mapping), 1))
        idx = 0
        for obs in self.mapping.values():
            dtheta[idx] = obs.dtheta(p)
            idx = idx + 1
        return dtheta

    def dpsi(self, p: Point3) -> float:
        dpsi = matrix(0.0, (len(self.mapping), 1))
        idx = 0
        for obs in self.mapping.values():
            dpsi[idx] = obs.dpsi(p)
            idx = idx + 1
        return dpsi
    
    def dv(self, p: Point3) -> float:
        dv = matrix(0.0, (len(self.mapping), 1))
        idx = 0
        for obs in self.mapping.values():
            dv[idx] = obs.dv(p)
            idx = idx + 1
        return dv
    
    def dt(self, p: Point3) -> float:
        dt = matrix(0.0, (len(self.mapping), 1))
        idx = 0
        for obs in self.mapping.values():
            dt[idx] = obs.dt(p)
            idx = idx + 1
        return dt

    def gradient(self, p: Point3) -> float:
        df = matrix(0.0, (len(self.mapping), 3))
        idx = 0
        for obs in self.mapping.values():
            df[idx,:] = obs.gradient(p).T
            idx = idx + 1
        return df