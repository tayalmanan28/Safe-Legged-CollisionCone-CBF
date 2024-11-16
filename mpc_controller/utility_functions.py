# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 14:50:16 2022

@author: My PC
"""
import math

def point_wrt_circle(x, y, c_x, c_y, r):
    return pow((x - c_x), 2) + pow((y - c_y), 2) - pow(r,2)

def norm2(x, y):
    return math.sqrt(pow(x,2) + pow(y,2))

def norm(x, y, z):
    return math.sqrt(x**2 + y**2 + z**2)


def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy

def matrix_multiplication(A, B):
    result = [[sum(a * b for a, b in zip(A_row, B_col))
                        for B_col in zip(*B)]
                                for A_row in A]
    return result
