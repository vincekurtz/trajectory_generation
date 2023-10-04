#!/usr/bin/env python

##
#
# A toy exmaple for exploring generative models for trajectory planning. A robot
# with double integrator dynamics should go around an obstacle. 
#
##

import numpy as np
from pydrake.all import *
import matplotlib.pyplot as plt

# Problem parameters
obstacle_position = [0.0, 0.0]
obstacle_radius = 1.0
target_position = [0.0, 2.0]
start_position = [0.0, -2.0]
