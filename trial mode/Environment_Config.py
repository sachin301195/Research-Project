# -*- coding: utf-8 -*-
"""
Created on Fri July 30 21:12:57 2021

@author: busach
"""
"""
	Two filling machines – W1 and W2
	Each with two kinds of pellets (which makes it flexible manufacturing system FMS).
	10 junction points (in conveyor network).
	5 materials namely Red, Black, Blue, Green and Water.
	Rack for “Red” pellet and “Green” pellet are mounted at W1.
	Rack for “Blue” pellet and “Black” pellet are mounted at W2.
	Water is a dependent variable. (To fill the void)
	Possible number of jobs (n) for two machines (m):
	There are total 2^4-1(null set)=15 possible choices (products). (Sequence doesn’t matter)
"""


"""
Configuration of the environment
All the variables are initialized here
"""

jobs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
matrials = ['red', 'black', 'green', 'blue']
work_stations = ['W1', 'W2']
switches = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
source_switch = 1
destination_switch = 2
velocity_of_conveyor = 1 # in m/s
accelaration_of_conveyor = 0 # in m/s**2

