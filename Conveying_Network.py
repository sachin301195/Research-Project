# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 22:18:48 2021

@author: busach
"""
# importing the required libraries
# import graphviz
# import argparse
# import matplotlib as plt

# import time
import gym
import tpn
import snakes.plugins

snakes.plugins.load([tpn, 'gv', 'bound'], 'snakes.nets', 'snk')
from snk import *

plcs = list("ABCDEFGHIJKLMNOPSTXY")
plcs.extend(['Green', 'Red', 'Blue', 'Violet'])  # resources

trns = ['t_AB', 't_BA', 't_SA', 't_SB', 't_AE', 't_EA', 't_AC', 't_CA', 't_CE', 't_EC',
        't_EF', 't_FE', 't_EG', 't_GE', 't_FG', 't_GF', 't_HD', 't_DH', 't_HF', 't_FH',
        't_DF', 't_FD', 't_HL', 't_LH', 't_LP', 't_PL', 't_HP', 't_PH', 't_GI', 't_IG',
        't_IJ', 't_JI', 't_GJ', 't_JG', 't_IP', 't_PI', 't_PN', 't_NP', 't_IN', 't_NI',
        't_ON', 't_NO', 't_NM', 't_MN', 't_OM', 't_MO', 't_KJ', 't_JK', 't_JB', 't_BJ',
        't_KB', 't_BK', 't_OT', 't_OK', 't_KO', 't_KT', 't_XD', 't_DX', 't_RXD', 't_RDX',
        't_XC', 't_CX', 't_GXC', 't_GCX', 't_YL', 't_LY', 't_BYL', 't_BLY', 't_YM', 't_MY',
        't_VYM', 't_VMY', 't_T']


class env(gym.GoalEnv):
    """
    Parameters
    ----------

    DESCRIPTION.
    ------------
    Generates the Conveyor Network Petri Net.

    Returns
    -------
    Network

    """

    def __init__(self, jobs, resources, orders):
        self.fs = jobs
        self.cs = [0 for _ in jobs]
        self.no_of_jobs = len(jobs)
        self.res = resources[0]
        self.red = resources[1]
        self.green = resources[2]
        self.blue = resources[3]
        self.violet = resources[4]
        self.orders = orders
        if self.no_of_jobs != len(self.orders):
            raise ValueError("length of number of jobs should be same as length of order")
        if sum(self.orders) != self.res:
            raise ValueError("Total sum of orders should be equal to the resource[0]")

    def tokens(self):
        self.token = {}
        for i in self.fs:
            self.token[f'job{i}'] = (0, i, 0)
            # exec(f't_job{self.fs[i]} = [self.cs[i], self.fs[i]]')
        self.t_init = []
        for i in range(self.no_of_jobs):
            for k in range(self.orders[i]):
                self.t_init.append(self.token[f'job{self.fs[i]}'])
        self.t_red = [1 for _ in range(self.red)]
        self.t_green = [2 for _ in range(self.green)]
        self.t_blue = [4 for _ in range(self.blue)]
        self.t_violet = [8 for _ in range(self.violet)]
        return self.t_init, self.t_red, self.t_green, self.t_blue, self.t_violet

    def network(self, bounds=5, minimum_time=0, maximum_time=5):
        # developing the network
        n = PetriNet('Network')
        # n.globals['source', 'destination'] = [kwargs['source'], kwargs['destination']]
        self.init, self.r, self.g, self.b, self.v = self.tokens()

        # Adding places
        for i in plcs:
            if i == 'S':
                n.add_place(Place('%s' % i, self.init, bound=(0, None)))
            elif i == 'Green':
                n.add_place(Place('%s' % i, self.g, bound=(0, None)))
            elif i == 'Red':
                n.add_place(Place('%s' % i, self.r, bound=(0, None)))
            elif i == 'Blue':
                n.add_place(Place('%s' % i, self.b, bound=(0, None)))
            elif i == 'Violet':
                n.add_place(Place('%s' % i, self.v, bound=(0, None)))
            else:
                n.add_place(Place('%s' % i, [], bound=bounds))

        # Adding Transitions
        trans = {}
        for i in trns:
            if i == 't_T':
                trans.update({i: Transition('%s' % i, Expression('c == f'), min_time=minimum_time + int(n.time() or 0),
                                            max_time=maximum_time + int(n.time() or 0))})
            else:
                trans.update({i: Transition('%s' % i, min_time=minimum_time + int(n.time() or 0),
                                            max_time=maximum_time + int(n.time() or 0))})
            n.add_transition(trans[i])

        # Adding input and output Arcs
        # source to A and B
        n.add_input('S', 't_SA', Tuple([Variable('c'), Variable('f'), Variable('count')]))
        n.add_input('S', 't_SB', Tuple([Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('A', 't_SA', Tuple([Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_output('B', 't_SB', Tuple([Variable('c'), Variable('f'), Expression('count + 1')]))

        # A to B and B to A
        n.add_input('A', 't_AB', Tuple([Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('B', 't_AB', Tuple([Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('B', 't_BA', Tuple([Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('A', 't_BA', Tuple([Variable('c'), Variable('f'), Expression('count + 1')]))

        # A to C and C to A
        n.add_input('A', 't_AC', Tuple([Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('C', 't_AC', Tuple([Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('C', 't_CA', Tuple([Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('A', 't_CA', Tuple([Variable('c'), Variable('f'), Expression('count + 1')]))

        # A to E and E to A
        n.add_input('A', 't_AE', Tuple([Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('E', 't_AE', Tuple([Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('E', 't_EA', Tuple([Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('A', 't_EA', Tuple([Variable('c'), Variable('f'), Expression('count + 1')]))

        # C to E and E to C
        n.add_input('E', 't_EC', Tuple([Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('C', 't_EC', Tuple([Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('C', 't_CE', Tuple([Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('E', 't_CE', Tuple([Variable('c'), Variable('f'), Expression('count + 1')]))

        # B to K and K to B
        n.add_input('K', 't_KB', Tuple([Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('B', 't_KB', Tuple([Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('B', 't_BK', Tuple([Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('K', 't_BK', Tuple([Variable('c'), Variable('f'), Expression('count + 1')]))

        # B to J and J to B
        n.add_input('J', 't_JB', Tuple([Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('B', 't_JB', Tuple([Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('B', 't_BJ', Tuple([Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('J', 't_BJ', Tuple([Variable('c'), Variable('f'), Expression('count + 1')]))

        # J to K and K to J
        n.add_input('J', 't_JK', Tuple([Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('K', 't_JK', Tuple([Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('K', 't_KJ', Tuple([Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('J', 't_KJ', Tuple([Variable('c'), Variable('f'), Expression('count + 1')]))

        # E to G and G to E
        n.add_input('G', 't_GE', Tuple([Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('E', 't_GE', Tuple([Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('E', 't_EG', Tuple([Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('G', 't_EG', Tuple([Variable('c'), Variable('f'), Expression('count + 1')]))

        # E to F and F to E
        n.add_input('F', 't_FE', Tuple([Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('E', 't_FE', Tuple([Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('E', 't_EF', Tuple([Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('F', 't_EF', Tuple([Variable('c'), Variable('f'), Expression('count + 1')]))

        # F to G and G to F
        n.add_input('G', 't_GF', Tuple([Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('F', 't_GF', Tuple([Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('F', 't_FG', Tuple([Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('G', 't_FG', Tuple([Variable('c'), Variable('f'), Expression('count + 1')]))

        # G to I and I to G
        n.add_input('G', 't_GI', Tuple([Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('I', 't_GI', Tuple([Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('I', 't_IG', Tuple([Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('G', 't_IG', Tuple([Variable('c'), Variable('f'), Expression('count + 1')]))

        # G to J and J to G
        n.add_input('G', 't_GJ', Tuple([Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('J', 't_GJ', Tuple([Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('J', 't_JG', Tuple([Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('G', 't_JG', Tuple([Variable('c'), Variable('f'), Expression('count + 1')]))

        # I to J and J to I
        n.add_input('I', 't_IJ', Tuple([Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('J', 't_IJ', Tuple([Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('J', 't_JI', Tuple([Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('I', 't_JI', Tuple([Variable('c'), Variable('f'), Expression('count + 1')]))

        # I to P and P to I
        n.add_input('I', 't_IP', Tuple([Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('P', 't_IP', Tuple([Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('P', 't_PI', Tuple([Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('I', 't_PI', Tuple([Variable('c'), Variable('f'), Expression('count + 1')]))

        # I to N and N to I
        n.add_input('I', 't_IN', Tuple([Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('N', 't_IN', Tuple([Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('N', 't_NI', Tuple([Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('I', 't_NI', Tuple([Variable('c'), Variable('f'), Expression('count + 1')]))

        # P to N and N to P
        n.add_input('N', 't_NP', Tuple([Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('P', 't_NP', Tuple([Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('P', 't_PN', Tuple([Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('N', 't_PN', Tuple([Variable('c'), Variable('f'), Expression('count + 1')]))

        # P to L and L to P
        n.add_input('L', 't_LP', Tuple([Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('P', 't_LP', Tuple([Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('P', 't_PL', Tuple([Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('L', 't_PL', Tuple([Variable('c'), Variable('f'), Expression('count + 1')]))

        # P to H and H to P
        n.add_input('H', 't_HP', Tuple([Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('P', 't_HP', Tuple([Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('P', 't_PH', Tuple([Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('H', 't_PH', Tuple([Variable('c'), Variable('f'), Expression('count + 1')]))

        # L to H and H to L
        n.add_input('H', 't_HL', Tuple([Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('L', 't_HL', Tuple([Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('L', 't_LH', Tuple([Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('H', 't_LH', Tuple([Variable('c'), Variable('f'), Expression('count + 1')]))

        # N to M and M to N
        n.add_input('N', 't_NM', Tuple([Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('M', 't_NM', Tuple([Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('M', 't_MN', Tuple([Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('N', 't_MN', Tuple([Variable('c'), Variable('f'), Expression('count + 1')]))

        # N to O and O to N
        n.add_input('N', 't_NO', Tuple([Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('O', 't_NO', Tuple([Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('O', 't_ON', Tuple([Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('N', 't_ON', Tuple([Variable('c'), Variable('f'), Expression('count + 1')]))

        # M to O and O to M
        n.add_input('O', 't_OM', Tuple([Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('M', 't_OM', Tuple([Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('M', 't_MO', Tuple([Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('O', 't_MO', Tuple([Variable('c'), Variable('f'), Expression('count + 1')]))

        # O to K and K to O
        n.add_input('O', 't_OK', Tuple([Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('K', 't_OK', Tuple([Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('K', 't_KO', Tuple([Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('O', 't_KO', Tuple([Variable('c'), Variable('f'), Expression('count + 1')]))

        # O to T and K to T
        n.add_input('O', 't_OT', Tuple([Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('T', 't_OT', Tuple([Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('K', 't_KT', Tuple([Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('T', 't_KT', Tuple([Variable('c'), Variable('f'), Expression('count + 1')]))

        # C to X and X to C with Green
        n.add_input('C', 't_CX', Tuple([Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('X', 't_CX', Tuple([Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('X', 't_XC', Tuple([Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('C', 't_XC', Tuple([Variable('c'), Variable('f'), Variable('count')]))
        n.add_input('C', 't_GCX', Tuple([Variable('c'), Variable('f'), Variable('count')]))
        n.add_input('Green', 't_GCX', Variable('x'))
        n.add_output('X', 't_GCX', Tuple([Expression('x + c'), Variable('f'), Expression('count + 1')]))
        n.add_input('X', 't_GXC', Tuple([Variable('c'), Variable('f'), Variable('count')]))
        n.add_input('Green', 't_GXC', Variable('x'))
        n.add_output('C', 't_GXC', Tuple([Expression('x + c'), Variable('f'), Variable('count')]))

        # D to X and X to D with Red
        n.add_input('D', 't_DX', Tuple([Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('X', 't_DX', Tuple([Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('X', 't_XD', Tuple([Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('D', 't_XD', Tuple([Variable('c'), Variable('f'), Variable('count')]))
        n.add_input('D', 't_RDX', Tuple([Variable('c'), Variable('f'), Variable('count')]))
        n.add_input('Red', 't_RDX', Variable('x'))
        n.add_output('X', 't_RDX', Tuple([Expression('x + c'), Variable('f'), Expression('count + 1')]))
        n.add_input('X', 't_RXD', Tuple([Variable('c'), Variable('f'), Variable('count')]))
        n.add_input('Red', 't_RXD', Variable('x'))
        n.add_output('D', 't_RXD', Tuple([Expression('x + c'), Variable('f'), Variable('count')]))

        # L to Y and Y to L with Blue
        n.add_input('L', 't_LY', Tuple([Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('Y', 't_LY', Tuple([Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('Y', 't_YL', Tuple([Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('L', 't_YL', Tuple([Variable('c'), Variable('f'), Variable('count')]))
        n.add_input('L', 't_BLY', Tuple([Variable('c'), Variable('f'), Variable('count')]))
        n.add_input('Blue', 't_BLY', Variable('x'))
        n.add_output('Y', 't_BLY', Tuple([Expression('x + c'), Variable('f'), Expression('count + 1')]))
        n.add_input('Y', 't_BYL', Tuple([Variable('c'), Variable('f'), Variable('count')]))
        n.add_input('Blue', 't_BYL', Variable('x'))
        n.add_output('L', 't_BYL', Tuple([Expression('x + c'), Variable('f'), Variable('count')]))

        # M to Y and Y to M with Violet
        n.add_input('M', 't_MY', Tuple([Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('Y', 't_MY', Tuple([Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('Y', 't_YM', Tuple([Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('M', 't_YM', Tuple([Variable('c'), Variable('f'), Variable('count')]))
        n.add_input('M', 't_VMY', Tuple([Variable('c'), Variable('f'), Variable('count')]))
        n.add_input('Violet', 't_VMY', Variable('x'))
        n.add_output('Y', 't_VMY', Tuple([Expression('x + c'), Variable('f'), Expression('count + 1')]))
        n.add_input('Y', 't_VYM', Tuple([Variable('c'), Variable('f'), Variable('count')]))
        n.add_input('Violet', 't_VYM', Variable('x'))
        n.add_output('M', 't_VYM', Tuple([Expression('x + c'), Variable('f'), Variable('count')]))

        # D to H and H to D
        n.add_input('H', 't_HD', Tuple([Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('D', 't_HD', Tuple([Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('D', 't_DH', Tuple([Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('H', 't_DH', Tuple([Variable('c'), Variable('f'), Expression('count + 1')]))

        # H to F and F to H
        n.add_input('H', 't_HF', Tuple([Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('F', 't_HF', Tuple([Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('F', 't_FH', Tuple([Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('H', 't_FH', Tuple([Variable('c'), Variable('f'), Expression('count + 1')]))

        # D to F and F to D
        n.add_input('D', 't_DF', Tuple([Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('F', 't_DF', Tuple([Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('F', 't_FD', Tuple([Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('D', 't_FD', Tuple([Variable('c'), Variable('f'), Expression('count + 1')]))

        # Termination
        n.add_input('T', 't_T', Flush('x'))

        return n, trans


environment = env([1], [12, 5, 5, 5, 5], [12])
net, transitions = environment.network()
net.draw('network.png')
print(net.get_marking())
print(net.node('S'))
