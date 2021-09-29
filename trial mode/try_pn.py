# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 15:46:26 2021

@author: busach
"""

import snakes.plugins
import tpn
snakes.plugins.load([tpn, 'gv'], 'snakes.nets', 'nets')
from nets import *
pn = PetriNet('TRY')
pn.globals['m'] = 10
init = [1, 2, 3]
pn.add_place(Place('A', init))
pn.add_transition(Transition('t1', Expression('m == 10'), max_time = 2, min_time = 0))
pn.add_transition(Transition('t2', Expression('m == 10'), min_time = 2))
pn.add_transition(Transition('t3', Expression('m == 10')))
pn.add_place(Place('B', []))
pn.add_place(Place('C', []))
pn.add_input('A', 't1', Variable('x'))
pn.add_input('A', 't2', Variable('x'))
pn.add_input('B', 't3', Variable('x'))
pn.add_output('C', 't3', Variable('x'))
pn.add_output('B', 't1', Variable('x'))
pn.add_output('B', 't2', Variable('x'))
pn.add_output('C', 't2', Variable('x'))
pn.draw('try_in.png')
pn.reset()
clock = 0.0
delay = pn.time()
modes = pn.transition('t1').modes()
clock += delay
pn.transition('t1').fire(modes[0])
pn.transition('t2').fire(modes[1])
# pn.transition('t3').fire(Substitution(x = 1))
pn.draw('try.png')
