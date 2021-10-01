# # -*- coding: utf-8 -*-
# """
# Created on Thu Aug 19 15:46:26 2021
#
# @author: busach
# """
# import sys
# import snakes
# import snakes.plugins
# import tpn
# snakes.plugins.load([tpn, 'gv'], 'snakes.nets', 'nets')
# from nets import *
# pn = PetriNet('TRY')
# pn.globals['m'] = 10
# init = [1, 2, 3]
# pn.add_place(Place('A', init))
# pn.add_transition(Transition('t1', Expression('m == 10'), max_time = 2, min_time = 0))
# pn.add_transition(Transition('t2', Expression('m == 10'), min_time = 2))
# pn.add_transition(Transition('t3', Expression('m == 10')))
# pn.add_place(Place('B', []))
# pn.add_place(Place('C', []))
# pn.add_input('A', 't1', Variable('x'))
# pn.add_input('A', 't2', Variable('x'))
# pn.add_input('B', 't3', Variable('x'))
# pn.add_output('C', 't3', Variable('x'))
# pn.add_output('B', 't1', Variable('x'))
# pn.add_output('B', 't2', Variable('x'))
# pn.add_output('C', 't2', Variable('x'))
# pn.draw('try_in.png')
# g = StateGraph(pn)
#
# pn.reset()
# print(g.current())
# clock = 0.0
# delay = pn.time()
# modes = pn.transition('t1').modes()
# clock += delay
# pn.transition('t1').fire(modes[0])
# print(g.current())
# # pn.transition('t2').fire(modes[1])
# # # pn.transition('t3').fire(Substitution(x = 1))
# # pn.draw('try.png')
#
# # pn.add_place(Place('p', [0]))
# # pn.add_transition(Transition('t', Expression('x<5')))
# # pn.add_input('p', 't', Variable('x'))
# # pn.add_output('p', 't', Expression('x+1'))
# # g = StateGraph(pn)
# # try: g.goto(2)
# # except ValueError: print(sys.exc_info()[1])
# # g.build()
# # g.goto(2)
# # g.net.get_marking()

from snakes.nets import *

n = PetriNet('N')
n.add_place(Place('p', [0]))
n.add_transition(Transition('t', Expression('x<5')))
n.add_input('p', 't', Variable('x'))
n.add_output('p', 't', Expression('x+1'))
g = StateGraph(n)
print(g.current())
g.build()
g.goto(2)
print(g.current())
print(list(g.successors()))
print(list(g.predecessors()))