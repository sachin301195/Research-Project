# # -*- coding: utf-8 -*-
# """
# Created on Fri Aug 13 12:25:00 2021

# @author: busach
# """

# import snakes.plugins
# snakes.plugins.load('gv', 'snakes.nets', 'snk')
# from snk import *

# pn = PetriNet("Hello World In Snakes")
# pn.add_place(Place('hello', ['hello', 'salute']))
# pn.add_place(Place('World', ['World', 'le_monade']))
# pn.add_place(Place('Sentence'))
# pn.add_transition(Transition('concat'))
# pn.add_input('hello', 'concat', Variable('h'))
# pn.add_input('World', 'concat', Variable('w'))
# pn.add_output('Sentence', 'concat', Expression("h + '_' + w"))
# pn.draw('hello-1.jpeg')
# modes = pn.transition('concat').modes()
# pn.transition('concat').fire(modes[2])
# pn.draw('hello-2.jpeg')

# from snakes.nets import *
# n = PetriNet('First Net')
# n.add_place(Place('p', [0]))
# n.add_transition(Transition('t', Expression('x<5')))
# n.add_input('p', 't', Variable('x'))
# n.add_output('p', 't', Expression('x+1'))

import snakes.plugins
snakes.plugins.load('gv', 'snakes.nets', 'nets')
from nets import *

def factory (cons, prod, init = [1, 2, 3]):
    n = PetriNet('N')
    n.add_place(Place('src', init))
    n.add_place(Place('tgt', []))
    t = Transition('t')
    n.add_transition(t)
    n.add_input('src', 't', cons)
    n.add_output('tgt', 't', prod)
    return n, t, t.modes()

net, trans, modes = factory(Value(1), Value(0))
net.draw('value-0.png')
print(modes)
trans.fire(modes[0])
net.draw('value-1.png')