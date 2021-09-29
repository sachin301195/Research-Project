# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 16:39:40 2021

@author: busach
"""

import tpn
import snakes.plugins
snakes.plugins.load(tpn, "snakes.nets", "snk")

from snk import *

n = PetriNet("stepper")
for i in range(3) :
    n.add_place(Place("p%s" % i, [dot]))
    n.add_transition(Transition("t%s" % i, min_time=i+1, max_time=i*2+1))
    n.add_input("p%s" % i, "t%s" % i, Value(dot))
init = n.get_marking()

n.reset()
clock = 0.0
for i in range(3) :
    print(" , ".join("%s[%s,%s]=%s"
                     % (t, t.min_time, t.max_time,
                        "#" if t.time is None else t.time)
                     for t in sorted(n.transition())))
    delay = n.time()
    print("[%s]" % clock, "delay:", delay)
    clock += delay
    print("[%s] fire: t%s" % (clock, i))
    n.transition("t%s" % i).fire(Substitution())
    