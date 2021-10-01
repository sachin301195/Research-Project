import sys
sys.path.append('C:\source\Research Project\Scripts\Research-Project')
import tpn
import snakes.plugins

snakes.plugins.load([tpn, 'gv', 'bound'], 'snakes.nets', 'snk')
from snk import *

plcs = list('ACDEFGJKSTX')
plcs.extend(['Green', 'Red']) #resources

trns = ['t_SA', 't_AC', 't_AE', 't_EF', 't_DF', 't_FD', 't_EG', 't_FG', \
        't_GJ', 't_JK', 't_KT', 't_T', 't_CX', 't_XD', 't_GCX', 't_RXD', \
        't_DX', 't_XC', 't_RDX', 't_GXC']

class TrialNet:
    def __init__(self, jobs, resources, orders):
        self.fs = jobs
        self.cs = [0 for _ in jobs]
        self.no_of_jobs = len(jobs)
        self.res = resources[0]
        self.red = resources[1]
        self.green = resources[2]
        self.orders = orders
        if self.no_of_jobs != len(self.orders):
            raise ValueError("length of number of jobs should be same as length of order")
        if sum(self.orders) != self.res:
            raise ValueError("Total sum of orders should be equal to the resource[0]")
            
    def tokens(self):
        self.token = {}
        for i in self.fs:
            self.token[f'job{i}'] = (0, i, 0)
        self.t_init = []
        for i in range(self.no_of_jobs):
            for k in range(self.orders[i]):
                self.t_init.append(self.token[f'job{self.fs[i]}'])
        self.t_red = [1 for _ in range(self.red)]
        self.t_green = [2 for _ in range(self.green)]
        return self.t_init, self.t_red, self.t_green
    
    def network(self, bounds = 5, minimum_time = 0, maximum_time = 5):
        n = PetriNet('trial net')
        self.init, self.r, self.g = self.tokens()
        # adding places
        for i in plcs:
            if i == 'S':
                n.add_place(Place('%s' % i, self.init, bound=(0, None)))
            elif i == 'Green':
                n.add_place(Place('%s' % i, self.g, bound=(0, None)))
            elif i == 'Red':
                n.add_place(Place('%s' % i, self.r, bound=(0, None)))
            else:
                n.add_place(Place('%s' % i, [], bound=bounds))
        
        # adding transitions
        trans = {}
        for i in trns:
            if i == 't_T':
                trans.update({i: Transition('%s' % i, Expression('c == f'), min_time=minimum_time + int(n.time() or 0),
                                            max_time=maximum_time + int(n.time() or 0))})
            else:
                trans.update({i: Transition('%s' % i, min_time=minimum_time + int(n.time() or 0),
                                            max_time=maximum_time + int(n.time() or 0))})
            n.add_transition(trans[i])
            
        # adding input and output arcs
        n.add_input('S', 't_SA', Tuple([Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('A', 't_SA', Tuple([Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('A', 't_AC', Tuple([Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('C', 't_AC', Tuple([Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('A', 't_AE', Tuple([Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('E', 't_AE', Tuple([Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('E', 't_EF', Tuple([Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('F', 't_EF', Tuple([Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('D', 't_DF', Tuple([Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('F', 't_DF', Tuple([Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('F', 't_FD', Tuple([Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('D', 't_FD', Tuple([Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('F', 't_FG', Tuple([Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('G', 't_FG', Tuple([Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('E', 't_EG', Tuple([Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('G', 't_EG', Tuple([Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('G', 't_GJ', Tuple([Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('J', 't_GJ', Tuple([Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('J', 't_JK', Tuple([Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('K', 't_JK', Tuple([Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('T', 't_T', Flush('x'))
        n.add_input('K', 't_KT', Tuple([Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('T', 't_KT', Tuple([Variable('c'), Variable('f'), Expression('count + 1')]))
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
        
        return n, trans

trial_net = TrialNet([1], [7, 5, 5], [7])
net, t = trial_net.network()
net.draw('trialnet.png')
net.reset()
clock = 0.0
delay = net.time()
g = StateGraph(net)
print(g.current())
mode = t['t_SA'].modes()
print(mode)
clock += delay
t['t_SA'].fire(mode[0])

net.draw('first_fire.png')
print(g.current())
# g.build()
# print(list(g.successors))

    
    