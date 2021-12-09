import sys
sys.path.append('snakes-master')
import snakes.plugins
from snakes.utils.simul import StateSpace
from snakes.nets import *
snakes.plugins.load(['gv', 'bound'], 'snakes.nets', 'snk')
from snk import *

places = ['S', 'S1', 'N1', 'A1', 'A2', 'A3', 'B1', 'B2', 'B3', 'N2', 'C1', 'C2', 'C3', 'D1', 'D2', 'D3',
          'E1', 'E2', 'E3', 'F1', 'F2', 'F3', 'J1', 'J2', 'J3', 'G1', 'G2', 'G3', 'K1', 'K2', 'K3', 'T1',
          'W1', 'Red', 'Green', 'N3', 'N4', 'N0', 'N6', 'N9']

trans = ['s1', 'SN1', 'AN1', 'BN1', 'P_A1', 'P_A2', 'P_A3', 'N_A1', 'N_A2', 'N_A3', 'P_B1', 'P_B2', 'P_B3',
         'N_B1', 'N_B2', 'N_B3', 'BN9', 'JN9', 'KN9', 'P_K1', 'P_K2', 'P_K3', 'N_K1', 'N_K2', 'N_K3', 'KN0',
         't1', 'T', 'P_J1', 'P_J2', 'P_J3', 'N_J1', 'N_J2', 'N_J3', 'JN6', 'GN6', 'P_G1', 'P_G2', 'P_G3',
         'N_G1', 'N_G2', 'N_G3', 'GN3', 'FN3', 'EN3', 'P_E1', 'P_E2', 'P_E3', 'N_E1', 'N_E2', 'N_E3', 'AN2',
         'EN2', 'CN2', 'P_C1', 'P_C2', 'P_C3', 'N_C1', 'N_C2', 'N_C3', 'C0W1', 'C1W1', 'C2W1', 'C3W1', 'D0W1',
         'D1W1', 'D2W1', 'D3W1', 'P_D1', 'P_D2', 'P_D3', 'N_D1', 'N_D2', 'N_D3', 'DN4', 'FN4', 'P_F1', 'P_F2',
         'P_F3', 'N_F1', 'N_F2', 'N_F3']


class TrialConveyorNetwork:
    def __init__(self, jobs, resources, quantity):
        self.fs = jobs
        self.cs = [0 for _ in jobs]
        self.no_of_jobs = len(jobs)
        self.res = resources[0]
        self.red = resources[1]
        self.green = resources[2]
        self.orders = quantity
        self.token = {}
        self.t_init = []
        self.t_red = [1 for _ in range(self.red)]
        self.t_green = [2 for _ in range(self.green)]
        self.transition = {}
        self.object_no = 0
        if self.no_of_jobs != len(self.orders):
            raise ValueError("length of number of jobs should be same as length of order")
        if sum(self.orders) != self.res:
            raise ValueError("Total sum of orders should be equal to the resource[0]")

    def tokens(self):
        for i in self.fs:
            self.token[f'job{i}'] = (0, 0, 0, i, 0)
        for i in range(self.no_of_jobs):
            for k in range(self.orders[i]):
                self.token[f'job{self.fs[i]}'] = self.token[f'job{self.fs[i]}'][:1] + (self.object_no, ) + \
                                                    self.token[f'job{self.fs[i]}'][2:]
                self.object_no += 1
                self.t_init.append(self.token[f'job{self.fs[i]}'])
        return self.t_init, self.t_red, self.t_green

    def trial_conveyor_petrinet(self, bounds = 1):
        n = PetriNet('trial_net')
        n.globals.declare('c = 0')
        n.globals.declare('f = 0')
        self.init, self.r, self.g = self.tokens()

        # adding places
        for i in places:
            if i == 'S':
                n.add_place(Place('%s' % i, self.init, bound=(0, None)))
            elif i == 'Green':
                n.add_place(Place('%s' % i, self.g, bound=(0, None)))
            elif i == 'Red':
                n.add_place(Place('%s' % i, self.r, bound=(0, None)))
            else:
                n.add_place(Place('%s' % i, [], bound=bounds))

        # adding transitions
        for i in trans:
            if i == 'T':
                self.transition.update({i : Transition('%s' % i, Expression('c == f'))})
            else:
                self.transition.update({i : Transition('%s' % i)})
            n.add_transition(self.transition[i])

        # adding input and output
        # Conveyor S
        n.add_input('S', 's1', Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('S1', 's1', Tuple([Value(1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('S1', 'SN1', Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('N1', 'SN1', Tuple([Value(1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))

        # Conveyor A
        n.add_input('N1', 'P_A1', Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('A1', 'P_A1', Tuple([Value(1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('A1', 'P_A2', Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('A2', 'P_A2', Tuple([Value(1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('A2', 'P_A3', Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('A3', 'P_A3', Tuple([Value(1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('A3', 'AN2', Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('N2', 'AN2', Tuple([Value(1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('N2', 'N_A3', Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('A3', 'N_A3', Tuple([Value(-1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('A3', 'N_A2', Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('A2', 'N_A2', Tuple([Value(-1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('A2', 'N_A1', Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('A1', 'N_A1', Tuple([Value(-1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('A1', 'AN1', Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('N1', 'AN1', Tuple([Value(-1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))

        # Conveyor B
        n.add_input('N1', 'N_B3', Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('B3', 'N_B3', Tuple([Value(-1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('B3', 'N_B2', Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('B2', 'N_B2', Tuple([Value(-1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('B2', 'N_B1', Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('B1', 'N_B1', Tuple([Value(-1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('B1', 'BN9', Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('N9', 'BN9', Tuple([Value(-1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('N9', 'P_B1', Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('B1', 'P_B1', Tuple([Value(1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('B1', 'P_B2', Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('B2', 'P_B2', Tuple([Value(1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('B2', 'P_B3', Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('B3', 'P_B3', Tuple([Value(1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('B3', 'BN1', Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('N1', 'BN1', Tuple([Value(1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))

        # Conveyor E
        n.add_input('N2', 'N_E3', Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('E3', 'N_E3', Tuple([Value(-1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('E3', 'N_E2', Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('E2', 'N_E2', Tuple([Value(-1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('E2', 'N_E1', Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('E1', 'N_E1', Tuple([Value(-1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('E1', 'EN3', Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('N3', 'EN3', Tuple([Value(-1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('N3', 'P_E1', Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('E1', 'P_E1', Tuple([Value(1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('E1', 'P_E2', Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('E2', 'P_E2', Tuple([Value(1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('E2', 'P_E3', Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('E3', 'P_E3', Tuple([Value(1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('E3', 'EN2', Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('N2', 'EN2', Tuple([Value(1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))

        # Conveyor F
        n.add_input('N3', 'P_F1', Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('F1', 'P_F1', Tuple([Value(1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('F1', 'P_F2', Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('F2', 'P_F2', Tuple([Value(1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('F2', 'P_F3', Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('F3', 'P_F3', Tuple([Value(1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('F3', 'FN4', Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('N4', 'FN4', Tuple([Value(1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('N4', 'N_F3', Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('F3', 'N_F3', Tuple([Value(-1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('F3', 'N_F2', Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('F2', 'N_F2', Tuple([Value(-1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('F2', 'N_F1', Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('F1', 'N_F1', Tuple([Value(-1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('F1', 'FN3', Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('N3', 'FN3', Tuple([Value(-1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))

        # Conveyor G
        n.add_input('N3', 'N_G3', Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('G3', 'N_G3', Tuple([Value(-1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('G3', 'N_G2', Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('G2', 'N_G2', Tuple([Value(-1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('G2', 'N_G1', Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('G1', 'N_G1', Tuple([Value(-1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('G1', 'GN6', Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('N6', 'GN6', Tuple([Value(-1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('N6', 'P_G1', Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('G1', 'P_G1', Tuple([Value(1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('G1', 'P_G2', Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('G2', 'P_G2', Tuple([Value(1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('G2', 'P_G3', Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('G3', 'P_G3', Tuple([Value(1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('G3', 'GN3', Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('N3', 'GN3', Tuple([Value(1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))

        # Conveyor J
        n.add_input('N6', 'N_J3', Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('J3', 'N_J3', Tuple([Value(-1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('J3', 'N_J2', Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('J2', 'N_J2', Tuple([Value(-1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('J2', 'N_J1', Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('J1', 'N_J1', Tuple([Value(-1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('J1', 'JN9', Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('N9', 'JN9', Tuple([Value(-1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('N9', 'P_J1', Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('J1', 'P_J1', Tuple([Value(1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('J1', 'P_J1', Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('J2', 'P_J2', Tuple([Value(1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('J2', 'P_J3', Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('J3', 'P_J3', Tuple([Value(1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('J3', 'JN6', Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('N6', 'JN6', Tuple([Value(1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))

        # Conveyor K
        n.add_input('N9', 'N_K3', Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('K3', 'N_K3', Tuple([Value(-1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('K3', 'N_K2', Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('K2', 'N_K2', Tuple([Value(-1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('K2', 'N_K1', Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('K1', 'N_K1', Tuple([Value(-1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('K1', 'KN0', Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('N0', 'KN0', Tuple([Value(-1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('N0', 'P_K1', Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('K1', 'P_K1', Tuple([Value(1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('K1', 'P_K2', Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('K2', 'P_K2', Tuple([Value(1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('K2', 'P_K3', Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('K3', 'P_K3', Tuple([Value(1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('K3', 'KN9', Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('N9', 'KN9', Tuple([Value(1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))

        # Conveyor C & D
        n.add_input('N2', 'P_C1', Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('C1', 'P_C1', Tuple([Value(1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('C1', 'P_C2', Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('C2', 'P_C2', Tuple([Value(1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('C2', 'P_C3', Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('C3', 'P_C3', Tuple([Value(1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        # C to W1
        n.add_input('C3', 'C0W1', Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('W1', 'C0W1', Tuple([Value(-1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('C3', 'C1W1', Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_input('Red', 'C1W1', Variable('x'))
        n.add_output('W1', 'C1W1', Tuple([Value(-1), Expression('c + x'), Variable('f'), Expression('count + 1')]))
        n.add_input('C3', 'C2W1', Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_input('Green', 'C2W1', Variable('y'))
        n.add_output('W1', 'C2W1', Tuple([Value(-1), Expression('c + y'), Variable('f'), Expression('count + 1')]))
        n.add_input('C3', 'C3W1', Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_input('Red', 'C3W1', Variable('x'))
        n.add_input('Green', 'C3W1', Variable('y'))
        n.add_output('W1', 'C3W1', Tuple([Value(-1), Expression('c + x + y'), Variable('f'), Expression('count + 1')]))
        # W1 to D
        n.add_input('W1', 'N_D3', Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('D3', 'N_D3', Tuple([Value(-1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('D3', 'N_D2', Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('D2', 'N_D2', Tuple([Value(-1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('D2', 'N_D1', Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('D1', 'N_D1', Tuple([Value(-1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('D1', 'DN4', Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('N4', 'DN4', Tuple([Value(-1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('N4', 'P_D1', Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('D1', 'P_D1', Tuple([Value(1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('D1', 'P_D2', Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('D2', 'P_D2', Tuple([Value(1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('D2', 'P_D3', Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('D3', 'P_D3', Tuple([Value(1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        # D to W1
        n.add_input('D3', 'D0W1', Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('W1', 'D0W1', Tuple([Value(1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('D3', 'D1W1', Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_input('Red', 'D1W1', Variable('x'))
        n.add_output('W1', 'D1W1', Tuple([Value(1), Expression('c + x'), Variable('f'), Expression('count + 1')]))
        n.add_input('D3', 'D2W1', Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_input('Green', 'D2W1', Variable('y'))
        n.add_output('W1', 'D2W1', Tuple([Value(1), Expression('c + y'), Variable('f'), Expression('count + 1')]))
        n.add_input('D3', 'D3W1', Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_input('Red', 'D3W1', Variable('x'))
        n.add_input('Green', 'D3W1', Variable('y'))
        n.add_output('W1', 'D3W1', Tuple([Value(1), Expression('c + x + y'), Variable('f'), Expression('count + 1')]))
        # W1 to C
        n.add_input('W1', 'N_C3', Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('C3', 'N_C3', Tuple([Value(-1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('C3', 'N_C2', Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('C2', 'N_C2', Tuple([Value(-1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('C2', 'N_C1', Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('C1', 'N_C1', Tuple([Value(-1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('C1', 'CN2', Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('N2', 'CN2', Tuple([Value(-1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))

        # Conveyor T
        n.add_input('N0', 't1', Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('T1', 't1', Tuple([Value(-1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('T1', 'T', Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))

        return n, self.transition


# trial_net = TrialConveyorNetwork([1, 1], [7, 5, 5], [4, 3])
# net, t = trial_net.trial_conveyor_petrinet()
# # print(net.get_marking())
# g = StateSpace(net)
# print(g.get())
# for i in g.modes(g.current):
#     t, b = i
# print(b['sq_no'])
# m = list(g.get()["S"])
# print(m[2][1])
# m = list(g.get().values())
# l = set()
# for i in m:
#     l =l.union(i)
# l = list(l)
# print(l)
# # if m[0][1] == 0:
# #     print('yes')
# r = [2, 2]
# m = g.modes(g.get())
# print(g.succ(g.current, 0))
# print(g.modes(g.current))
# print(g.succ(g.current, -1))
# print(g.current)
# print(g.succ(g.current, 0))
# print(net.get_marking())
# print(g.modes(g.current))
# print(g.succ(g.current, -1))
# print(net.get_marking())
# print(g.modes(g.current))
# net.draw('Updated_Network.png')
# if 'Green' in g.get():
#     print(g.current)
#     net.add_marking(Marking(Red = MultiSet(r)))
#     print(net.get_marking())
#     g.current = g.add(net.get_marking())
#     a = dict(g.get())
#     del a['Red']
#     del a['Green']
#     print(a)
#     length = 0
#     for i in range(len(a)):
#         length += len(list(a.values())[i])
#         print(length)
#     print('final len: ' + str(length))
#     a = list(a.values())[0]
#     print(len(a))
#     print(g.current)
#     print(True)
#     print(g.modes(g.current))
