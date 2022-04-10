import sys

sys.path.append('./snakes_master')
import snakes.plugins
from snakes.utils.simul import StateSpace
from snakes.nets import *

snakes.plugins.load(['gv', 'bound'], 'snakes.nets', 'snk')
from snk import *

places = ['S', 'S1', 'N1', 'A1', 'A2', 'A3', 'B1', 'B2', 'B3', 'N2', 'C1', 'C2', 'C3', 'D1', 'D2', 'D3',
          'E1', 'E2', 'E3', 'F1', 'F2', 'F3', 'J1', 'J2', 'J3', 'G1', 'G2', 'G3', 'K1', 'K2', 'K3', 'T1',
          'W1', 'Red', 'Green', 'N3', 'N4', 'N0', 'N6', 'N9', 'I1', 'I2', 'I3', 'H1', 'H2', 'H3', 'L1',
          'L2', 'L3', 'M1', 'M2', 'M3', 'O1', 'O2', 'O3', 'P1', 'P2', 'P3', 'Q1', 'Q2', 'Q3', 'N5', 'N7',
          'N8', 'W2', 'Blue', 'Violet']

trans = ['s1', 'SN1', 'AN1', 'BN1', 'P_A1', 'P_A2', 'P_A3', 'N_A1', 'N_A2', 'N_A3', 'P_B1', 'P_B2', 'P_B3',
         'N_B1', 'N_B2', 'N_B3', 'BN9', 'JN9', 'KN9', 'P_K1', 'P_K2', 'P_K3', 'N_K1', 'N_K2', 'N_K3', 'KN0',
         't1', 'T', 'P_J1', 'P_J2', 'P_J3', 'N_J1', 'N_J2', 'N_J3', 'JN6', 'GN6', 'P_G1', 'P_G2', 'P_G3',
         'N_G1', 'N_G2', 'N_G3', 'GN3', 'FN3', 'EN3', 'P_E1', 'P_E2', 'P_E3', 'N_E1', 'N_E2', 'N_E3', 'AN2',
         'EN2', 'CN2', 'P_C1', 'P_C2', 'P_C3', 'N_C1', 'N_C2', 'N_C3', 'C0W1', 'C1W1', 'C2W1', 'C3W1', 'D0W1',
         'D1W1', 'D2W1', 'D3W1', 'P_D1', 'P_D2', 'P_D3', 'N_D1', 'N_D2', 'N_D3', 'DN4', 'FN4', 'P_F1', 'P_F2',
         'P_F3', 'N_F1', 'N_F2', 'N_F3', 'P_H1', 'P_H2', 'P_H3', 'N_H1', 'N_H2', 'N_H3', 'P_I1', 'P_I2', 'P_I3',
         'N_I1', 'N_I2', 'N_I3', 'P_L1', 'P_L2', 'P_L3', 'N_L1', 'N_L2', 'N_L3', 'P_M1', 'P_M2', 'P_M3', 'N_M1',
         'N_M2', 'N_M3', 'P_O1', 'P_O2', 'P_O3', 'N_O1', 'N_O2', 'N_O3', 'P_P1', 'P_P2', 'P_P3', 'N_P1', 'N_P2',
         'N_P3', 'P_Q1', 'P_Q2', 'P_Q3', 'N_Q1', 'N_Q2', 'N_Q3', 'HN4', 'HN5', 'IN6', 'IN7', 'LN5', 'L0W2', 'L4W2',
         'L8W2', 'L12W2', 'M0W2', 'M4W2', 'M8W2', 'M12W2', 'MN8', 'ON8', 'ON0', 'PN5', 'PN7', 'QN7', 'QN8']


class ConveyorNetwork:
    def __init__(self, jobs, resources, quantity):
        self.fs = jobs
        self.cs = [0 for _ in jobs]
        self.no_of_jobs = len(jobs)
        self.res = resources[0]
        self.red = resources[1]
        self.green = resources[2]
        self.blue = resources[3]
        self.violet = resources[4]
        self.orders = quantity
        self.token = {}
        self.t_init = []
        self.t_red = [1 for _ in range(self.red)]
        self.t_green = [2 for _ in range(self.green)]
        self.t_blue = [4 for _ in range(self.blue)]
        self.t_violet = [8 for _ in range(self.violet)]
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
                self.token[f'job{self.fs[i]}'] = self.token[f'job{self.fs[i]}'][:1] + (self.object_no,) + \
                                                 self.token[f'job{self.fs[i]}'][2:]
                self.object_no += 1
                self.t_init.append(self.token[f'job{self.fs[i]}'])
        return self.t_init, self.t_red, self.t_green, self.t_blue, self.t_violet

    def conveyor_petrinet(self, bounds=1):
        n = PetriNet('trial_net')
        n.globals.declare('c = 0')
        n.globals.declare('f = 0')

        init, r, g, b, v = self.tokens()

        # adding places
        for i in places:
            if i == 'S':
                n.add_place(Place('%s' % i, init, bound=(0, None)))
            elif i == 'Green':
                n.add_place(Place('%s' % i, g, bound=(0, None)))
            elif i == 'Red':
                n.add_place(Place('%s' % i, r, bound=(0, None)))
            elif i == 'Blue':
                n.add_place(Place('%s' % i, b, bound=(0, None)))
            elif i == 'Violet':
                n.add_place(Place('%s' % i, v, bound=(0, None)))
            else:
                n.add_place(Place('%s' % i, [], bound=bounds))

        # adding transitions
        for i in trans:
            if i == 't1':
                self.transition.update({i: Transition('%s' % i, Expression('c == f'))})
            elif i == 'C1W1' or i == 'D1W1':
                self.transition.update \
                    ({i: Transition('%s' % i,
                                    Expression('f in [1, 5, 9, 13] and c<f and c!=1 and c!=5 and c!=9 and c!=13'))})
            elif i == 'C2W1' or i == 'D2W1':
                self.transition.update({i: Transition('%s' % i,
                                    Expression('f in [2, 6, 10, 14] and c<f and c!=2 and c!=6 and c!=10 and c!=14'))})
            elif i == 'C3W1' or i == 'D3W1':
                self.transition.update({i: Transition('%s' % i,
                                    Expression('f in [3, 7, 11, 15] and c<f and c!=3 and c!=7 and c!=11 and c!=15'))})
            elif i == 'L4W2' or i == 'M4W2':
                self.transition.update({i: Transition('%s' % i,
                                    Expression('f in [4, 5, 6, 7] and c<f and c!=4 and c!=5 and c!=6 and c!=7'))})
            elif i == 'L8W2' or i == 'M8W2':
                self.transition.update({i: Transition('%s' % i,
                                    Expression('f in [8, 9, 10, 11] and c<f and c!=8 and c!=9 and c!=10 and c!=11'))})
            elif i == 'L12W2' or i == 'M12W2':
                self.transition.update({i: Transition('%s' % i,
                                Expression('f in [12, 13, 14, 15] and c<f and c!=12 and c!=13 and c!=14 and c!=15'))})
            else:
                self.transition.update({i: Transition('%s' % i)})
            n.add_transition(self.transition[i])

        # adding input and output
        # Conveyor S
        n.add_input('S', 's1',
                    Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('S1', 's1',
                     Tuple([Value(1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('S1', 'SN1',
                    Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('N1', 'SN1',
                     Tuple([Value(1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))

        # Conveyor A
        n.add_input('N1', 'P_A1',
                    Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('A1', 'P_A1',
                     Tuple([Value(1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('A1', 'P_A2',
                    Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('A2', 'P_A2',
                     Tuple([Value(1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('A2', 'P_A3',
                    Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('A3', 'P_A3',
                     Tuple([Value(1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('A3', 'AN2',
                    Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('N2', 'AN2',
                     Tuple([Value(1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('N2', 'N_A3',
                    Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('A3', 'N_A3',
                     Tuple([Value(-1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('A3', 'N_A2',
                    Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('A2', 'N_A2',
                     Tuple([Value(-1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('A2', 'N_A1',
                    Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('A1', 'N_A1',
                     Tuple([Value(-1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('A1', 'AN1',
                    Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('N1', 'AN1',
                     Tuple([Value(-1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))

        # Conveyor B
        n.add_input('N1', 'N_B3',
                    Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('B3', 'N_B3',
                     Tuple([Value(-1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('B3', 'N_B2',
                    Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('B2', 'N_B2',
                     Tuple([Value(-1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('B2', 'N_B1',
                    Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('B1', 'N_B1',
                     Tuple([Value(-1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('B1', 'BN9',
                    Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('N9', 'BN9',
                     Tuple([Value(-1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('N9', 'P_B1',
                    Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('B1', 'P_B1',
                     Tuple([Value(1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('B1', 'P_B2',
                    Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('B2', 'P_B2',
                     Tuple([Value(1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('B2', 'P_B3',
                    Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('B3', 'P_B3',
                     Tuple([Value(1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('B3', 'BN1',
                    Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('N1', 'BN1',
                     Tuple([Value(1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))

        # Conveyor E
        n.add_input('N2', 'N_E3',
                    Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('E3', 'N_E3',
                     Tuple([Value(-1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('E3', 'N_E2',
                    Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('E2', 'N_E2',
                     Tuple([Value(-1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('E2', 'N_E1',
                    Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('E1', 'N_E1',
                     Tuple([Value(-1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('E1', 'EN3',
                    Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('N3', 'EN3',
                     Tuple([Value(-1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('N3', 'P_E1',
                    Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('E1', 'P_E1',
                     Tuple([Value(1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('E1', 'P_E2',
                    Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('E2', 'P_E2',
                     Tuple([Value(1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('E2', 'P_E3',
                    Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('E3', 'P_E3',
                     Tuple([Value(1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('E3', 'EN2',
                    Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('N2', 'EN2',
                     Tuple([Value(1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))

        # Conveyor F
        n.add_input('N3', 'P_F1',
                    Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('F1', 'P_F1',
                     Tuple([Value(1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('F1', 'P_F2',
                    Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('F2', 'P_F2',
                     Tuple([Value(1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('F2', 'P_F3',
                    Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('F3', 'P_F3',
                     Tuple([Value(1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('F3', 'FN4',
                    Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('N4', 'FN4',
                     Tuple([Value(1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('N4', 'N_F3',
                    Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('F3', 'N_F3',
                     Tuple([Value(-1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('F3', 'N_F2',
                    Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('F2', 'N_F2',
                     Tuple([Value(-1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('F2', 'N_F1',
                    Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('F1', 'N_F1',
                     Tuple([Value(-1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('F1', 'FN3',
                    Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('N3', 'FN3',
                     Tuple([Value(-1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))

        # Conveyor G
        n.add_input('N3', 'N_G3',
                    Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('G3', 'N_G3',
                     Tuple([Value(-1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('G3', 'N_G2',
                    Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('G2', 'N_G2',
                     Tuple([Value(-1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('G2', 'N_G1',
                    Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('G1', 'N_G1',
                     Tuple([Value(-1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('G1', 'GN6',
                    Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('N6', 'GN6',
                     Tuple([Value(-1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('N6', 'P_G1',
                    Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('G1', 'P_G1',
                     Tuple([Value(1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('G1', 'P_G2',
                    Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('G2', 'P_G2',
                     Tuple([Value(1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('G2', 'P_G3',
                    Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('G3', 'P_G3',
                     Tuple([Value(1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('G3', 'GN3',
                    Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('N3', 'GN3',
                     Tuple([Value(1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))

        # Conveyor J
        n.add_input('N6', 'N_J3',
                    Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('J3', 'N_J3',
                     Tuple([Value(-1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('J3', 'N_J2',
                    Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('J2', 'N_J2',
                     Tuple([Value(-1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('J2', 'N_J1',
                    Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('J1', 'N_J1',
                     Tuple([Value(-1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('J1', 'JN9',
                    Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('N9', 'JN9',
                     Tuple([Value(-1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('N9', 'P_J1',
                    Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('J1', 'P_J1',
                     Tuple([Value(1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('J1', 'P_J1',
                    Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('J2', 'P_J2',
                     Tuple([Value(1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('J2', 'P_J3',
                    Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('J3', 'P_J3',
                     Tuple([Value(1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('J3', 'JN6',
                    Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('N6', 'JN6',
                     Tuple([Value(1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))

        # Conveyor K
        n.add_input('N9', 'N_K3',
                    Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('K3', 'N_K3',
                     Tuple([Value(-1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('K3', 'N_K2',
                    Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('K2', 'N_K2',
                     Tuple([Value(-1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('K2', 'N_K1',
                    Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('K1', 'N_K1',
                     Tuple([Value(-1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('K1', 'KN0',
                    Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('N0', 'KN0',
                     Tuple([Value(-1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('N0', 'P_K1',
                    Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('K1', 'P_K1',
                     Tuple([Value(1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('K1', 'P_K2',
                    Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('K2', 'P_K2',
                     Tuple([Value(1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('K2', 'P_K3',
                    Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('K3', 'P_K3',
                     Tuple([Value(1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('K3', 'KN9',
                    Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('N9', 'KN9',
                     Tuple([Value(1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))

        # Conveyor C & D
        n.add_input('N2', 'P_C1',
                    Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('C1', 'P_C1',
                     Tuple([Value(1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('C1', 'P_C2',
                    Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('C2', 'P_C2',
                     Tuple([Value(1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('C2', 'P_C3',
                    Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('C3', 'P_C3',
                     Tuple([Value(1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        # C to W1
        n.add_input('C3', 'C0W1',
                    Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('W1', 'C0W1',
                     Tuple([Value(-1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('C3', 'C1W1',
                    Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        # n.add_input('Red', 'C1W1', Variable('x'))
        n.add_output('W1', 'C1W1',
                     Tuple([Value(-1), Variable('sq_no'), Expression('c + 1'), Variable('f'), Expression('count + 1')]))
        n.add_input('C3', 'C2W1',
                    Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        # n.add_input('Green', 'C2W1', Variable('y'))
        n.add_output('W1', 'C2W1',
                     Tuple([Value(-1), Variable('sq_no'), Expression('c + 2'), Variable('f'), Expression('count + 1')]))
        n.add_input('C3', 'C3W1',
                    Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        # n.add_input('Red', 'C3W1',
        #             Variable('x'))
        # n.add_input('Green', 'C3W1', Variable('y'))
        n.add_output('W1', 'C3W1',
                     Tuple([Value(-1), Variable('sq_no'), Expression('c + 3'), Variable('f'), Expression('count + 1')]))
        # W1 to D
        n.add_input('W1', 'N_D3',
                    Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('D3', 'N_D3',
                     Tuple([Value(-1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('D3', 'N_D2',
                    Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('D2', 'N_D2',
                     Tuple([Value(-1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('D2', 'N_D1',
                    Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('D1', 'N_D1',
                     Tuple([Value(-1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('D1', 'DN4',
                    Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('N4', 'DN4',
                     Tuple([Value(-1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('N4', 'P_D1',
                    Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('D1', 'P_D1',
                     Tuple([Value(1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('D1', 'P_D2',
                    Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('D2', 'P_D2',
                     Tuple([Value(1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('D2', 'P_D3',
                    Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('D3', 'P_D3',
                     Tuple([Value(1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        # D to W1
        n.add_input('D3', 'D0W1',
                    Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('W1', 'D0W1',
                     Tuple([Value(1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('D3', 'D1W1',
                    Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        # n.add_input('Red', 'D1W1', Variable('x'))
        n.add_output('W1', 'D1W1',
                     Tuple([Value(1), Variable('sq_no'), Expression('c + 1'), Variable('f'), Expression('count + 1')]))
        n.add_input('D3', 'D2W1',
                    Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        # n.add_input('Green', 'D2W1', Variable('y'))
        n.add_output('W1', 'D2W1',
                     Tuple([Value(1), Variable('sq_no'), Expression('c + 2'), Variable('f'), Expression('count + 1')]))
        n.add_input('D3', 'D3W1',
                    Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        # n.add_input('Red', 'D3W1', Variable('x'))
        # n.add_input('Green', 'D3W1', Variable('y'))
        n.add_output('W1', 'D3W1',
                     Tuple([Value(1), Variable('sq_no'), Expression('c + 3'), Variable('f'), Expression('count + 1')]))
        # W1 to C
        n.add_input('W1', 'N_C3',
                    Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('C3', 'N_C3',
                     Tuple([Value(-1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('C3', 'N_C2',
                    Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('C2', 'N_C2',
                     Tuple([Value(-1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('C2', 'N_C1',
                    Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('C1', 'N_C1',
                     Tuple([Value(-1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('C1', 'CN2',
                    Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('N2', 'CN2',
                     Tuple([Value(-1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))

        # Conveyor H
        n.add_input('N4', 'N_H3',
                    Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('H3', 'N_H3',
                     Tuple([Value(-1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('H3', 'N_H2',
                    Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('H2', 'N_H2',
                     Tuple([Value(-1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('H2', 'N_H1',
                    Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('H1', 'N_H1',
                     Tuple([Value(-1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('H1', 'HN5',
                    Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('N5', 'HN5',
                     Tuple([Value(-1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('N5', 'P_H1',
                    Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('H1', 'P_H1',
                     Tuple([Value(1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('H1', 'P_H2',
                    Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('H2', 'P_H2',
                     Tuple([Value(1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('H2', 'P_H3',
                    Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('H3', 'P_H3',
                     Tuple([Value(1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('H3', 'HN4',
                    Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('N4', 'HN4',
                     Tuple([Value(1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))

        # Conveyor I
        n.add_input('N6', 'N_I3',
                    Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('I3', 'N_I3',
                     Tuple([Value(-1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('I3', 'N_I2',
                    Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('I2', 'N_I2',
                     Tuple([Value(-1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('I2', 'N_I1',
                    Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('I1', 'N_I1',
                     Tuple([Value(-1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('I1', 'IN7',
                    Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('N7', 'IN7',
                     Tuple([Value(-1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('N7', 'P_I1',
                    Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('I1', 'P_I1',
                     Tuple([Value(1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('I1', 'P_I2',
                    Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('I2', 'P_I2',
                     Tuple([Value(1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('I2', 'P_I3',
                    Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('I3', 'P_I3',
                     Tuple([Value(1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('I3', 'IN6',
                    Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('N6', 'IN6',
                     Tuple([Value(1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))

        # Conveyor O
        n.add_input('N0', 'N_O3',
                    Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('O3', 'N_O3',
                     Tuple([Value(-1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('O3', 'N_O2',
                    Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('O2', 'N_O2',
                     Tuple([Value(-1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('O2', 'N_O1',
                    Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('O1', 'N_O1',
                     Tuple([Value(-1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('O1', 'ON8',
                    Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('N8', 'ON8',
                     Tuple([Value(-1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('N8', 'P_O1',
                    Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('O1', 'P_O1',
                     Tuple([Value(1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('O1', 'P_O2',
                    Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('O2', 'P_O2',
                     Tuple([Value(1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('O2', 'P_O3',
                    Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('O3', 'P_O3',
                     Tuple([Value(1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('O3', 'ON0',
                    Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('N0', 'ON0',
                     Tuple([Value(1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))

        # Conveyor P
        n.add_input('N5', 'N_P3',
                    Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('P3', 'N_P3',
                     Tuple([Value(-1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('P3', 'N_P2',
                    Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('P2', 'N_P2',
                     Tuple([Value(-1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('P2', 'N_P1',
                    Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('P1', 'N_P1',
                     Tuple([Value(-1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('P1', 'PN7',
                    Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('N7', 'PN7',
                     Tuple([Value(-1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('N7', 'P_P1',
                    Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('P1', 'P_P1',
                     Tuple([Value(1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('P1', 'P_P2',
                    Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('P2', 'P_P2',
                     Tuple([Value(1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('P2', 'P_P3',
                    Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('P3', 'P_P3',
                     Tuple([Value(1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('P3', 'PN5',
                    Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('N5', 'PN5',
                     Tuple([Value(1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))

        # Conveyor Q
        n.add_input('N7', 'N_Q3',
                    Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('Q3', 'N_Q3',
                     Tuple([Value(-1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('Q3', 'N_Q2',
                    Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('Q2', 'N_Q2',
                     Tuple([Value(-1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('Q2', 'N_Q1',
                    Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('Q1', 'N_Q1',
                     Tuple([Value(-1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('Q1', 'QN8',
                    Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('N8', 'QN8',
                     Tuple([Value(-1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('N8', 'P_Q1',
                    Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('Q1', 'P_Q1',
                     Tuple([Value(1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('Q1', 'P_Q2',
                    Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('Q2', 'P_Q2',
                     Tuple([Value(1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('Q2', 'P_Q3',
                    Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('Q3', 'P_Q3',
                     Tuple([Value(1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('Q3', 'QN7',
                    Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('N7', 'QN7',
                     Tuple([Value(1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))

        # Conveyor L & M
        n.add_input('N8', 'P_M1',
                    Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('M1', 'P_M1',
                     Tuple([Value(1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('M1', 'P_M2',
                    Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('M2', 'P_M2',
                     Tuple([Value(1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('M2', 'P_M3',
                    Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('M3', 'P_M3',
                     Tuple([Value(1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        # M to W2
        n.add_input('M3', 'M0W2',
                    Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('W2', 'M0W2',
                     Tuple([Value(-1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('M3', 'M4W2',
                    Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        # n.add_input('Blue', 'M4W2', Variable('x'))
        n.add_output('W2', 'M4W2',
                     Tuple([Value(-1), Variable('sq_no'), Expression('c + 4'), Variable('f'), Expression('count + 1')]))
        n.add_input('M3', 'M8W2',
                    Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        # n.add_input('Violet', 'M8W2', Variable('y'))
        n.add_output('W2', 'M8W2',
                     Tuple([Value(-1), Variable('sq_no'), Expression('c + 8'), Variable('f'), Expression('count + 1')]))
        n.add_input('M3', 'M12W2',
                    Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        # n.add_input('Blue', 'M12W2',
        #             Variable('x'))
        # n.add_input('Violet', 'M12W2', Variable('y'))
        n.add_output('W2', 'M12W2',
                     Tuple([Value(-1), Variable('sq_no'), Expression('c +12'), Variable('f'), Expression('count + 1')]))
        # W2 to L
        n.add_input('W2', 'P_L1',
                    Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('L1', 'P_L1',
                     Tuple([Value(-1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('L1', 'P_L2',
                    Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('L2', 'P_L2',
                     Tuple([Value(-1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('L2', 'P_L3',
                    Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('L3', 'P_L3',
                     Tuple([Value(-1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('L3', 'LN5',
                    Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('N5', 'LN5',
                     Tuple([Value(-1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('N5', 'N_L3',
                    Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('L3', 'N_L3',
                     Tuple([Value(1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('L3', 'N_L2',
                    Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('L2', 'N_L2',
                     Tuple([Value(1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('L2', 'N_L1',
                    Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('L1', 'N_L1',
                     Tuple([Value(1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        # L to W2
        n.add_input('L1', 'L0W2',
                    Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('W2', 'L0W2',
                     Tuple([Value(1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('L1', 'L4W2',
                    Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        # n.add_input('Blue', 'L4W2', Variable('x'))
        n.add_output('W2', 'L4W2',
                     Tuple([Value(1), Variable('sq_no'), Expression('c + 4'), Variable('f'), Expression('count + 1')]))
        n.add_input('L1', 'L8W2',
                    Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        # n.add_input('Violet', 'L8W2', Variable('y'))
        n.add_output('W2', 'L8W2',
                     Tuple([Value(1), Variable('sq_no'), Expression('c + 8'), Variable('f'), Expression('count + 1')]))
        n.add_input('L1', 'L12W2',
                    Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        # n.add_input('Blue', 'L12W2', Variable('x'))
        # n.add_input('Violet', 'L12W2', Variable('y'))
        n.add_output('W2', 'L12W2',
                     Tuple([Value(1), Variable('sq_no'), Expression('c+12'), Variable('f'), Expression('count + 1')]))
        # W2 to M
        n.add_input('W2', 'N_M3',
                    Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('M3', 'N_M3',
                     Tuple([Value(-1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('M3', 'N_M2',
                    Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('M2', 'N_M2',
                     Tuple([Value(-1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('M2', 'N_M1',
                    Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('M1', 'N_M1',
                     Tuple([Value(-1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('M1', 'MN8',
                    Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('N8', 'MN8',
                     Tuple([Value(-1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))

        # Conveyor T
        n.add_input('N0', 't1',
                    Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))
        n.add_output('T1', 't1',
                     Tuple([Value(-1), Variable('sq_no'), Variable('c'), Variable('f'), Expression('count + 1')]))
        n.add_input('T1', 'T',
                    Tuple([Variable('dir'), Variable('sq_no'), Variable('c'), Variable('f'), Variable('count')]))

        return n, self.transition

#
# trial_net = ConveyorNetwork([15, 14], [7, 5, 5, 5, 5], [4, 3])
# net, t = trial_net.conveyor_petrinet()
# print(net.get_marking())
# mark = net.get_marking()
# print(mark['S'])

# net.draw('Network.png')
# modes = net.transition('s1').modes()
# net.transition('s1').fire(modes[0])
# modes = net.transition('s1').modes()
# m = net.get_marking()
# print(modes)
# print(modes[0]['sq_no'])
# print(list(m['S1'])[0][-1])
# print(net.post('SN1'))
#
#
# g = StateSpace(net)
# print(g.get())
# mode = g.modes(g.current)
# print(mode)
# point = {}
# step=0


# for trns, binding in mode:
#     #print(trns)
#     #print(binding)
#     point[trns] = binding
#     print(binding('sq_no') == 2)
# print(point)
# m = list(g.get().keys())
# mode = g.modes(g.current)
# print(mode)
# for t, b in mode:
#     print(str(t))
#     if str(t) == 's1':
#         print(t, b)
# # t, b = mode
# print(t, b)

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

# import random
# NEXT_TRANSITIONS = {'S': ['s1'], 'S1': ['SN1'], 'N1': ['P_A1', 'N_B3'], 'A1': ['AN1', 'P_A2'], 'A2': ['N_A1', 'P_A3'],
#                     'A3': ['N_A2', 'AN2'], 'B1': ['BN9', 'P_B2'], 'B2': ['N_B1', 'P_B3'], 'B3': ['N_B2', 'BN1'],
#                     'N2': ['N_A3', 'P_C1', 'N_E3'], 'C1': ['CN2', 'P_C2'], 'C2': ['N_C1', 'P_C3'], 'C3':
#                     ['N_C2', 'C2W1', 'C3W1', 'C0W1', 'C1W1'], 'D1': ['P_D2', 'DN4'], 'D2': ['N_D1', 'P_D3'],
#                     'D3': ['N_D2', 'D1W1', 'D0W1', 'D3W1', 'D2W1'], 'E1': ['EN3', 'P_E2'], 'E2': ['P_E3', 'N_E1'],
#                     'E3': ['N_E2', 'EN2'], 'F1': ['FN3', 'P_F2'], 'F2': ['N_F1', 'P_F3'], 'F3': ['N_F2', 'FN4'],
#                     'J1': ['P_J2', 'JN9'], 'J2': ['P_J3', 'N_J1'], 'J3': ['N_J2', 'JN6'], 'G1': ['P_G2', 'GN6'],
#                     'G2': ['P_G3', 'N_G1'], 'G3': ['N_G2', 'GN3'], 'K1': ['P_K2', 'KN0'], 'K2': ['P_K3', 'N_K1'],
#                     'K3': ['N_K2', 'KN9'], 'T1': ['T'], 'W1': ['N_D3', 'N_C3'], 'Red': ['D1W1', 'D3W1', 'C1W1', 'C3W1'],
#                     'Green': ['C2W1', 'C3W1', 'D2W1', 'D3W1'], 'N3': ['P_E1', 'N_G3', 'P_F1'], 'N4':
#                     ['P_D1', 'N_F3', 'N_H3'], 'N0': ['P_K1', 't1', 'N_O3'], 'N6': ['P_G1', 'N_J3', 'N_I3'],
#                     'N9': ['P_B1', 'N_K3', 'P_J1'], 'H1': ['P_H2', 'HN5'], 'H2': ['N_H1', 'P_H3'],
#                     'H3': ['N_H2', 'HN4'], 'I1': ['IN7', 'P_I2'], 'I2': ['N_I1', 'P_I3'], 'I3': ['N_I2', 'IN6'],
#                     'L1': ['P_L2', 'L0W2', 'L4W2', 'L8W2', 'L12W2'], 'L2': ['P_L3', 'N_L1'], 'L3': ['N_L2', 'LN5'],
#                     'M1': ['MN8', 'P_M2'], 'M2': ['N_M1', 'P_M3'], 'M3': ['N_M2', 'M0W2', 'M4W2', 'M8W2', 'M12W2'],
#                     'O1': ['ON8', 'P_O2'], 'O2': ['N_O1', 'P_O3'], 'O3': ['N_O2', 'ON0'], 'P1': ['PN7', 'P_P2'],
#                     'P2': ['N_P1', 'P_P3'], 'P3': ['N_P2', 'PN5'], 'Q1': ['QN8', 'P_Q2'], 'Q2': ['N_Q1', 'P_Q3'],
#                     'Q3': ['N_Q2', 'QN7'], 'N5': ['N_L3', 'P_H1', 'N_P3'], 'N7': ['P_P1', 'N_Q3', 'P_I1'], 'N8':
#                     ['P_M1', 'P_Q1', 'P_O1'], 'W2': ['P_L1', 'N_M3'], 'Blue': ['M4W2', 'M12W2', 'L4W2', 'L12W2'],
#                     'Violet': ['M8W2', 'M12W2', 'L8W2', 'L12W2']}
# trial_net = ConveyorNetwork([1], [5, 6, 6, 6, 6], [5])
# net, t = trial_net.conveyor_petrinet()
# print(net.get_marking())
# modes = net.transition('s1').modes()
# print(modes)
# done = False
# while not done:
#     place = list(net.get_marking().keys())
#     place.remove('Green')
#     place.remove('Red')
#     place.remove('Blue')
#     place.remove('Violet')
#     print(place)
#     if len(place) == 0:
#         done = True
#         break
#     current_place = place[-1]
#     trans = NEXT_TRANSITIONS[current_place]
#
#     trans_fire = random.choice(trans)
#
#     modes = net.transition(trans_fire).modes()
#     if place[-1] == 'T1':
#         print(place)
#         print(trans)
#         print(trans_fire)
#         print(modes)
#         print(net.get_marking())
#     if len(modes) != 0:
#         try:
#             token = [(modes[0]['dir'], modes[0]['sq_no'], modes[0]['c'], modes[0]['f'], modes[0]['count'])]
#             net.transition(trans_fire).fire(modes[0])
#         except:
#             print('------------------------------ERROR--------------------------------------')
#             # net.add_marking(Marking(S=MultiSet(token)))
#             net.place(current_place).add(token)
#     else:
#         continue
#     print(current_place)
#     print(token)
#     print(place)
#     print(trans)
#     print(trans_fire)
#     print(modes)
#     print(net.get_marking())
#     # done = True
