# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 21:10:32 2021

@author: busach
"""

import graphviz

# d = graphviz.Digraph(comment = 'Round Table')
# d.node('A', 'King Sachin')
# d.node('B', 'Queen Suman')
# d.node('C', 'Binny')
# d.edges(['AB', 'AC'])
# d.edge('B', 'C', constraint = 'true')
# print(d.source)
# d.render('test-output/round-table.gv', view = True)
# d.format = 'jpg'
# d.render()

# ps = graphviz.Digraph(name = 'Pet_Shop', node_attr = {'shape' : 'plaintext'})
# ps.node('parrot')
# ps.node('dead')
# ps.edge('parrot', 'dead')
# ps.graph_attr['rankdir'] = 'LR'
# ps.edge_attr.update(arrowhead = 'vee', arrowsize = '2')
# print(ps.source)

ni = graphviz.Graph('ni')
ni.attr('node', shape = 'rarrow')
ni.node('1', 'Ni!')
ni.node('2', 'Ni!')
ni.node('3', 'Ni!', shape = 'egg')
ni.node('4', 'Ni!', shape = 'line')
ni.attr('node', shape = 'star')
ni.node('5', 'Ni!')
ni.node('6', 'Ni!')
ni.attr(rankdir = 'LR')
ni.edges(['12', '23', '34', '45', '56'])
ni.format = 'jpg'
ni.render(view = True)

