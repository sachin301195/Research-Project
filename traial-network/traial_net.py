import tpn
import snakes.plugins

snakes.plugins.load([tpn, 'gv', 'bound'], 'snakes.nets', 'snk')
from snk import *