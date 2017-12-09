import snap
import collections
from graphviz import Digraph
import random
from colorsys import hsv_to_rgb
import pickle

# counter = collections.Counter()
# with open('network_graph_small_small.txt', 'r') as f:
# 	lines = f.readlines()[3:]
# 	for l in lines:
# 		src, dest = l.split()
# 		counter.update([src, dest])

# selected_nodes = set([x[0] for x in counter.most_common(100)])

	

def getColor(val, minval, maxval):
    """ Convert val in range minval..maxval to the range 0..120 degrees which
        correspond to the colors Red and Green in the HSV colorspace.
    """
    h = (float(val-minval) / (maxval-minval)) * 120

    return str(h/360)+'  1.000 1.000'

    # r, g, b = hsv_to_rgb(h/360, 1., 1.)
    # return r, g, b


g_snap = snap.TNGraph.New() 

with open('../data/network_graph_small_small.txt', 'r') as f:
	lines = f.readlines()[3:]
 	for l in lines:
 		src, dest = l.split()
 		src, dest = int(src), int(dest)
 		if not g_snap.IsNode(src):
			g_snap.AddNode(src)
		if not g_snap.IsNode(dest):
			g_snap.AddNode(dest)
		g_snap.AddEdge(src, dest)

edge_probs = pickle.load(open('../data/graph_probs_small_small.pickle','rb'))

# Components = snap.TCnComV()
# snap.GetSccs(g_snap, Components)
# selected_nodes = set()
# for CnCom in Components:
#     print CnCom.Len()
#     if CnCom.Len() == 19492:
#     	selected_nodes = [x for x in CnCom]

# selected_nodes = set(random.sample(selected_nodes, 600))

max_nid = snap.GetMxDegNId(g_snap)
BfsTree = snap.GetBfsTree(g_snap, max_nid, True, False)
selected_nodes = set()
selected_edges = collections.defaultdict(list)
counter = collections.Counter()
for EI in BfsTree.Edges():
	src, dest = EI.GetSrcNId(), EI.GetDstNId()
	counter.update([src, dest])
	if counter[src] >10 or counter[dest] >10:
		continue
	selected_nodes.add(src)
	selected_nodes.add(dest)
	if len(selected_nodes) >= 100:
		break

g = Digraph('G', filename='test.gv')
g.graph_attr.update(ranksep = "1.2 equally")

g.attr('node', shape='circle', fixedsize='true', width = '.3')
for n in selected_nodes:
	g.node(str(n), label='')

for l in lines:
	src, dest = l.split()
	if int(src) in selected_nodes and int(dest) in selected_nodes:
		selected_edges[src].append(dest)

for src, v in selected_edges.items():
	prob_total = sum([edge_probs[(src, dest)] for dest in v])
	for dest in v:
		g.edge(src, dest, color=getColor(edge_probs[(src, dest)]/prob_total, 0,1))


g.view()