import snap
import collections
from graphviz import Digraph
import random

# counter = collections.Counter()
# with open('network_graph_small_small.txt', 'r') as f:
# 	lines = f.readlines()[3:]
# 	for l in lines:
# 		src, dest = l.split()
# 		counter.update([src, dest])

# selected_nodes = set([x[0] for x in counter.most_common(100)])

g_snap = snap.TNGraph.New() 

with open('network_graph_small_small.txt', 'r') as f:
	lines = f.readlines()[3:]
 	for l in lines:
 		src, dest = l.split()
 		src, dest = int(src), int(dest)
 		if not g_snap.IsNode(src):
			g_snap.AddNode(src)
		if not g_snap.IsNode(dest):
			g_snap.AddNode(dest)
		g_snap.AddEdge(src, dest)


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

print len(selected_nodes)
g = Digraph('G', filename='test.gv')
g.graph_attr.update(ranksep = "1.2 equally")

g.attr('node', shape='circle', fixedsize='true', width = '.3')
for n in selected_nodes:
	g.node(str(n), label='')

for l in lines:
	src, dest = l.split()
	if int(src) in selected_nodes and int(dest) in selected_nodes:
		g.edge(src, dest)

g.view()