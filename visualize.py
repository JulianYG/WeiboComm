import snap
import collections
from graphviz import Digraph
import random
from colorsys import hsv_to_rgb
import pickle
from cem import NodeStat, EdgeStat, Config

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

with open('./data/network_graph_small_small.txt', 'r') as f:
	lines = f.readlines()[3:]
 	for l in lines:
 		src, dest = l.split()
 		src, dest = int(src), int(dest)
 		if not g_snap.IsNode(src):
			g_snap.AddNode(src)
		if not g_snap.IsNode(dest):
			g_snap.AddNode(dest)
		g_snap.AddEdge(src, dest)

conf = Config() 
sina_network = snap.LoadEdgeList(snap.PNEANet, conf.network_file)
edge_probs = pickle.load(open('./data/graph_probs_small_small.pkl','rb'))

edge_dist = pickle.load(open('./data/node_res_small_small_real.pkl','rb'))
# print(edge_dist['sigma'])
NodeStat.sample_probability(
        sina_network, 0, edge_dist['mu'], edge_dist['sigma'])
res_probs = NodeStat.get_prob_dict(sina_network, 0) 




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
selected_edges_res = collections.defaultdict(list)
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


for l in lines:
	src, dest = l.split()
	if int(src) in selected_nodes and int(dest) in selected_nodes:
		selected_edges[src].append([dest, edge_probs[(int(src), int(dest))], 
			res_probs[(int(src), int(dest))]])

min_v, max_v = float('inf'), float('-inf')
min_v_res, max_v_res = float('inf'), float('-inf')
for src, v in selected_edges.items():
	prob_total = sum([p for _, p, _ in v])
	res_prob_total = sum([p for _, _, p in v])
	n = len(v)
	for i in range(len(v)):
		selected_edges[src][i][1] /= (prob_total/n)
		if res_prob_total == 0:
			selected_edges[src][i][2] = 1/n
		else:
			selected_edges[src][i][2] /= (res_prob_total/n)
		min_v = min(min_v, selected_edges[src][i][1])
		max_v = max(max_v, selected_edges[src][i][1])
		min_v_res = min(min_v_res, selected_edges[src][i][2])
		max_v_res = max(max_v_res, selected_edges[src][i][2])

print min_v, max_v, min_v_res, max_v_res
g1 = Digraph('G', engine='fdp')
g1.graph_attr.update(ranksep = "1.2 equally")

g2 = Digraph('G', engine='fdp')
g2.graph_attr.update(ranksep = "1.2 equally")

g1.attr('node', shape='circle', fixedsize='true', width = '.3')
for n in selected_nodes:
	g1.node(str(n), label='')
g2.attr('node', shape='circle', fixedsize='true', width = '.3')
for n in selected_nodes:
	g2.node(str(n), label='')

for src, v in selected_edges.items():
	for dest, p, _ in v:		
		g1.edge(src, dest, color=getColor(p, min_v,max_v))

for src, v in selected_edges.items():
	for dest, _, p in v:	
		g2.edge(src, dest, color=getColor(p, min_v_res,max_v_res))

g1.render(filename='img/truth.gv')
g2.render(filename='img/res.gv')