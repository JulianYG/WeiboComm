import csv
import snap as s
import numpy as np

NETWORK = './data/weibo_network.txt'
GRAPH = './data/network_graph.txt'


def write_to_graph(load_file_name, save_file_name): 
	graph = s.PNGraph.New()

	# 1787443
	with open(load_file_name) as f:

		reader = csv.reader(f)
		n, m = reader.next()[0].split('\t')
		graph.Reserve(int(n), int(m))

		while True:
			row = reader.next()
			if row is None:
				break

			data = row[0].split('\t')
			node = int(data[0])
			number = np.array(data[2:], dtype=np.int).reshape((int(data[1]), 2))

			for n in number:

				if not graph.IsNode(node):
					graph.AddNode(node)
				if not graph.IsNode(n[0]):
					graph.AddNode(n[0])	

				graph.AddEdge(node, n[0])

				# If flag marked, reciprocal following relation
				if n[1]:
					graph.AddEdge(n[0], node)

	s.SaveEdgeList(graph, save_file_name)
	return graph

def get_influence_set(graph):

	pass

if __name__ == '__main__':
	sina = write_to_graph(NETWORK, GRAPH)

	infl = get_influence_set(sina)
	

