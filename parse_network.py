<<<<<<< HEAD
=======
import csv
>>>>>>> 2cc942f71725aef13d0259efd265b52f6777c5d0
import snap as s
import numpy as np

NETWORK = './data/weibo_network.txt'
GRAPH = './data/network_graph.txt'


def write_to_graph(load_file_name, save_file_name): 
	graph = s.PNGraph.New()

	# 1787443
	row_number = 0
	with open(load_file_name) as f:
		for line in f:
			row_number += 1
			if row_number == 1:
				elements = line.split()
				n = int(elements[0])
				m = int(elements[1])
				graph.Reserve(int(n), int(m))
				continue
		
			#row = line
			if row_number%5000 == 0:
				print row_number

			data = line.split('\t')

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
	

