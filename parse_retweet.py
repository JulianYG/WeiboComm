import snap
import numpy as np

NETWORK = './total.txt'
GRAPH = './network_retweet.txt'


def write_to_graph(load_file_name, save_file_name): 
	graph = snap.PNGraph.New()

	# 1787443
	row_number = 0
	original_uid = None
	retweet_num = None

	with open(load_file_name) as f:
		for line in f:
			row_number += 1
			if row_number%10000 == 0:
				print row_number
			elements = line.split()
			if row_number % 2 == 1:
				original_uid = int(elements[2])
				retweet_num = int(elements[3])
			else:
				if not graph.IsNode(original_uid):
					graph.AddNode(original_uid)
				for i in range(0, len(elements), 2):
					retweet_uid = int(elements[i])
					if not graph.IsNode(retweet_uid):
						graph.AddNode(retweet_uid)	
					graph.AddEdge(original_uid, retweet_uid)

	snap.SaveEdgeList(graph, save_file_name)

if __name__ == '__main__':
	write_to_graph(NETWORK, GRAPH)


