
from cem import NodeStat, EdgeStat, Config

import pickle
import numpy as np
import scipy as sp


if __name__ == '__main__':

    # load config
    conf = Config()    
    sina_network = snap.LoadEdgeList(snap.PNGraph, conf.network_file)

    with open(conf.edge_result, 'rb') as f:
    	edge_dist = pickle.load(f)

    with open(conf.node_result, 'rb') as f:
    	node_dist = pickle.load(f)

    node_edge_prob = NodeStat.sample_probability(
    	sina_network, node_dist['mu'], node_dist['sigma'])

    edge_edge_prob = EdgeStat.sample_probability(
    	sina_network, edge_dist['mu'], edge_dist['sigma'])

    print('Size of node prob dict: {}, edge prob dict: {}'.format(
    	len(node_edge_prob), len(edge_edge_prob)
    ))

    assert(len(node_edge_prob) == len(edge_edge_prob))

    node_prob = np.array(node_edge_prob.values(), dtype=np.float32)
    edge_prob = np.array(edge_edge_prob.values(), dtype=np.float32)

    diff = node_prob - edge_prob

    print('Mean, standard deviation of probability difference vectors: {}, {}, {}'.format(
    	np.mean(diff), np.std(diff)))

   	print('Maximum, minimum difference between probability vectors: {}, {}'.format(
   		np.max(np.abs(node_prob - edge_prob)), np.min(np.abs(node_prob - edge_prob))))

   	print('Entropy, energy of probability difference vectors: {}, {}'.format(
		sc.stats.entropy(diff), np.linalg.norm(diff)))
 	
    # TODO: plot prob difference distribution by node degrees

