
from cem import NodeStat, EdgeStat, Config

import snap
import random
import pickle
import numpy as np
import scipy.stats as sp


if __name__ == '__main__':

    # load config
    conf = Config()    
    sina_network = snap.LoadEdgeList(snap.PNEANet, conf.network_file)

    with open(conf.edge_result, 'rb') as f:
        edge_dist = pickle.load(f)

    with open(conf.node_result, 'rb') as f:
        node_dist = pickle.load(f)

    print('Sampling probabilities...')

    # Can use empty dict for sigma here, if choose deterministic
    NodeStat.sample_probability(
        sina_network, 0, node_dist['mu'], node_dist['sigma'])

    EdgeStat.sample_probability(
        sina_network, 1, edge_dist['mu'], edge_dist['sigma'])

    node_edge_prob = NodeStat.get_prob_dict(sina_network, 0)
    edge_edge_prob = EdgeStat.get_prob_dict(sina_network, 1)

    print('Size of node prob dict: {}, edge prob dict: {}'.format(
        len(node_edge_prob), len(edge_edge_prob)
    ))
    assert(len(node_edge_prob) == len(edge_edge_prob))

    edges = random.sample(node_edge_prob.keys(), 10)

    print('Sanity check. Comparing probabilities of randomly selected edges:')

    print('Node model: ', [node_edge_prob[e] for e in edges])
    print('Edge model: ', [edge_edge_prob[e] for e in edges])

    node_prob = np.array(node_edge_prob.values(), dtype=np.float32)
    edge_prob = np.array(edge_edge_prob.values(), dtype=np.float32)

    diff = np.abs(node_prob - edge_prob)

    print('Mean, standard deviation of probability difference vectors: {}, {}'.format(
        np.mean(diff), np.std(diff)))

    print('Maximum, minimum difference between probability vectors: {}, {}'.format(
        np.max(diff), np.min(diff)))

    print('Entropy, energy of probability difference vectors: {}, {}'.format(
        sp.entropy(diff), np.linalg.norm(diff)))

    ############ Run algo on generated graph, and compare probs with ground truth
    print('Loading probilities...')

    with open(conf.ground_truth, 'rb') as f:
        probs = pickle.load(f)

    keys = node_edge_prob.keys()
    val_true = [probs[k] for k in keys]
    val_pred_node = [node_edge_prob[k] for k in keys]
    val_pred_edge = [edge_edge_prob[k] for k in keys]

    val_true = np.array(val_true)
    val_pred_node = np.array(val_pred_node)
    val_pred_edge = np.array(val_pred_edge)

    print "l2 rand distance (node vs true): ", np.sqrt(np.sum((np.random.uniform(size=len(node_edge_prob)) - val_true)**2))
    print "l2 distance (node vs true): ", np.sqrt(np.sum(( val_pred_node - val_true)**2))
    print "l2 distance (edge vs true): ", np.sqrt(np.sum(( val_pred_edge - val_true)**2))

    
    # TODO: plot prob difference distribution by node degrees
    degree_avgProb = {}
    




