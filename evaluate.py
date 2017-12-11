
from __future__ import print_function
from cem import NodeStat, EdgeStat, Config

import snap
import random
import pickle
import numpy as np
import scipy.stats as sp
import matplotlib.pyplot as plt
import math

def assign_prob(graph):
    probs = {}
    for node_v in graph.Nodes():
        in_d = node_v.GetInDeg()
        r = np.random.uniform(size = in_d)
        p = r/r.sum()
        # r = [random.uniform(0,1) for i in range(out_d)]
        # sum_ = sum(r)
        # p = [i/sum_ for i in r]
        v = node_v.GetId()
        for idx in range(node_v.GetInDeg()):
            u = node_v.GetInNId(idx)
            probs[(u,v)] = p[idx]
    return probs

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

    probs_rand = assign_prob(sina_network)
    # val_rand = np.array([probs_rand[k] for k in node_edge_prob])
    # val_true = np.array([probs[k] for k in node_edge_prob])
    # val_pred_node = np.array([node_edge_prob[k] for k in node_edge_prob])
    # val_pred_edge = np.array([edge_edge_prob[k] for k in node_edge_prob])
    keys_not_1 = []
    for k in node_edge_prob:
        if probs[k] < 1:
            keys_not_1.append(k)
    val_rand = np.array([probs_rand[k] for k in keys_not_1])
    val_true = np.array([probs[k] for k in keys_not_1])
    val_pred_node = np.array([node_edge_prob[k] for k in keys_not_1])
    val_pred_edge = np.array([edge_edge_prob[k] for k in keys_not_1])
    
    
    # print("l2 rand distance: ", 
    #     np.linalg.norm((np.random.uniform(size=len(node_edge_prob)) - val_true)**2))
    print("l1 rand distance: ", np.linalg.norm(val_rand - val_true, ord=1))
    print("l1 distance (node vs true): ", np.linalg.norm(val_pred_node - val_true, ord=1))
    print("l1 distance (edge vs true): ", np.linalg.norm(val_pred_edge - val_true, ord=1))

    print("l2 rand distance: ", np.linalg.norm(val_rand - val_true))
    print("l2 distance (node vs true): ", np.linalg.norm(val_pred_node - val_true))
    print("l2 distance (edge vs true): ", np.linalg.norm(val_pred_edge - val_true))

    # TODO: plot prob difference distribution by node degrees
    prob_pred = edge_edge_prob
    degree_avgProb_in = {}
    degree_avgProb_out = {}
    for u_node in sina_network.Nodes():
        u = u_node.GetId()
        # ## averaging out-links
        # d_out = u_node.GetOutDeg()
        # if d_out > 0:
        #     if(d_out not in degree_avgProb_out):
        #         degree_avgProb_out[d_out] = []
        #     avg_out = 0
        #     for idx in range(d_out):
        #         v = u_node.GetOutNId(idx)
        #         avg_out += (abs(prob_pred[(u,v)] - probs[(u,v)])/probs[(u,v)])
        #     degree_avgProb_out[d_out].append(avg_out/d_out)

        ## averaging in-links
        d_in = u_node.GetInDeg()
        if d_in > 0:
            if(d_in not in degree_avgProb_in):
                degree_avgProb_in[d_in] = []
            avg_in = 0
            for idx in range(d_in):
                v = u_node.GetInNId(idx)
                # avg_in += (abs(prob_pred[(v,u)] - probs[(v,u)])/probs[(v,u)])
                avg_in += (abs(prob_pred[(v,u)] - probs[(v,u)]))
                # print (prob_pred[(v,u)], probs[(v,u)])
            degree_avgProb_in[d_in].append(avg_in)
            # if (avg_in/d_in) > 1:
            #     print (avg_in/d_in)

    # keys_out = degree_avgProb_out.keys()
    # keys_out.sort()
    # avg_out = [sum(degree_avgProb_out[k])/len(degree_avgProb_out[k]) for k in keys_out]
    keys_in = degree_avgProb_in.keys()
    keys_in.sort()
    avg_in = [sum(degree_avgProb_in[k])/len(degree_avgProb_in[k]) for k in keys_in]

    # for k in keys_in:
    #     if sum(degree_avgProb_in[k])/len(degree_avgProb_in[k]) > 5:
    #         print (sum(degree_avgProb_in[k])/len(degree_avgProb_in[k]))
    #         print (degree_avgProb_in[k])




# plt.loglog(keys_in, avg_in,'.',color = 'r', label="In-Link")
# plt.xlabel('log(Degree)')
# plt.ylabel('log(Average Percentage Error)')
keys_in_clipped = []
avg_in_clipped = []
for k in range(len(keys_in)):
    if avg_in[k] < 10 and avg_in[k] > 0:
        # print (avg_in[k])
        keys_in_clipped.append(keys_in[k])
        avg_in_clipped.append(avg_in[k])

avg_in_clipped_log = [math.log(x,0.1) for x in avg_in_clipped] 
plt.plot(keys_in_clipped, avg_in_clipped_log,'.',color = 'r', label="In-Link")
# plt.semilogy(keys_in_clipped, avg_in_clipped,'.',color = 'r', label="In-Link")
plt.xlabel('Degree')
plt.ylabel('Average Percentage Error')
plt.title('Error Distribution(Edge Model)')

# plt.plot(keys_out, avg_out,'.', color = 'b', label="Out-Link")
# plt.legend(loc='upper right', frameon=False)
plt.show()
   

    




