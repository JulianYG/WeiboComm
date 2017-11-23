

# follow_graph: get number of followees 

# Dict: sigma, mu for each node

# retweet_graph: time_stamps

import snap as s 
import numpy as np 

np.random.seed(42)

class Config:
    
    def __init__(self):
        
        self.network_file = './data/network_graph_small.txt'
        self.retweet_file = './data/network_retweet.txt'
    
        self.num_examples = 15
        self.num_top = 4
        self.epsilon = 1e-3
        

class NodesStats(object):
    def __init__(self, node_ids, mu, sig):
        self.node_ids = node_ids
        self.mu = mu
        self.sig = sig
        self.nodes_mu = {}
        self.nodes_sig = {}
    
    def sample(self):
        for id in node_ids:
            self.node_mu `


class Graphize(object):


    def __init__(self, network_g, ):
        self._network = network_g

    def _find_path(node_stat, ):
    	pass

    def evaluate(node_stat, retweet_g):
    	
    	pass
        

if __name__ == '__main__':
    
    # load config
    conf = Config()
    prev_scores, scores = np.ones(conf.num_examples), np.zeros(conf.num_examples)
    
    sina_network = s.LoadEdgeList(s.PNGraph, conf.sina_network)
    retweet_graph = s.LoadEdgeList(s.PNGraph, conf.retweet_file)

    graph = Graphize(sina_network)
    node_stats = NodesStats(sina_network, conf.num_examples)
    
    while np.sum((scores - prev_scores) ** 2) > conf.epsilon:

        scores = np.array([(graph.evaluate(node_stats, 
        	retweet_graph), i) for i in range(conf.num_examples)], dtype=np.float32)
        top_m = [i for (s, i) in sorted(scores)[:conf.num_top]]
        
        # pdate_mu/sigma by refitting mu, sigma on top_m
        node_stat.update_param(top_m)

      

