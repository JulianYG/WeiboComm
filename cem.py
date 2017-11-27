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

        self.max_iter = 1000
        self.avg_sig = 0.1



class NodesStats(object):
    """ Represent one set of retweet probabilities for every node
    """
    def __init__(self, network_g, initial_mu, initial_sig):
        outDegV = snap.TIntPrV()
        snap.GetNodeOutDegV(Graph, OutDegV)
        self.out_degree_dict = {}
        self.X = {}
        for item in OutDegV:
            nid, deg = item.GetVal1(), item.GetVal2()
            self.out_degree_dict[nid] = deg
            self.X[nid] = np.random.normal(1/float(deg) + initial_mu, initial_sig)
        

    
    def update(self, new_MU, new_SIG):
        """ Update X (retweet probabilities) for each node.
        new_MU: dictionary of node_id to corresponding new mu values
        new_SIG: dictionary of node_id to corresponding new sigma values
        """
        for nid, deg in self.out_degree_dict.items():
            mu, sig = new_MU[nid], new_SIG[nid]
            self.X[nid] = np.random.normal(mu, sig)

    def getX(self):
        return self.X


class Graphize(object):


    def __init__(self, network_g, ):
        self._network = network_g

    def _find_path(node_stat, ):
    	pass

    def evaluate(node_stat, retweet_g):
    	
    	pass
        

def get_new_MU_SIG(list_of_nodeStats):
    """ Given a list of NodeStats objects, 
    return new dictionaries mapping from nid to new mu and sig values.
    """
    node_ids = list_of_nodeStats[0].getX().keys()
    new_MU, new_SIG = {}, {}
    samples = collections.defaultdict(list)
    for nodeStat in list_of_nodeStats:
        X = nodeStat.getX()
        for nid, x in X.items():
            samples[nid].append(x)

    for nid in node_ids:
        new_MU[nid] = np.mean(samples[nid])
        new_SIG[nid] = np.std(samples[nid])
    return new_MU, new_SIG


if __name__ == '__main__':
    
    # load config
    conf = Config()
    # prev_scores, scores = np.ones(conf.num_examples), np.zeros(conf.num_examples)
    
    sina_network = s.LoadEdgeList(s.PNGraph, conf.network_file)
    retweet_graph = s.LoadEdgeList(s.PNGraph, conf.retweet_file)

    graph = Graphize(sina_network)

    node_stats = [NodesStats(sina_network, 0, sigma) for _ in conf.num_examples]

    t = 0
    while t < max_iter and conf.avg_sig > conf.epsilon:

        scores = np.array([(graph.evaluate(node_stats[i], 
        	retweet_graph), i) for i in range(conf.num_examples)], dtype=np.float32)
        top_m = [i for (s, i) in sorted(scores)[:conf.num_top]]
        
        new_MU, new_SIG = get_new_MU_SIG([node_stats[i] for i in top_m])
        # pdate_mu/sigma by refitting mu, sigma on top_m
        for i in range(conf.num_examples):
            node_stats[i].update(new_MU, new_SIG)

        conf.avg_sig = np.mean(new_SIG.values())

      

