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
        snap.GetNodeOutDegV(network_g, OutDegV)
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

    def evaluate_assignment(self, path_dict):
        """ For each assignment of edge probabilities (self.X), assign a score to how well
        this assignments is according to some criterion we learnt from retweet graph. All these
        heuristics learnt from retweet graph is stored in path_dict

        path_dict: a dictionary with key being a pair of reachable nodes in retweet graph,
            key being a list of tuples, where each tuple represents a path from the two nodes, and the
            corresponding likelihood of that path.
            eg. {(A,X): [([A, B, C, X], 3), ([A, D, X], -2)]}
        """
        total_score = 0.0
        for pair, v in path_dict:
            pair_score = [] # list of score that has the length of number of paths between the pair
            for path, likelihood_score in v:
                normalized_weight = 1
                for nid in path:
                    normalized_weight *= float(self.X[nid])/(1/self.out_degree_dict[nid])
                pair_score.append(normalized_weight*likelihood_score)

            # add the average of pair_score to total_score
            total_score += sum(pair_score)/float(len(pair_score))
        return total_score



        

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
    # retweet_graph = s.LoadEdgeList(s.PNGraph, conf.retweet_file)

    # TODO: get path_dict from qiwen's code
    path_dict = None

    graph = Graphize(sina_network)

    node_stats = [NodesStats(sina_network, 0, sigma) for _ in conf.num_examples]

    t = 0
    while t < max_iter and conf.avg_sig > conf.epsilon:

        scores = np.array([(node_stats[i].evaluate_assignment(path_dict)
            , i) for i in range(conf.num_examples)], dtype=np.float32)
        top_m = [i for (s, i) in sorted(scores)[:conf.num_top]]
        
        new_MU, new_SIG = get_new_MU_SIG([node_stats[i] for i in top_m])
        # pdate_mu/sigma by refitting mu, sigma on top_m
        for i in range(conf.num_examples):
            node_stats[i].update(new_MU, new_SIG)

        conf.avg_sig = np.mean(new_SIG.values())

      

