# follow_graph: get number of followees 

# Dict: sigma, mu for each node
# retweet_graph: time_stamps

import snap
import numpy as np
import pickle
import collections

np.random.seed(42)


class Config:
    def __init__(self):
        
        self.network_file = './data/network_graph_small_small.txt'
        self.retweet_file = './data/total.txt'
        self.path_dict = './data/path.pkl'
        self.result = './data/res.pkl'
    
        self.num_examples = 128
        self.num_top = 16
        self.epsilon = 1e-3
        self.sigma = 0.01

        self.max_iter = 1000
        self.avg_sig = 0.1


class NodesStats(object):
    """ Represent one set of retweet probabilities for every node
    """
    def __init__(self, network_g, initial_mu, initial_sig):
        inDegV = snap.TIntPrV()
        snap.GetNodeInDegV(network_g, inDegV)
        self.in_degree_dict = {}
        self.X = {}
        for item in inDegV:
            nid, deg = item.GetVal1(), item.GetVal2()
            self.in_degree_dict[nid] = deg
            if deg == 0:
                self.X[nid] = 0
            else:
                p = np.random.normal(1. / deg + initial_mu, initial_sig)
                self.X[nid] = p if p > 0 else 0
        

    def update(self, new_MU, new_SIG):
        """ Update X (retweet probabilities) for each node.
        new_MU: dictionary of node_id to corresponding new mu values
        new_SIG: dictionary of node_id to corresponding new sigma values
        """
        for nid, deg in self.in_degree_dict.items():
            mu, sig = new_MU[nid], new_SIG[nid]
            self.X[nid] = np.random.normal(mu, sig)

    def evaluate_assignment(self, path_dict, 
        missing_score=0, 
        conflict_score=-1, 
        correct_score=1):
        """ For each assignment of edge probabilities (self.X), assign a score to how well
        this assignments is according to some criterion we learnt from retweet graph. All these
        heuristics learnt from retweet graph is stored in path_dict

        path_dict: a dictionary with key being a pair of reachable nodes in retweet graph,
            key being a list of tuples, where each tuple represents a path from the two nodes, and the
            corresponding likelihood of that path.
            eg. {(A,X): [([A, B, C, X], missing, conflict, correct), ([A, D, X], -2)]}
        """
        total_score = 0.0
        for pair, v in path_dict.items():
            pair_score = [] # list of score that has the length of number of paths between the pair
            for path, ms, cns, crs in v:
                normalized_weight = 1.
                for nid in path:
                    normalized_weight *= float(self.X[nid])/(1./self.in_degree_dict[nid])
                
                score = normalized_weight * \
                    (missing_score * ms + conflict_score * cns + correct_score * crs)
                # print(normalized_weight, score)
                pair_score.append(score)

            # add the average of pair_score to total_score
            total_score += sum(pair_score) / float(len(pair_score))
        return total_score


def get_new_MU_SIG(list_of_nodeStats):
    """ Given a list of NodeStats objects, 
    return new dictionaries mapping from nid to new mu and sig values.
    """
    node_ids = list_of_nodeStats[0].X.keys()
    new_MU, new_SIG = {}, {}
    samples = collections.defaultdict(list)
    for nodeStat in list_of_nodeStats:
        X = nodeStat.X
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
    
    sina_network = snap.LoadEdgeList(snap.PNGraph, conf.network_file)

    # TODO: get path_dict from qiwen's code
    with open(conf.path_dict, 'rb') as f:
        path_dict = pickle.load(f)
    node_stats = [NodesStats(sina_network, 0, conf.sigma) for _ in range(conf.num_examples)]

    t = 0
    while t < conf.max_iter and conf.avg_sig > conf.epsilon:

        print('Iteration {}....'.format(t))
        scores = [(node_stats[i].evaluate_assignment(path_dict), i)\
            for i in range(conf.num_examples)]
        top_m = [i for (s, i) in sorted(scores)[:conf.num_top]]
        new_MU, new_SIG = get_new_MU_SIG([node_stats[i] for i in top_m])

        # pdate_mu/sigma by refitting mu, sigma on top_m
        for i in range(conf.num_examples):
            node_stats[i].update(new_MU, new_SIG)

        conf.avg_sig = np.mean(new_SIG.values())
        t += 1

    print('Writing results into {}...'.format(conf.result))
    with open(conf.result, 'wb') as f:
        pickle.dump({'M': new_MU, 'S': new_SIG}, f, protocol=pickle.HIGHEST_PROTOCOL)

