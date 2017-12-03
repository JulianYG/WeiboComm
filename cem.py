# follow_graph: get number of followees 

# Dict: sigma, mu for each node
# retweet_graph: time_stamps

import snap
import numpy as np
import pickle
import collections

np.random.seed(42)


class NodeStat(object):
    """ 
    Represent one set of retweet probabilities for every node
    """
    def __init__(self, network_g, initial_mu, initial_sig):

        self._mu = initial_mu
        self._sig = initial_sig
        self._graph = network_g

        self.in_degree_dict = {}
        self.X = {}

        inDegV = snap.TIntPrV()
        snap.GetNodeInDegV(network_g, inDegV)
        
        self._assign(inDegV)

    def _assign(self, indeg):

        for item in indeg:
            nid, deg = item.GetVal1(), item.GetVal2()
            self.in_degree_dict[nid] = deg

            if deg == 0:
                continue

            p = np.random.normal(1. / deg + self._mu, self._sig)
            self.X[nid] = np.clip(p, 0., 1.)

    def update(self, new_MU, new_SIG):
        """ 
        Update X (retweet probabilities) for each node.
        new_MU: dictionary of node_id to corresponding new mu values
        new_SIG: dictionary of node_id to corresponding new sigma values
        """
        for nid, deg in self.in_degree_dict.items():
            if deg == 0:
                continue
            mu, sig = new_MU[nid], new_SIG[nid]
            self.X[nid] = np.random.normal(mu, sig)

    def evaluate_assignment(self, path_dict, 
        missing_score=0, 
        conflict_score=-1, 
        correct_score=1):
        """ 
        For each assignment of edge probabilities (self.X), assign a score to how well
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
                for nid in path[1:]:
                    normalized_weight *= self.X[nid] * self.in_degree_dict[nid]
                
                score = normalized_weight * \
                    (missing_score * ms + conflict_score * cns + correct_score * crs)

                pair_score.append(score)

            # add the average of pair_score to total_score
            total_score += sum(pair_score) / float(len(pair_score))
        return total_score


class EdgeStat(object):

    def __init__(self, network_g, initial_mu, initial_sig):

        self._mu = initial_mu
        self._sig = initial_sig
        self._graph = network_g

        self.in_degree_dict = collections.defaultdict(dict)
        self.X = collections.defaultdict(dict)

        inDegV = snap.TIntPrV()
        snap.GetNodeInDegV(network_g, inDegV)
        
        self._assign(inDegV)

    def _assign(self, indeg):

        for item in indeg:
            nid, deg = item.GetVal1(), item.GetVal2()
            self.in_degree_dict[nid] = deg
            if deg == 0:
                continue

            node = self._graph.GetNI(nid)

            # Sample a random probability for each in link
            p = np.clip(np.random.normal(
                1. / deg + np.ones(deg) * self._mu, 
                np.ones(deg) * self._sig), 0., 1.)
            norm_p = p / p.sum()

            for i in range(deg):
                neighbor = node.GetInNId(i)
                self.X[(neighbor, nid)] = norm_p[i]

    def update(self, mu, sig):

        for nid, deg in self.in_degree_dict.items():
            if deg == 0:
                continue

            node = self._graph.GetNI(nid)

            # Re-sample from new mu and sigma, and normalize
            new_p = np.clip(
                np.random.normal(
                    [mu[(nid, node.GetInNId(i))] for i in range(deg)],
                    [sig[(nid, node.GetInNId(i))] for i in range(deg)]
                ), 0., 1.)
            norm_new_p = new_p / p.sum()

            # Prob normalized version of edges
            for i in range(deg):
                neighbor = node.GetInNId(i)
                self.X[(neighbor, nid)] = norm_new_p[i]

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

                for i in range(len(path) - 1):

                    normalized_weight *= self.X[(path[i], 
                        path[i + 1])] * self.in_degree_dict[path[i + 1]]
             
                score = normalized_weight * \
                    (missing_score * ms + conflict_score * cns + correct_score * crs)

                pair_score.append(score)

            # add the average of pair_score to total_score
            total_score += sum(pair_score) / float(len(pair_score))
        return total_score


def get_new_stats(stats_lst):
    """ Given a list of NodeStats objects, 
    return new dictionaries mapping from nid to new mu and sig values.
    """
    keys = stats_lst[0].X.keys()
    new_M, new_SIG = {}, {}
    samples = collections.defaultdict(list)
    for stat in stats_lst:
        X = stat.X
        for k, x in X.items():
            samples[k].append(x)

    for k in keys:
        new_M[k] = np.mean(samples[k])
        new_SIG[k] = np.std(samples[k])
    return new_M, new_SIG


class Config:
    def __init__(self):
        
        self.network_file = './data/network_graph_small_small.txt'
        self.retweet_file = './data/total.txt'
        self.path_dict = './data/path.pkl'
        self.result = './data/edge_res.pkl'

        self.stat = EdgeStat
    
        self.num_examples = 128
        self.num_top = 16
        self.epsilon = 1e-3
        self.sigma = 0.01

        self.max_iter = 1000
        self.avg_sig = 0.1


if __name__ == '__main__':
    
    # load config
    conf = Config()
    # prev_scores, scores = np.ones(conf.num_examples), np.zeros(conf.num_examples)
    
    sina_network = snap.LoadEdgeList(snap.PNGraph, conf.network_file)

    # TODO: get path_dict from qiwen's code
    with open(conf.path_dict, 'rb') as f:
        path_dict = pickle.load(f)
    stats = [conf.stat(sina_network, 0, conf.sigma) for _ in range(conf.num_examples)]

    t = 0
    while t < conf.max_iter and conf.avg_sig > conf.epsilon:

        print('Iteration {}....'.format(t))
        scores = [(stats[i].evaluate_assignment(path_dict), i)\
            for i in range(conf.num_examples)]
        top_m = [i for (s, i) in sorted(scores)[:conf.num_top]]
        new_MU, new_SIG = get_new_stats([stats[i] for i in top_m])

        # pdate_mu/sigma by refitting mu, sigma on top_m
        for i in range(conf.num_examples):
            stats[i].update(new_MU, new_SIG)

        conf.avg_sig = np.mean(new_SIG.values())
        print('Average sigma this iter: {}'.format(conf.avg_sig))
        t += 1

    print('Writing results into {}...'.format(conf.result))
    with open(conf.result, 'wb') as f:
        pickle.dump(stats.X, f, protocol=pickle.HIGHEST_PROTOCOL)

