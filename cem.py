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
            self.X[nid] = np.clip(np.random.normal(mu, sig), 0., 1.)

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

    @staticmethod
    def sample_probability(network, mu, sigma):

        # First generate preference indicator for each node
        preference_dict = collections.defaultdict(float)

        inDegV, indeg = snap.TIntPrV(), collections.defaultdict(int)
        snap.GetNodeInDegV(network, inDegV)
        for item in indeg:
            nid, deg = item.GetVal1(), item.GetVal2()
            indeg[nid] = deg

        for nid, deg in indeg.items():
            if deg == 0:
                continue
            preference_dict[nid] = np.clip(np.random.normal(mu[nid], sig[nid]), 0., 1.)

        # Next sample probabilities of edges using preference indicators
        # For NodeStat, X represents likelihood of retweeting other users;
        # need to calculate probabilities again with edges
        outDegV, prob_dict = snap.TIntPrV(), collections.defaultdict(float)
        snap.GetNodeOutDegV(self._graph, outDegV)

        for item in outDegV:
            nid, deg = item.GetVal1(), item.GetVal2()
            node = self._graph.GetNI(nid)

            value = np.array([self.X[node.GetOutNId(i)] for i in range(deg)],
                dtype=np.float32)

            prob = value / value.sum()

            # Note here the order is out link
            for i in range(deg):
                neighbor = node.GetOutNId(i)
                prob_dict[(neighbor, nid)] = prob[i]

        return prob_dict


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
                    [mu[(node.GetInNId(i), nid)] for i in range(deg)],
                    [sig[(node.GetInNId(i), nid)] for i in range(deg)]
                ), 0., 1.)
            norm_new_p = new_p / new_p.sum()

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

    @staticmethod
    def sample_probability(network, mu, sigma):
        """
        To sample probability for edges, same procedure as update
        """
        inDegV, indeg = snap.TIntPrV(), collections.defaultdict(int)
        snap.GetNodeInDegV(network, inDegV)
        for item in indeg:
            nid, deg = item.GetVal1(), item.GetVal2()
            indeg[nid] = deg

        prob_dict = collections.defaultdict(float)

        for nid, deg in indeg.items():
            if deg == 0:
                continue

            node = network.GetNI(nid)

            # Re-sample from new mu and sigma, and normalize
            new_p = np.clip(
                np.random.normal(
                    [mu[(node.GetInNId(i), nid)] for i in range(deg)],
                    [sig[(node.GetInNId(i), nid)] for i in range(deg)]
                ), 0., 1.)
            norm_new_p = new_p / new_p.sum()

            # Prob normalized version of edges
            for i in range(deg):
                neighbor = node.GetInNId(i)
                prob_dict[(neighbor, nid)] = norm_new_p[i]

        return prob_dict


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
        self.edge_result = './data/edge_res_small_small.pkl'
        self.node_result = './data/node_res_small_small.pkl'
    
        self.num_examples = 32
        self.num_top = 6
        self.epsilon = 1e-5
        self.sigma = 0.01
        self.mu = 0.

        self.max_iter = 200
        self.avg_sig = 0.1

