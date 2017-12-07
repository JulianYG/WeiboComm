# follow_graph: get number of followees 

# Dict: sigma, mu for each node
# retweet_graph: time_stamps

import snap
import numpy as np
import pickle
import collections

np.random.seed(42)


class Stat(object):

    def __init__(self, network_g, initial_mu, sigma_ratio):

        self._graph = network_g

        # The probability dictionary
        self.X = collections.defaultdict(float)

        # The popularity dictionary
        self.P = collections.defaultdict(float)
        self._initialize(initial_mu, sigma_ratio)

    def _initalize(self, mu, sigma):
        return NotImplemented

    @staticmethod
    def sample_probability(network, mu, sigma):
        return NotImplemented

    @staticmethod
    def update_stat(stats_lst):
        """ Given a list of NodeStats objects, 
        return new dictionaries mapping from nid to new mu and sig values.
        """
        mu, sigma = collections.defaultdict(float), collections.defaultdict(float)
        samples = collections.defaultdict(list)
        for stat in stats_lst:
            for k in stat:
                samples[k].append(stat[k])

        for k in stats_lst[0]:
            mu[k] = np.mean(samples[k])
            sigma[k] = np.std(samples[k])
        return mu, sigma

    def update_network(self, mu, sigma):
        self.P, self.X = self.sample_probability(self._graph, mu, sigma)

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
        for pair in path_dict:
            v = path_dict[pair]
            
            # list of score that has the length of number of paths between the pair
            pair_score = [] 

            for path, ms, cns, crs in v:
                normalized_weight = 1.
                for i in range(len(path) - 1):
                    # Scale by the in degree of the node to normalize
                    normalized_weight *= self.X[(path[i], 
                        path[i + 1])] * self._graph.GetNI(path[i + 1]).GetInDeg()
             
                score = normalized_weight * \
                    (missing_score * ms + conflict_score * cns + correct_score * crs)
                pair_score.append(score)

            # add the average of pair_score to total_score
            total_score += sum(pair_score) / len(pair_score)
        return total_score


class NodeStat(Stat):
    """ 
    Represent one set of retweet probabilities for every node
    """
    def __init__(self, network_g, initial_mu, sigma_ratio):

        # represents the popularity of each node
        super(NodeStat, self).__init__(network_g, initial_mu, sigma_ratio)

    def _initialize(self, mu, sigma_ratio):
        """
        NodeStat uses out links to initalize popularity, 
        then sample edge probabilities using in links
        """
        outdeg = snap.TIntPrV()
        snap.GetNodeOutDegV(self._graph, outdeg)

        max_out_nid = snap.GetMxOutDegNId(self._graph)
        max_out_deg = self._graph.GetNI(max_out_nid).GetOutDeg()

        for item in outdeg:
            nid, deg = item.GetVal1(), float(item.GetVal2())
            
            # Initialized according to scaled number of followers
            self.P[nid] = np.random.normal(
                deg / max_out_deg + mu, deg / max_out_deg * sigma_ratio)

        self.X = NodeStat._compute_prob(self._graph, self.P)

    @staticmethod
    def _compute_prob(network, P):

        # Next sample probabilities of edges using preference indicators
        # For NodeStat, X represents likelihood of retweeting other users;
        # need to calculate probabilities again with edges
        prob_dict = collections.defaultdict(float)

        for nid in P:
            deg = network.GetNI(nid).GetInDeg()
            if deg == 0:
                continue
            node = network.GetNI(nid)

            prob = np.clip([P[node.GetInNId(i)] for i in range(deg)],
                0., None)

            # Handle corner cases
            if prob.sum() == 0:
                prob = np.ones(deg, dtype=np.float32)
           
            prob /= prob.sum()

            # Note here the order is out link
            for i in range(deg):
                neighbor = node.GetInNId(i)
                prob_dict[(neighbor, nid)] = prob[i]

        return prob_dict

    @staticmethod
    def sample_probability(network, mu, sigma):

        # First generate popularity indicator for each node
        popularity_dict = collections.defaultdict(float)
        for nid in mu:
            popularity_dict[nid] = np.random.normal(mu[nid], sigma[nid])

        # Next compute probabilities
        return popularity_dict, NodeStat._compute_prob(network, popularity_dict)


class EdgeStat(Stat):

    def __init__(self, network_g, initial_mu, sigma_ratio):

        super(EdgeStat, self).__init__(network_g, initial_mu, sigma_ratio)

    def _initialize(self, mu, sigma_ratio):

        indeg = snap.TIntPrV()
        snap.GetNodeInDegV(self._graph, indeg)

        for item in indeg:
            nid, deg = item.GetVal1(), item.GetVal2()
            if deg == 0:
                continue

            node = self._graph.GetNI(nid)

            # Sample a random probability for each in link
            p = np.clip(np.random.normal(
                np.ones(deg, dtype=np.float32) / deg, 
                sigma_ratio / np.ones(deg)), 0., 1.)

            # Handle corner cases
            if p.sum() == 0:
                p = np.ones(deg, dtype=np.float32)
            
            p /= p.sum()

            for i in range(deg):
                neighbor = node.GetInNId(i)
                self.X[(neighbor, nid)] = p[i]

    @staticmethod
    def sample_probability(network, mu, sigma):
        """
        To sample probability for edges, same procedure as update_network
        """
        prob_dict = collections.defaultdict(float)

        # Note each nid and all it's followees should only be updated once
        # during each sampling, need to be very careful!
        visited = set()

        for _, nid in mu:

            deg = network.GetNI(nid).GetInDeg()
            if deg == 0 or nid in visited:
                continue

            node = network.GetNI(nid)

            # Re-sample from new mu and sigma, and normalize
            new_p = np.clip(
                np.random.normal(
                    [mu[(node.GetInNId(i), nid)] for i in range(deg)],
                    [sigma[(node.GetInNId(i), nid)] for i in range(deg)]
                ), 0., 1.)

            if new_p.sum() == 0:
                new_p = np.ones(deg, dtype=np.float32)

            new_p /= new_p.sum()

            # Prob normalized version of edges
            for i in range(deg):
                neighbor = node.GetInNId(i)
                prob_dict[(neighbor, nid)] = new_p[i]
            visited.add(nid)

        return {}, prob_dict


class Config:
    def __init__(self):
        
        self.network_file = './data/network_graph.txt'
        self.retweet_file = './data/total.txt'
        self.path_dict = './data/path_10_25.pkl'
        self.edge_result = './data/edge_res.pkl'
        self.node_result = './data/node_res.pkl'
        self.max_path_len = 25

        self.node_log = './log/node_full.txt'
        self.node_plot = './log/node_full.png'
        self.edge_log = './log/edge_full.txt'
        self.edge_plot = './log/edge_full.png'
    
        self.num_examples = 32
        self.num_top = 5
        self.epsilon = 1e-5
        self.sigma_ratio = 0.25
        self.mu = 0.

        self.max_iter = 500

