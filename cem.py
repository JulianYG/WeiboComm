# follow_graph: get number of followees 

# Dict: sigma, mu for each node
# retweet_graph: time_stamps

import snap
import numpy as np
import pickle
import collections

np.random.seed(42)


class Stat(object):

    def __init__(self, sid, network_g, initial_mu, sigma_ratio):

        self._graph = network_g
        self.prob = 'prob_{}'.format(sid)
        self.sid = sid
        self._initialize(initial_mu, sigma_ratio)
        print('Initialization {} finished.'.format(sid))

    def _initalize(self, mu, sigma):
        return NotImplemented

    def get_attr(self, element_id, attr):
        return NotImplemented

    @staticmethod
    def update_stat(graph, stats_lst):
        """ 
        Given a list of NodeStats objects, 
        return new dictionaries mapping from nid to new mu and sig values.
        """
        return NotImplemented

    @staticmethod
    def sample_probability(network, sid, mu, sigma):
        return NotImplemented

    @staticmethod
    def get_prob_dict(network, sid):

        prob_dict = collections.defaultdict(float)
        for edge in network.Edges():
            key = (edge.GetSrcNId(), edge.GetDstNId())
            prob_dict[key] = network.GetFltAttrDatE(edge, 'prob_{}'.format(sid))

        return prob_dict

    def update_network(self, mu, sigma):
        self.sample_probability(self._graph, self.sid, mu, sigma)

    def evaluate_assignment(self, path_dict, 
        missing_score=0, 
        conflict_score=-1, 
        correct_score=1):
        """ 
        For each assignment of edge probabilities, assign a score to how well
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

                    edge = self._graph.GetEI(path[i], path[i + 1])

                    # Scale by the in degree of the node to normalize
                    normalized_weight *= self._graph.GetFltAttrDatE(edge, self.prob)\
                        * self._graph.GetNI(path[i + 1]).GetInDeg()
             
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
    def __init__(self, sid, network_g, initial_mu, sigma_ratio):

        # represents the popularity of each node
        self.pop = 'pop_{}'.format(sid)
        super(NodeStat, self).__init__(sid, network_g, initial_mu, sigma_ratio)
        
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

            init_pop_mu = deg / max_out_deg + mu
            init_pop_sig = deg / max_out_deg * sigma_ratio

            # Initialized according to scaled number of followers
            init_pop = np.random.normal(init_pop_mu, init_pop_sig)
            self._graph.AddFltAttrDatN(
                nid, init_pop, self.pop)

        NodeStat._compute_prob(self._graph, self.sid)

    def get_attr(self, nid, attr):
        return self._graph.GetFltAttrDatN(nid, attr)

    @staticmethod
    def update_stat(graph, stat_lst):

        mu, sigma = collections.defaultdict(float), collections.defaultdict(float)
        for node in graph.Nodes():
            nid = node.GetId()
            pops = [s.get_attr(nid, 'pop_{}'.format(s.sid)) for s in stat_lst]

            # Assign new gaussian params by data values
            mu[nid] = np.mean(pops)
            sigma[nid] = np.std(pops)

        return mu, sigma

    @staticmethod
    def _compute_prob(network, sid):

        # Next sample probabilities of edges using preference indicators
        # For NodeStat, X represents likelihood of retweeting other users;
        # need to calculate probabilities again with edges
        for node in network.Nodes():

            deg = node.GetInDeg()
            if deg == 0:
                continue

            prob = np.array(
                [network.GetFltAttrDatN(node.GetInNId(i), 
                    'pop_{}'.format(sid)) for i in range(deg)], dtype=np.float32)

            # Handle corner cases
            if prob.sum() == 0 or len(prob) == 1:
                prob = np.ones(deg, dtype=np.float32)
            else:
                prob = (prob - prob.min()) / (prob.max() - prob.min())
            prob /= prob.sum()
        
            # Note here the order is src->dest
            for i in range(deg):
                edge = network.GetEI(node.GetInNId(i), node.GetId())
                network.AddFltAttrDatE(edge, prob[i], 'prob_{}'.format(sid))

    @staticmethod
    def sample_probability(network, sid, mu, sigma):

        # First generate popularity indicator for each node
        for nid in mu:
            network.AddFltAttrDatN(
                network.GetNI(nid), 
                np.random.normal(mu[nid], sigma[nid]), 'pop_{}'.format(sid))

        # Next compute probabilities
        NodeStat._compute_prob(network, sid)


class EdgeStat(Stat):

    def __init__(self, sid, network_g, initial_mu, sigma_ratio):

        super(EdgeStat, self).__init__(sid, network_g, initial_mu, sigma_ratio)

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
                edge = self._graph.GetEI(node.GetInNId(i), node.GetId())
                self._graph.AddFltAttrDatE(edge, p[i], self.prob)

    def get_attr(self, eid, attr):
        return self._graph.GetFltAttrDatE(eid, attr)

    @staticmethod
    def update_stat(graph, stat_lst):

        mu, sigma = collections.defaultdict(float), collections.defaultdict(float)

        for edge in graph.Edges():

            edge_tup = (edge.GetSrcNId(), edge.GetDstNId())
            probs = [s.get_attr(edge, 'prob_{}'.format(s.sid)) for s in stat_lst]

            # Assign new gaussian params by data values
            mu[edge_tup] = np.mean(probs)
            sigma[edge_tup] = np.std(probs)
            
        return mu, sigma

    @staticmethod
    def sample_probability(network, sid, mu, sigma):
        """
        To sample probability for edges, same procedure as update_network
        """

        # Note each nid and all it's followees should only be updated once
        # during each sampling, need to be very careful!
        visited = set()

        for _, nid in mu:
            node = network.GetNI(nid)

            deg = node.GetInDeg()
            if deg == 0 or nid in visited:
                continue

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
                edge = network.GetEI(node.GetInNId(i), nid)
                network.AddFltAttrDatE(edge, new_p[i], 'prob_{}'.format(sid))

            visited.add(nid)


class Config:

    def __init__(self):
        
        self.network_file = './data/network_graph_small_small.txt'
        self.retweet_file = './data/total.txt'
        self.path_dict = './data/path_real.pkl'
        self.edge_result = './data/edge_res_small_small.pkl'
        self.node_result = './data/node_res_small_small.pkl'
        self.ground_truth = './data/graph_probs_small_small_in.pickle'
        self.max_path_len = 25

        self.node_log = './log/node_real.txt'
        self.node_plot = './log/node_real.png'
        self.edge_log = './log/edge_real.txt'
        self.edge_plot = './log/edge_real.png'
    
        self.num_examples = 16
        self.num_top = 5
        self.epsilon = 3e-6
        self.sigma_ratio = 0.25
        self.mu = 0.

        self.max_iter = 500

