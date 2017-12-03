
import snap
import numpy as np
import pickle
import collections

from cem import NodeStat, Config, get_new_stats

if __name__ == '__main__':

    # load config
    conf = Config()    
    sina_network = snap.LoadEdgeList(snap.PNGraph, conf.network_file)

    # TODO: get path_dict from qiwen's code
    with open(conf.path_dict, 'rb') as f:
        path_dict = pickle.load(f)
    stats = [NodeStat(sina_network, conf.mu, conf.sigma) for _ in range(conf.num_examples)]

    t = 0
    while t < conf.max_iter:

        print('Iteration {}....'.format(t))
        scores = [(stats[i].evaluate_assignment(path_dict), i)\
            for i in range(conf.num_examples)]
        top_scores = sorted(scores, reverse=True, key=lambda x: x[0])[:conf.num_top]
        top_m = [i for (s, i) in top_scores]
        print('Top scores: {}'.format(top_scores))
        new_MU, new_SIG = get_new_stats([stats[i] for i in top_m])

        # pdate_mu/sigma by refitting mu, sigma on top_m
        for i in range(conf.num_examples):
            stats[i].update(new_MU, new_SIG)

        conf.avg_sig = np.mean(new_SIG.values())
        print('Average sigma this iter: {}'.format(conf.avg_sig))
        t += 1

        if conf.avg_sig < conf.epsilon:

            # For NodeStat, X represents likelihood of retweeting other users;
            # need to calculate probabilities again with edges
            outDegV = snap.TIntPrV()
            snap.GetNodeOutDegV(sina_network, outDegV)
            prob_dict = {}

            for item in outDegV:
                nid, deg = item.GetVal1(), item.GetVal2()
                node = sina_network.GetNI(nid)

                value = np.array([stats[top_m[0]].X[node.GetOutNId(i)] for i in range(deg)],
                    dtype=np.float32)

                prob = value / value.sum()

                # Note here the order is out link
                for i in range(deg):
                    neighbor = node.GetOutNId(i)
                    prob_dict[(nid, neighbor)] = prob[i]

            print('Writing results into {}...'.format(conf.result))
            with open(conf.result, 'wb') as f:
                pickle.dump(prob_dict, f, protocol=pickle.HIGHEST_PROTOCOL)



