
import snap
import numpy as np
import pickle
import collections

from cem import NodeStat, Config

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
        new_MU, new_SIG = NodeStat.update_stat([stats[i].P for i in top_m])

        # pdate_mu/sigma by refitting mu, sigma on top_m
        for i in range(conf.num_examples):
            stats[i].update_network(new_MU, new_SIG)

        conf.avg_sig = np.mean(new_SIG.values())
        print('Average sigma this iter: {}'.format(conf.avg_sig))
        t += 1

        if conf.avg_sig < conf.epsilon:

            print('Writing results into {}...'.format(conf.node_result))
            with open(conf.node_result, 'wb') as f:
                # Format: followee, follower, prob follower retweet followee
                pickle.dump({'mu': new_MU, 'sigma': new_SIG}, f, protocol=pickle.HIGHEST_PROTOCOL)

            break

