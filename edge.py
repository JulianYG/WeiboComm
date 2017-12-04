
import snap
import numpy as np
import pickle
import collections

from cem import EdgeStat, Config


if __name__ == '__main__':

    # load config
    conf = Config()    
    sina_network = snap.LoadEdgeList(snap.PNGraph, conf.network_file)

    # TODO: get path_dict from qiwen's code
    with open(conf.path_dict, 'rb') as f:
        path_dict = pickle.load(f)
    stats = [EdgeStat(sina_network, conf.mu, conf.sigma) for _ in range(conf.num_examples)]

    max_mu_update, max_sig_update = 1e3, 1e3

    t, old_mu, old_sig = 0, collections.defaultdict(lambda: conf.mu, {}), collections.defaultdict(lambda: conf.sigma, {})
    while t < conf.max_iter:

        print('Iteration {}....'.format(t))
        scores = [(stats[i].evaluate_assignment(path_dict), i)\
            for i in range(conf.num_examples)]
        top_scores = sorted(scores, reverse=True, key=lambda x: x[0])[:conf.num_top]
        top_m = [i for (s, i) in top_scores]
        print('Top scores: {}'.format(top_scores))
        new_MU, new_SIG = EdgeStat.update_stat([stats[i].X for i in top_m])

        # Act as a placeholder
        if t == 0:
            old_mu = conf.mu * np.ones(len(new_MU), dtype=np.float32)
            old_sig = conf.sigma * np.ones(len(new_SIG), dtype=np.float32)

        # pdate_mu/sigma by refitting mu, sigma on top_m
        for i in range(conf.num_examples):
            stats[i].update_network(new_MU, new_SIG)

        max_mu_update = np.max(np.abs(np.array(new_MU.values()) - old_mu))
        max_sig_update = np.max(np.abs(np.array(new_SIG.values()) - old_sig))
        old_mu = np.array(new_MU.values())
        old_sig = np.array(new_SIG.values())

        print('Maximum mu, sigma update deviation: {}, {}'.format(max_mu_update, 
            max_sig_update))
        t += 1

        if t == 1 or max_mu_update < conf.epsilon and max_sig_update < conf.epsilon:

            print('Writing results into {}...'.format(conf.edge_result))

            # For edgeStat, X represents edge probability of retweeting its neighbor
            with open(conf.edge_result, 'wb') as f:
                # mu, sigma; gaussian distribution of the probability
                # meaning: followee, follower, prob follower retweet followee
                pickle.dump({'mu': new_MU, 'sigma': new_SIG}, f, protocol=pickle.HIGHEST_PROTOCOL)

            break
    