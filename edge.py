
import snap
import logging, sys
import numpy as np
import pickle
import collections

from matplotlib import pyplot as plt
from cem import EdgeStat, Config


if __name__ == '__main__':

    logging.basicConfig(stream=sys.stdout, 
        level=logging.INFO, format='%(asctime)s %(message)s')

    # load config
    conf, sigs = Config(), []
    sina_network = snap.LoadEdgeList(snap.PNGraph, conf.network_file)

    # TODO: get path_dict from qiwen's code
    with open(conf.path_dict, 'rb') as f:
        path_dict = pickle.load(f)
    stats = [EdgeStat(sina_network, conf.mu, conf.sigma_ratio) for _ in range(conf.num_examples)]

    logger = open(conf.edge_log, 'w')
    t, avg_sig = 0, 1e3

    while t < conf.max_iter:

        logging.info('Iteration {}....'.format(t))
        scores = [(stats[i].evaluate_assignment(path_dict), i)\
            for i in range(conf.num_examples)]
        top_scores = sorted(scores, reverse=True, key=lambda x: x[0])[:conf.num_top]
        top_m = [i for (s, i) in top_scores]
        logging.info('Top scores: {}'.format(top_scores))
        new_MU, new_SIG = EdgeStat.update_stat([stats[i].X for i in top_m])

        # pdate_mu/sigma by refitting mu, sigma on top_m
        for i in range(conf.num_examples):
            stats[i].update_network(new_MU, new_SIG)

        avg_sig = np.mean(new_SIG.values())
        logging.info('Average sigma this iter: {}'.format(avg_sig))
        logger.write('{}\n'.format(avg_sig))
        sigs.append(avg_sig)
        t += 1

        if avg_sig < conf.epsilon:

            logging.info('Writing results into {}...'.format(conf.edge_result))

            # For edgeStat, X represents edge probability of retweeting its neighbor
            with open(conf.edge_result, 'wb') as f:
                # mu, sigma; gaussian distribution of the probability
                # meaning: followee, follower, prob follower retweet followee
                pickle.dump({'mu': new_MU, 'sigma': new_SIG}, f, protocol=pickle.HIGHEST_PROTOCOL)

            logger.close()
            plt.plot(sigs, 'r-')
            plt.savefig(conf.edge_plot)
            break
    