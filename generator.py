
import snap
import math
import random
from datetime import datetime, timedelta

""" In snap, there are two models for generating directed scale-free graph (following power laws)
GenBaraHierar() 
https://snap.stanford.edu/snappy/doc/reference/GenBaraHierar.html

GenCopyModel
https://snap.stanford.edu/snappy/doc/reference/GenCopyModel.html

We can use both and tune parameters to closely match weibo data
"""

class gen_config:
    def __init__(self):
        # Param for using copy model of follower graph
        self.num_nodes = 10000
        self.copy_model_beta = 0.6

        # Param for using BaraHierar model of follower graph
        self.levels = 3

        # Param for generating retweet data
        self.num_retweet = 1000
        self.N_0 = 200 # number of tweets with path length of 2
        self.exp_decay_const = math.log(1+float(self.N_0)/self.num_retweet)
        self.min_retweet_time = datetime.strptime("2016/01/01 00:00", "%Y/%m/%d %H:%M")
        self.max_retweet_time = datetime.strptime("2017/01/01 00:00", "%Y/%m/%d %H:%M")


conf = gen_config()

followers_graph = snap.GenCopyModel(conf.num_nodes, conf.copy_model_beta, snap.TRnd())
# followers_graph = snap.GenBaraHierar(snap.PNGraph, conf.levels, True)


retweet_data = [] # list of ((path), (retweet_time))
num_tweets, target_len = conf.N_0, 2
Rnd = snap.TRnd(42)
Rnd.Randomize()
time_range = int((conf.max_retweet_time - conf.min_retweet_time).total_seconds()/60)

while num_tweets > 1:
    generated = 0
    # print generated, num_tweets
    while generated < num_tweets:
        path = [followers_graph.GetRndNId(Rnd)]
        while len(path) < target_len:
            nbrs = [i for i in followers_graph.GetNI(path[-1]).GetOutEdges()]
            # print nbrs
            if not nbrs: 
                break
            path.append(random.choice(nbrs))
            

        if len(path) == target_len:
            random_minutes = random.sample(range(time_range), target_len-1)
            retweet_time = [(conf.min_retweet_time + 
                timedelta(minutes=x)).strftime('%Y/%m/%d %H:%M') for x in random_minutes]

            retweet_data.append((path, retweet_time))
            generated += 1

    # decay number of tweets to generate and increase path
    num_tweets *= math.exp(-conf.exp_decay_const)
    target_len += 1


# TODO: dump retweet_data and follower_graph to pickle




