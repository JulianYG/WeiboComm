import snap
import random
import math
import numpy as np
from datetime import datetime, timedelta
import pickle

min_retweet_time = datetime.strptime("2016-01-01-00:00:00", "%Y-%m-%d-%H:%M:%S")
max_retweet_time = datetime.strptime("2016-05-01-00:00:00", "%Y-%m-%d-%H:%M:%S")


def assign_prob(graph):
    probs = {}
    for node_v in graph.Nodes():
        in_d = node_v.GetInDeg()
        r = np.random.uniform(size = in_d)
        p = r/r.sum()
        # r = [random.uniform(0,1) for i in range(out_d)]
        # sum_ = sum(r)
        # p = [i/sum_ for i in r]
        v = node_v.GetId()
        for idx in range(node_v.GetInDeg()):
            u = node_v.GetInNId(idx)
            probs[(u,v)] = p[idx]
    return probs

def gen_path(graph, probs, path, target_len):
    if len(path) == target_len:
        return
    last_id = path[-1]
    last_node = graph.GetNI(last_id)
    out_d = last_node.GetOutDeg()
    if out_d == 0 :
        return
    coin = random.uniform(0,1)
    cum_prob = 0
    for idx in range(out_d):
        next_id = last_node.GetOutNId(idx)
        p = probs[(last_id, next_id)]
        
        if coin > cum_prob and coin <= cum_prob + p:
            path.append(next_id)
            gen_path(graph, probs, path, target_len)
            break
        cum_prob += p


def assign_time(path_len, min_retweet_time, max_retweet_time):
    #path_len = len(path) 
    time_range = int((max_retweet_time - min_retweet_time).total_seconds())
    # print time_range
    # return " "
    random_seconds = random.sample(range(time_range), path_len)
    random_seconds.sort()
    retweet_time = [(min_retweet_time + timedelta(seconds=x)).strftime('%Y-%m-%d-%H:%M:%S') for x in random_seconds]
    # print retweet_time 
    return retweet_time


def gen_retweet_set(graph, paths_len_num):
    with open("./data/artificial_retweet_1000_in.txt",'w+') as save_file:
        ########assign new probilities
        probs = assign_prob(graph)
        print('saving the whole dictionary...')
        with open("./data/graph_probs_small_small_in.pickle", 'wb') as handle:
            pickle.dump(probs, handle, protocol=pickle.HIGHEST_PROTOCOL)
        ########end assign new probilities

        ########load previous probilities
        # load pickle
        # print('loading probilities...')
        # with open('./data/graph_probs_small_small.pickle', 'rb') as handle:
        #   probs = pickle.load(handle)
          #print paths
        ########end load probilities

        print('generating paths..')
        for target_len, num_paths in paths_len_num:
            print target_len, num_paths
            while num_paths > 0:
                start_node = graph.GetRndNId()
                path = [start_node]
                gen_path(graph, probs, path, target_len)
                if len(path) == target_len:
                    num_paths -= 1
                    if num_paths%100 == 0:
                        print num_paths
                    retweet_time = assign_time(target_len, min_retweet_time, max_retweet_time)
                    save_file.write(str(start_node) + ' ' + str(retweet_time[0]) + '\n')
                    retweet_line = ""
                    for i in range(1, target_len):
                        retweet_line += (str(path[i])+' '+ str(retweet_time[i])+' ')
                    save_file.write(retweet_line + '\n')
    

if __name__ == '__main__':
    following_file = "./data/network_graph_small_small.txt"
    graph = snap.LoadEdgeList(snap.PNGraph, following_file)

    print graph.GetNodes()
    print graph.GetEdges()
    # graph = snap.PNGraph.New()
    # for i in range(4):
    #   graph.AddNode(i+1)
    # graph.AddEdge(1,4)
    # graph.AddEdge(1,2)
    # graph.AddEdge(1,3)
    # graph.AddEdge(2,3)
    # graph.AddEdge(4,3)
    lamb = 0.8
    len_num = []
    for i in range(2, 25):
        num = 1000*lamb*math.exp(-lamb*(i+(np.random.normal(0,0.3))))
        len_num.append((i, int(num)))
    print len_num

    gen_retweet_set(graph, len_num)
   

    # probs =  assign_prob(graph)
    # print probs 

    # path = [1]
    # gen_path(graph, probs, path, 3)
    # print path
    # print assign_time(len(path), min_retweet_time, max_retweet_time)

    # count_4 = 0
    # count_2 = 0
    # count_3 = 0
    # for i in range(1000):
    #     path = [1]
    #     gen_path(graph, probs, path, 2)
    #     # print path
    #     if path[-1] == 2:
    #         count_2 += 1
    #     if path[-1] == 3:
    #         count_3 += 1
    #     if path[-1] == 4:
    #         count_4 += 1
    # print count_2, count_3, count_4


    

    # gen_retwweet(graph, )

    #[(2, 1039), (3, 325), (4, 88), (5, 30), (6, 10), (7, 2), (8, 0), (9, 0)]
    # [(2, 12949), (3, 4030), (4, 846), (5, 287), (6, 94), (7, 25), (8, 8), (9, 2)]
    # [(2, 109258), (3, 36282), (4, 9817), (5, 2391), (6, 1018), (7, 251), (8, 100), (9, 23)]
