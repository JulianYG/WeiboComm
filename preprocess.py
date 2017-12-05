import pickle
import snap
import datetime

from cem import Config


# RetweetDict: {A:[(tweet1_time,{B:retweet_time})]}
def constructRetweetDict(retweet_file):
	retweets = {}
	retweet_people = {}
	row_number = 0
	original_time = None
	original_uid = None
	retweet_num = None
	with open(retweet_file) as f:
		for line in f:
			row_number += 1
			if row_number%10000 == 0:
				print row_number
			elements = line.split()
			if row_number % 2 == 1:
				original_time = parseTime(elements[1])
				original_uid = int(elements[2])
				retweet_num = int(elements[3])
			else:
				uid_time = {}
				people = set([])
				for i in range(0, len(elements), 2):
					uid = int(elements[i])
					time = parseTime(elements[i+1])
					# ignore self edges
					if uid == original_uid:
						continue
					people.add(uid)
					if uid in uid_time:
						if time < uid_time[uid]:
							uid_time[uid] = time
					else:
						uid_time[uid] = time
				if original_uid in retweets:
					retweets[original_uid].append((original_time, uid_time))
					retweet_people[original_uid] |= people
				else:
					retweets[original_uid] = [(original_time, uid_time)]
					retweet_people[original_uid] = people
	return retweets, retweet_people

# parse timestamp like 2012-08-15-20:07:32
def parseTime(timestamp):
	return datetime.datetime.strptime(timestamp, '%Y-%m-%d-%H:%M:%S')


def dfs(curr, source, retweet_people_source, Graph, path, paths, visited, retweet_info, max_path_len):
	if curr in retweet_people_source:
		if (source, curr) not in paths:
			paths[(source, curr)] = []
		newpath = path+[curr]
		missing, conflict, correct = pathStats(newpath, retweet_info)
		# missing, conflict, correct = 0,0,0
		paths[(source, curr)].append((newpath, missing, conflict, correct))
	 
	if curr in visited:
		return
	visited.add(curr)
	if len(path) >= max_path_len:
		return
	# iterate A's neighbors
	node_curr = Graph.GetNI(curr)
	for idx in range(node_curr.GetOutDeg()):

	    B = node_curr.GetOutNId(idx)
	    dfs(B, source, retweet_people_source, Graph, path+[curr], paths, visited, retweet_info, max_path_len)
	

def findPaths(Graph, paths, retweet_info, retweet_people, max_path_len):
	ori_tweeter = retweet_info.keys()
	for tweeter in ori_tweeter:
		if not Graph.IsNode(tweeter):
			continue
		visited = set([])
		path = []
		# print retweet_people[tweeter]
		dfs(tweeter, tweeter, retweet_people[tweeter], Graph, path, paths, visited, retweet_info, max_path_len)


def pathStats(path, retweet_info):
	missing = 0
	conflict = 0
	correct = 0
	retweet_info_source = retweet_info[path[0]]
	for tup in retweet_info_source:
		time, uid_time = tup
		last_time = time
		for i in range(1,len(path)):
			if path[i] not in uid_time:
				missing += 1
			else:
				if i > 1 and uid_time[path[i]] <= last_time:
					conflict += 1
				else:
					correct += 1
				last_time = uid_time[path[i]]
	return missing, conflict, correct



# ########################################### toy graph to test 
# ######## test dfs(curr, source, retweet_people_source, Graph, path, paths, visited, retweet_info)
# graph = snap.PNGraph.New()
# for i in range(5):
# 	graph.AddNode(i+1)
# graph.AddEdge(1,4)
# graph.AddEdge(1,2)
# graph.AddEdge(2,3)
# graph.AddEdge(3,1)
# graph.AddEdge(4,2)
# graph.AddEdge(4,5)

# retweet_info, retweet_people = constructRetweetDict("../data/data/toy_retweet.txt")
# # print "retweet_info"
# print retweet_info
# print "retweet_people"
# print retweet_people

# paths = {}
# findPaths(graph, paths, retweet_info, retweet_people)
# print paths
# ########################################## toy example end 

if __name__ == '__main__':

	conf, paths = Config(), {}

	max_path_len = conf.max_path_len

	print('processing retweet file...')
	retweet_info, retweet_people = constructRetweetDict(conf.retweet_file)

	print('loading following graph...')
	Graph = snap.LoadEdgeList(snap.PNGraph, conf.network_file)

	print('finding paths...')
	findPaths(Graph, paths, retweet_info, retweet_people, max_path_len)

	# save pickle
	print('saving the whole dictionary...')
	with open(conf.path_dict, 'wb') as handle:
		pickle.dump(paths, handle, protocol=pickle.HIGHEST_PROTOCOL)


	# # load pickle
	# with open('paths.pickle', 'rb') as handle:
	# 	paths = pickle.load(handle)


