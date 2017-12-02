import pickle
import snap
import datetime

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
					retweet_people[original_uid].append(people)
				else:
					retweets[original_uid] = [(original_time, uid_time)]
					retweet_people[original_uid] = [people]
	return retweets, retweet_people

# parse timestamp like 2012-08-15-20:07:32
def parseTime(timestamp):
	year, month, day, time = timestamp.split('-')
	hour, minute, second = time.split(':')
	t = datetime.datetime(int(year), int(month), int(day), int(hour), int(minute), int(second))
	# print t
	return t


def findPaths(A, retweet_people_A, Graph):
	paths = set([])


	return paths

def pathStats(path, retweet_info_A):
	missing = 0
	conflict = 0
	connect = 0

	return missing, conflict, connect




retweet_file = "../data/data/total.txt"
following_file = "../data/data/network_graph_small.txt"

# retweet_info = constructRetweetDict(retweet_file)

# Graph = snap.LoadEdgeList(snap.PNGraph, "../data/data/network_graph_small.txt")
# print Graph.GetNodes()
# print Graph.GetEdges()

# # save pickle
# with open('paths.pickle', 'wb') as handle:
# 	pickle.dump(paths, handle, protocol=pickle.HIGHEST_PROTOCOL)


# # load pickle
# with open('paths.pickle', 'rb') as handle:
# 	paths = pickle.load(handle)

