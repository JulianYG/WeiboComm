

# follow_graph: get number of followees 

# Dict: sigma, mu for each node

# retweet_graph: time_stamps

import snap as s 
import numpy as np 


class Graphize(object):


	def __init__(self, network_g, retweet_g):

		self._network = network_g
		self._retweet = retweet_g


	def seed(n):
		np.random.seed(n)

	def assign_prob(sigma, mu):


	def 


if __name__ == '__main__':

	# load graphs 

	

	mu, sigma = 0, 0.1
	eps = 1e10

	while eps > 1e-3:


		

		for _ in :

			x = Graphize( a, b)
			x.assign_prob(mu, sigma)

			...

			score = x.evaluate()

			generate_mu, sigma = np.random.sigma()

			x.assign_prob



