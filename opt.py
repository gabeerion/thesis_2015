#! /usr/bin/python
# testcc.py -- tests adaptive congruency class calculations

import math
import numpy as np
from scipy import stats
from scipy.stats import kstest as ks
from matplotlib import pyplot as plt

def main():
	dist = stats.beta(10,5)
	steps = 100000
	size = 100
	alpha = 0.05
	changes = int(alpha*size)
	savefile = 'states.csv'
	plotfile = 'dists.png'

#	distrvs = dist.rvs(size)
	current = np.random.random(size)
	cur_ks = ks(current, dist.cdf)[1]

	states = np.zeros((steps,size))

	for i in xrange(steps):
		prop = np.copy(current)
		prop[np.random.choice(range(size),changes)] = np.random.random(changes)
		prop_ks = ks(prop,dist.cdf)[1]
		diff = prop_ks-cur_ks
		if diff>0:
			current = prop
			cur_ks = prop_ks
		states[i] = current
		print cur_ks

	np.savetxt(savefile, states)

if __name__ == '__main__': main()