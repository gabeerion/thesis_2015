#! /usr/bin/python
# testcc.py -- tests adaptive congruency class calculations

import math
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt

def rounded(num, bins):
	return math.round(num*bins)/bins

def main():
	dist = stats.beta(10,5)
	target = stats.beta(10,20)
	pvar = 0.01
	steps = 100000
	bins = 10
	savefile = 'states.csv'
	plotfile = 'dists.png'

	prev = np.random.random()

	states = []
	counts = np.zeros(bins)
	counts[rounded(prev,bins)] += 1
	for i in xrange(steps):
		cur = stats.norm(prev,pvar).rvs()
		counts[rounded(cur,bins)] += 1
		counts /= np.sum(counts)
		curlik = dist.pdf(cur)*target.pdf(cur)
		a = dist.pdf(cur)/dist.pdf(prev)
		if a > np.random.random(): prev = cur
		states.append(prev)
	np.savetxt(savefile, states)

	xr = np.linspace(0,1,1000)
	plt.plot(xr,dist.pdf(xr))
	plt.hist(states, alpha=.5, normed=1)
	plt.savefig(plotfile)





if __name__ == '__main__': main()