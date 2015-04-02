#! /usr/bin/python

import sys, math, argparse, os, sys
import numpy as np
from matplotlib import pyplot as plt
import mcmc as m
import hc_opt as h

ALIGNFILE = 'm14small.csv'
OUTFILE = 'csubacctest.csv'
REPS = 1000
ALLEN = 200
SUBLEN = 150
SEQLEN = 1797
BOOTREPS = (10,100,1000,10000,100000)

def boot_pd(pd,dellen,seqlen, bootreps):
	subclusts = []
	for i in xrange(bootreps):
		inds = np.random.choice(xrange(pd.shape[0]),dellen,replace=0)
		subclusts.append(m.pdclust(pd[inds][:,inds], seqlen))
	return np.array(subclusts)

dataset = np.genfromtxt(ALIGNFILE, delimiter=',').astype(int)
pd = m.impute.pdn(dataset)
results = []
cclass = m.exact_boot(pd, SUBLEN, SEQLEN)
for b in BOOTREPS:
	approx = boot_pd(pd,SUBLEN,SEQLEN,b)
	results.append(approx)

print cclass
for r in results:
	print len(r), np.mean(r)
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.set_xscale('log')
#plt.plot((0,100000),(cclass,cclass))
plt.scatter(*np.array(zip(BOOTREPS,np.abs(np.array(map(np.mean,results))-cclass))).transpose(), color='blue',alpha=0.5)
plt.title('Subsampling approximations converge to value of c_sub given by exact formula')
plt.xlabel('Number of subsamples used in calculation (log scale)')
plt.ylabel('abs(exact value - subsampling estimate)')
plt.show() 	