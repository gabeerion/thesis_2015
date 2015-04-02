#! /usr/bin/python

import sys, math, argparse, os, sys
import numpy as np
from matplotlib import pyplot as plt
import mcmc as m
import hc_opt as h

ALIGNFILE = 'm14small.csv'
OUTFILE = 'csubacctest_subsample.csv'
REPS = 100
ALLEN = 200
SUBLEN = 150
SEQLEN = 1797

dataset = np.genfromtxt(ALIGNFILE, delimiter=',').astype(int)
m.BOOTREPS = 10
results = []
for i in xrange(REPS):
	# Generate a dataset to test on
#	dataset = al[np.random.choice(xrange(al.shape[0]),ALLEN,replace=0)][:,np.random.choice(xrange(al.shape[1]),SEQLEN,replace=0)]
	trueclust = m.clust(dataset)

	subsample = dataset[np.random.choice(xrange(dataset.shape[0]),SUBLEN,replace=0)]
	subclust = m.clust(subsample)

	imputed = m.impute.impute(subsample, ALLEN-SUBLEN)
	impclust = m.clust(imputed)
	impclass = m.boot_pd(m.impute.pdn(imputed),SUBLEN,SEQLEN)[0]

	results.append((trueclust,subclust,impclust,impclass,np.abs(impclass-subclust),np.abs(impclust-trueclust),(impclust-trueclust)**2))

results = np.array(results)
np.savetxt(OUTFILE,results)

print np.corrcoef(results[:,-3],results[:,-2])

plt.scatter(results[:,-3],results[:,-2], color='blue', alpha=0.2)
plt.show()