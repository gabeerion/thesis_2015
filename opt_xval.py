#! /usr/bin/python

import sys, math, argparse, os, sys
import numpy as np
from matplotlib import pyplot as plt
import mcmc as m
import hc_opt as h

ALIGNFILE = '50s_medsub.csv'
OUTFILE = 'opt_xval_results.csv'
REPS = 10
SUBLEN = 30
SEQLEN = 1797
TARGETREPS = 1000

m.THRESHOLD = 0.12
h.m.THRESHOLD = 0.12
dataset = np.genfromtxt(ALIGNFILE, delimiter=',').astype(int)
trueclust = m.clust(dataset)
ALLEN = dataset.shape[0]

header = ['truth','subclust','opt_est','opt_5p','opt_95p','imp_est','imp_5p','imp_95p', 'optcov', 'impcov']
results = []
for i in xrange(REPS):
	print i
	sub = dataset[np.random.choice(xrange(ALLEN),SUBLEN,replace=0)]
	subclust = m.clust(sub)

	imp = h.multimpute(sub,ALLEN-SUBLEN)

	opt_target = h.build_target(sub,TARGETREPS,'tempoutfile')
	opt = h.opt_ci(sub,ALLEN-SUBLEN, opt_target, steps=1000)

	r = [trueclust, subclust, opt[0], opt[1], opt[2], imp[0], imp[1], imp[2], opt[1]<=trueclust<=opt[2], imp[1]<=trueclust<=imp[2]]
	results.append(r)
results = np.array(results)
np.savetxt(OUTFILE,results)
print ['Sub MSE', 'Imp MSE', 'Opt MSE', 'Imp Cov', 'Opt Cov']
print [np.mean((results[:,0]-results[:,1])**2),np.mean((results[:,0]-results[:,5])**2),np.mean((results[:,0]-results[:,2])**2), np.sum(results[:,-1])/float(results.shape[0]),np.sum(results[:,-2])/float(results.shape[0])]