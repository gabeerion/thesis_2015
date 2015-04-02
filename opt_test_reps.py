#! /usr/bin/python

import sys, math, argparse, os, sys
import numpy as np
from matplotlib import pyplot as plt
import mcmc as m
import hc_opt as h

USAGE = "Usage: test_reps.py align_file number_of_repetitions subsamples_per_repetition"
IMPUTATIONS = 10
STEPS = 1000

def main():
	m.THRESHOLD = .12
	h.m.THRESHOLD = .12
	# Parse input
	parser = argparse.ArgumentParser(description='validate MCMC performance')
	parser.add_argument('alignfile', help='file containing alignment to use')
	parser.add_argument('-n', '--numseqs', type=int, help='number of sequences to use in subsample', required=True)
	parser.add_argument('-m', '--numsites', type=int, help='number of sites to include in subsample', required=True)
	parser.add_argument('-q', '--subseqs', type=int, help='number of sequences to subsample to', required=True)
	parser.add_argument('-r', '--repetitions', type=int, help='number of repetitions to do', required=True)
	parser.add_argument('-s', '--subsamples', type=int, help='number of subsamples per repetition', required=True)
	args = parser.parse_args()
	try: al = np.genfromtxt(args.alignfile, delimiter=',').astype(int)
	except ValueError: print 'Invalid alignment file'; exit()
	reps = args.repetitions
	subsamples = args.subsamples
	allen = args.numseqs
	seqlen = args.numsites
	subseqs = args.subseqs

	fname = args.alignfile[:-4]+'_tr'
	imps = allen-subseqs
	devnull = open(os.devnull, 'w')

	print """
Loaded data:
Full alignment is %dx%d sequences & sites.
Randomly selected datasets will be %dx%d sequences and sites.
Subsamples will be %dx%d sequences and sites.
There will be %d repetitions of %d subsamples each.\n
	""" % (al.shape[0], al.shape[1], allen, seqlen, subseqs, seqlen, reps, subsamples)

	results = []
	print 'dset\tsubset\ttrue_clust\tsub_clust\tmcmc_clust\timp_clust'
	# Main loop
	for i in xrange(reps):
		np.random.seed(i)
		# Generate a dataset to test on
		dataset = al[np.random.choice(xrange(al.shape[0]),allen,replace=0)][:,np.random.choice(xrange(al.shape[1]),seqlen,replace=0)]
#		dataset = al[np.random.choice(xrange(al.shape[0]),allen,replace=0)]
		trueclust = m.clust(dataset)
		for j in xrange(subsamples):
			np.random.seed((i,j))
			# Generate a subsample of the current dataset to simulate missingness
			subsample = dataset[np.random.choice(xrange(dataset.shape[0]),subseqs,replace=0)]
			subclust = m.clust(subsample)

			# Attempt to recover with imputation
			imp = h.multimpute(subsample,imps)
#			imputed_states = [m.impute.impute(subsample, imps) for k in xrange(IMPUTATIONS)]
#			imputed_clusts = map(m.clust, imputed_states)
#			avg_impclust = np.mean(imputed_clusts)

			# Attempt to recover with KS optimization
			"""m.V_TDIST, m.V_STATES = '%s_%d_%d_target.csv' % (fname,i,j), '%s_%d_%d_states.csv' % (fname,i,j)
			sys.stdout, sys.stderr = devnull, devnull
			states, tdist = m.mcmc_ns(al=subsample, imps=imps)
			sys.stdout, sys.stderr = sys.__stdout__, sys.__stderr__"""
			tdist = h.build_target(subsample, 100,'%s_%d_%d_target.csv' % (fname,i,j))
#			cc, clust, ks = h.opt(subsample,imps,tdist)
			opt = h.opt_ci(subsample,imps,tdist, steps=STEPS)
			
			"""# Plot results
			cclass_hist = plt.hist(cc,normed=1,alpha=.5, label='Sampled congruency classes', color='green')
#			xr = np.linspace(np.min(states[:,2]),np.max(states[:,2]),1000)
			xr = np.linspace(0,1,1000)
			tpdf = tdist.pdf(xr)
			plt.plot(xr,tpdf, label='Target distribution', color='blue')
			ymax = np.max([np.max(cclass_hist[0]), np.max(tpdf)])
			plt.plot((subclust,subclust),(0,ymax),label='Observed clustering value (congruency class)', color='blue')
			mean_cclass = np.mean(cc)
			plt.plot((mean_cclass,mean_cclass),(0,ymax),label='MCMC average congruency class', color='green')
			plt.legend()
			plt.savefig('%s_%d_%d_cclass.png' % (fname,i,j))
			plt.clf()

			clust_hist = plt.hist(clust,normed=1,alpha=.5, label='Sampled clustering values', color='green')
			ymax = np.max(clust_hist[0])
			plt.plot((subclust,subclust),(0,ymax),label='Observed clustering value', color='blue')
			plt.plot((trueclust,trueclust),(0,ymax),label='True clustering value', color='red')
			plt.plot((avg_impclust,avg_impclust),(0,ymax),label='Imputed point estimate (mean)', color='purple')
			mean_clust = np.mean(clust)
			plt.plot((mean_clust,mean_clust),(0,ymax),label='Mean MCMC estimate', color='green')
			plt.legend()
			plt.savefig('%s_%d_%d_clust.png' % (fname,i,j))
			plt.clf()"""
#			results.append([i,j,trueclust,subclust,mean_clust,avg_impclust])
			results.append([i,j,trueclust,subclust,opt[0],opt[1],opt[2],imp[0],imp[1],imp[2],opt[1]<=trueclust<=opt[2], imp[1]<=trueclust<=imp[2]])
#			print '%d\t%d\t%.2f\t\t%.2f\t\t%.2f\t\t%.2f' % (i,j,trueclust,subclust,mean_clust,avg_impclust)
			print '%d\t%d\t%.2f\t\t%.2f\t\t%.2f\t\t%.2f' % (i,j,trueclust,subclust,opt[0],imp[0])

	np.savetxt('%s_summary.csv'%fname, results, delimiter=',', fmt='%s')
	results = np.array(results)
	print [np.mean((results[:,2]-results[:,3])**2),np.mean((results[:,2]-results[:,7])**2),np.mean((results[:,2]-results[:,4])**2), np.sum(results[:,-1])/float(results.shape[0]),np.sum(results[:,-2])/float(results.shape[0])]

if __name__ == '__main__': main()