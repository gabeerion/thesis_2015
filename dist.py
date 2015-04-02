#! /usr/bin/python

import sys
sys.path.insert(0, '/home/gabe/mcmc')

import mcmc as m
import scratch as s
import numpy as np
from Bio import AlignIO

SAVECSV = 1
OUTFILE = 'cdist.csv'

clusts = []
for i in xrange(51):
	if i == 26: continue
	al = AlignIO.read('%d.fasta'%i, 'fasta')
	arr = s.arint(np.array(al))
	c = m.clust(arr)
	if SAVECSV: np.savetxt('%d.csv'%i, arr)
	clusts.append(c)
np.savetxt(OUTFILE, clusts)
