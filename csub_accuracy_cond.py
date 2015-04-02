from matplotlib import pyplot as plt
import numpy as np
import mcmc as m
import impute
import itertools, sys

m.THRESHOLD = 0.12
small = np.genfromtxt('m14small.csv', delimiter=',').astype('int')
if len(sys.argv)!=2: print 'Only one argument taken (filename)'; exit()
fname = sys.argv[-1]
#print small.shape

INITSIZE = small.shape[0]
STARTSIZE = 50
SUBSIZE = 30
IMPNUM = 100
MEDSAMPLE = 100
FRAMEDIM = 10

# Set up plot
fig, axarr = plt.subplots(FRAMEDIM,FRAMEDIM)
for i in range(FRAMEDIM):
	for j in range(FRAMEDIM):
		axarr[i,j].set_xticks([0.0,1.0])
		axarr[i,j].set_yticks([0.0,1.0])
plt.setp([a.get_xticklabels() for r in axarr[0:-1, :] for a in r], visible=False)
plt.setp([a.get_yticklabels() for r in axarr[:, 1:] for a in r], visible=False)


# Choose a median-looking dataset to examine
subsamples = [small[np.random.choice(xrange(INITSIZE),STARTSIZE,replace=0)] for i in xrange(MEDSAMPLE)]
subclusts = map(m.clust, subsamples)
index = np.argsort(subclusts)[len(subclusts)/2]
sub = subsamples[index]
trueclust = subclusts[index]
##print np.array(sorted(subclusts))
print sub.shape
print trueclust

subdevs = []
truedevs = []
correlations=[]
for frame in itertools.product(range(FRAMEDIM),range(FRAMEDIM)):
	print frame,
	subsample = sub[np.random.choice(xrange(STARTSIZE),SUBSIZE,replace=0)]
	subclust = m.clust(subsample)
	print subsample.shape,	
	print subclust,

	imps = [impute.impute(subsample,STARTSIZE-SUBSIZE) for k in xrange(IMPNUM)]
	print imps[0].shape
	clusts = map(m.clust, imps)
	classes = [m.exact_boot(impute.pdn(imp),subsample.shape[0],subsample.shape[1]) for imp in imps]
	subdev = np.abs(np.array(classes)-subclust)
	truedev = np.abs(np.array(clusts)-trueclust)
	corr = np.corrcoef(subdev,truedev)[0,1]

	correlations.append(corr)
	subdevs.append(subdev)
	truedevs.append(truedev)

for i in xrange(len(correlations)):
	if np.isnan(correlations[i]): correlations[i]=-1.1
correlations = np.array(correlations)

#	plt.title('Values of c_sub closer to observed clustering value yield more accurate estimates')
argsorted = np.argsort(correlations)[::-1]
for i, frame in enumerate(itertools.product(range(FRAMEDIM),range(FRAMEDIM))):
	index = argsorted[i]
	axarr[frame[0],frame[1]].scatter(subdevs[index],truedevs[index], alpha=0.3, color='blue')
#	axarr[frame[0],frame[1]].title('r=%.2f' % corr)
cleancorr = correlations[np.where(correlations>=-1)]
print np.mean(cleancorr), np.median(cleancorr)
np.savetxt(fname+'_cac_correlations.csv', correlations)
np.savetxt(fname+'_cac_subdevs.csv', subdevs)
np.savetxt(fname+'_cac_truedevs.csv', truedevs)

plt.savefig(fname+'.png')
plt.show()