from matplotlib import pyplot as plt
import numpy as np
import mcmc as m
import impute
import itertools

m.THRESHOLD = 0.12
small = np.genfromtxt('m14small.csv', delimiter=',').astype('int')
print small.shape
fig, axarr = plt.subplots(10,10)
for i in range(10):
	for j in range(10):
		axarr[i,j].set_xticks([0.0,1.0])
		axarr[i,j].set_yticks([0.0,1.0])
plt.setp([a.get_xticklabels() for r in axarr[0:-1, :] for a in r], visible=False)
plt.setp([a.get_yticklabels() for r in axarr[:, 1:] for a in r], visible=False)

subsamples = [small[np.random.choice(xrange(200),50,replace=0)] for i in xrange(10)]
subclusts = map(m.clust, subsamples)
#index = np.argsort(subclusts)[len(subclusts)/2]
#sub = subsamples[index]
#trueclust = m.clust(sub)
#print np.array(sorted(subclusts))
#print sub.shape
#print trueclust

subdevs = []
truedevs = []
correlations=[]
#for frame in itertools.product(range(10),range(10)):
for i in xrange(10):
	truth = subsamples[i]
	trueclust = subclusts[i]
	rowsd = []
	rowtd = []
	rowc=[]
	for j in xrange(10):
		print i,j
		subsample = truth[np.random.choice(xrange(50),30,replace=0)]
		subclust = m.clust(subsample)
	#	print subsample.shape	
	#	print subclust

		imps = [impute.impute(subsample,20) for k in xrange(10)]
		clusts = map(m.clust, imps)
		classes = [m.exact_boot(impute.pdn(imp),subsample.shape[0],subsample.shape[1]) for imp in imps]
		subdev = np.abs(np.array(classes)-subclust)
		truedev = np.abs(np.array(clusts)-trueclust)
		corr = np.corrcoef(subdev,truedev)[0,1]

		correlations.append(corr)
		subdevs.append(subdev)
		truedevs.append(truedev)

		rowsd.append(subdev)
		rowtd.append(truedev)
		rowc.append(corr)
	rowc, rowtd, rowsd = np.array(rowc), np.array(rowtd), np.array(rowsd)
	rowc[np.isnan(rowc)] = -1.1
	argsorted = np.argsort(rowc[::-1])
	for j in xrange(10):
		index = argsorted[j]
		axarr[i,j].scatter(rowsd[index],rowtd[index],alpha=0.3,color='blue')

fig, axarr = plt.subplots(10,10)
for i in range(10):
	for j in range(10):
		axarr[i,j].set_xticks([0.0,1.0])
		axarr[i,j].set_yticks([0.0,1.0])
plt.setp([a.get_xticklabels() for r in axarr[0:-1, :] for a in r], visible=False)
plt.setp([a.get_yticklabels() for r in axarr[:, 1:] for a in r], visible=False)

for i in xrange(len(correlations)):
	if np.isnan(correlations[i]): correlations[i]=-1.1
correlations = np.array(correlations)

#	plt.title('Values of c_sub closer to observed clustering value yield more accurate estimates')
argsorted = np.argsort(correlations)[::-1]
for i, frame in enumerate(itertools.product(range(10),range(10))):
	index = argsorted[i]
	axarr[frame[0],frame[1]].scatter(subdevs[index],truedevs[index], alpha=0.3, color='blue')
#	axarr[frame[0],frame[1]].title('r=%.2f' % corr)
cleancorr = correlations[np.where(correlations>=-1)]
print np.mean(cleancorr), np.median(cleancorr)
np.savetxt('cac_correlations.csv', correlations)
np.savetxt('cac_subdevs.csv', subdevs)
np.savetxt('cac_truedevs.csv', truedevs)

plt.show()

[-0.473671610935,0.640385452186,-0.584067063847,0.646209196195,0.595124054508,0.976674262592,0.985012571981,0.922373274766,0.900492998183,]